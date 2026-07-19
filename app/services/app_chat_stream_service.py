"""App-Adapter für /app/chat/stream (und /app/chat non-stream, Fallback).

Teilt sich den RAG-Graph-Kern (`assistant_chat_graph`) mit dem Agent-Pfad
(`app/api/chat.py`), aber mit App-Persistenz (`rag_talks`/`rag_turns`/
`rag_references`) statt LangGraph-Checkpoint. Siehe
ragapp/plans/filo-chat-ui-design.md §7.1 und filo-arbeitstext-contract.md §3/§5/§6.
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any, AsyncIterator, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.config import settings
from app.db.ports import TalksPort
from app.retrieval.services.action_prompt_service import load_assistant_rag_collection
from app.retrieval.services.providers import get_deepseek_chat
from app.retrieval.services.usage_recorder import UsageRecorder
from app.retrieval.utils.token_estimate import (
    DEEPSEEK_CONTEXT_LIMIT_TOKENS,
    estimate_context_tokens,
)
from app.tools.prompts import DOCUMENT_EDITOR_SYSTEM_PROMPT
from app.tools.types import ToolContext, ToolResult

logger = logging.getLogger(__name__)

_MAX_HISTORY_TURNS = 8
_USAGE_COMMENT_RE = re.compile(r"\n?<!-- usage \{.*?\} -->\s*$")


async def _resolve_context_paragraph_text(
    *,
    explicit: str | None,
    context_ids: dict[str, Any] | None,
) -> str:
    """Absatztext für „Philo fragen“: Client-Text bevorzugt, sonst Lookup in rag_paragraphs."""
    text = (explicit or "").strip()
    if text:
        return text
    pid = str((context_ids or {}).get("paragraph_id") or "").strip()
    if not pid:
        return ""
    try:
        import asyncio

        from sqlalchemy import text as sql_text

        from app.db.session import get_engine

        def _read() -> str:
            with get_engine().connect() as conn:
                row = conn.execute(
                    sql_text(
                        """
                        SELECT text_raw, segment_title, paragraph_number
                        FROM rag_paragraphs
                        WHERE id = CAST(:id AS uuid)
                          AND deprecated_at IS NULL
                        LIMIT 1
                        """
                    ),
                    {"id": pid},
                ).mappings().first()
            if not row:
                return ""
            body = str(row["text_raw"] or "").strip()
            if not body:
                return ""
            number = row.get("paragraph_number")
            title = str(row.get("segment_title") or "").strip()
            header_parts: list[str] = []
            if number is not None:
                header_parts.append(f"Absatz {number}")
            if title:
                header_parts.append(title)
            header = " · ".join(header_parts)
            return f"{header}\n\n{body}" if header else body

        return await asyncio.to_thread(_read)
    except Exception:
        logger.exception("Failed to load paragraph context for %s", pid)
        return ""


def _strip_usage_comment(text: str) -> str:
    """Removes the `<!-- usage {...} -->` HTML comment compose_answer/finalize
    append to final_response (Iter.-1 telemetry marker) — must not leak into the
    persisted/user-facing assistant_message."""
    return _USAGE_COMMENT_RE.sub("", text or "")


def _tool_loop_system_prompt(document_outline: dict[str, Any] | None) -> str:
    """Contract §3: Outline gehört zum Tool-Loop-Kontext (nicht in den RAG-Prompt)."""
    prompt = DOCUMENT_EDITOR_SYSTEM_PROMPT
    if document_outline:
        prompt += (
            "\n\nAktuelle Dokument-Gliederung "
            "(nutze `paragraph_id` bzw. `heading_path` aus dieser Outline für `update_document`):\n"
            + json.dumps(document_outline, ensure_ascii=False)
        )
    return prompt


def _thinking_type_for_mode(mode: str | None) -> str:
    """Filo §7.2: `mode` → `thinking.type` (`chat` → disabled, `nachdenken` → enabled)."""
    return "enabled" if mode == "nachdenken" else "disabled"


def _history_messages_from_turns(
    rows: Sequence[dict[str, Any]], *, max_turns: int = _MAX_HISTORY_TURNS
) -> list[BaseMessage]:
    """Rebuilds the graph's `messages` state from `rag_turns` rows (App-Chat has
    no LangGraph-Checkpoint — `rag_turns` is the canonical history, Filo §7.1)."""
    recent = list(rows)[-max_turns:]
    messages: list[BaseMessage] = []
    for row in recent:
        user_msg = row.get("user_message")
        assistant_msg = row.get("assistant_message")
        if user_msg:
            messages.append(HumanMessage(content=user_msg))
        if assistant_msg:
            messages.append(AIMessage(content=assistant_msg))
    return messages


async def _run_graph_events(graph: Any, initial_state: dict, config: dict) -> AsyncIterator[dict]:
    """Runs the graph and yields normalized events (not yet SSE-JSON-wrapped):

        {"kind": "status", "step", "label"}
        {"kind": "thinking", "content"}
        {"kind": "token", "content"}
        {"kind": "final", "final_response", "citations", "confidence_score",
                           "intent", "sufficiency", "usage"}
    """
    tokens_sent = False

    async for event in graph.astream_events(initial_state, config, version="v2"):
        kind = event.get("event")

        if kind == "on_custom_event" and event.get("name") == "ace_progress":
            payload = event.get("data") or {}
            yield {"kind": "status", "step": payload.get("step", ""), "label": payload.get("label", "")}

        elif kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            reasoning = (chunk.additional_kwargs or {}).get("reasoning_content")
            if reasoning:
                yield {"kind": "thinking", "content": reasoning}
            if chunk.content:
                tokens_sent = True
                yield {"kind": "token", "content": chunk.content}

        elif kind == "on_chain_end" and event.get("name") == "finalize":
            output = event["data"].get("output") or {}
            response_text = _strip_usage_comment(output.get("final_response", ""))
            if response_text and not tokens_sent:
                yield {"kind": "token", "content": response_text}
            yield {
                "kind":             "final",
                "final_response":   response_text,
                "citations":        output.get("citations", []),
                "confidence_score": output.get("confidence_score", 0.0),
                "intent":           output.get("intent", ""),
                "sufficiency":      output.get("sufficiency", ""),
                "usage":            output.get("usage_metadata"),
            }


async def _run_tool_loop(
    registry: Any,
    ctx: ToolContext,
    tools_schema: list[dict[str, Any]],
    seed_messages: list[dict[str, Any]],
    *,
    max_rounds: int = 2,
) -> list[ToolResult]:
    """Runs the App-Tool function-calling loop (Contract §6) after the RAG
    response. `read_blocks` results are fed back as tool messages for a
    follow-up round; any other tool call ends the loop."""
    if not tools_schema:
        return []

    messages = list(seed_messages)
    results: list[ToolResult] = []
    deepseek = get_deepseek_chat()

    for round_index in range(max_rounds):
        completion = await deepseek.chat(
            messages,
            max_tokens=2200,
            tools=tools_schema,
            tool_choice="auto",
            _debug_caller="app_tool_loop",
        )
        tool_calls = completion.tool_calls
        if not tool_calls:
            break

        messages.append({
            "role": "assistant",
            "content": completion.content or "",
            "tool_calls": tool_calls,
        })

        read_blocks_called = False
        for call in tool_calls:
            function = call.get("function") or {}
            tool_id = function.get("name")
            try:
                args = json.loads(function.get("arguments") or "{}")
            except (TypeError, ValueError):
                logger.warning(
                    "App tool call arguments not valid JSON (tool=%s, likely truncated by "
                    "max_tokens): %r", tool_id, function.get("arguments"),
                )
                args = {}

            try:
                result = await registry.invoke(tool_id, ctx, args)
            except Exception:
                logger.exception("App tool invocation failed: %s", tool_id)
                continue

            results.append(result)
            messages.append({
                "role": "tool",
                "tool_call_id": call.get("id", ""),
                "content": json.dumps(result.payload),
            })

            if tool_id == "read_blocks":
                read_blocks_called = True

        if not read_blocks_called or round_index == max_rounds - 1:
            break

    return results


async def stream_app_chat(
    graph: Any,
    talks: TalksPort,
    *,
    user_id: str,
    user_name: str,
    message: str,
    personality: str,
    talk_id: str | None = None,
    mode: str | None = None,
    model: str | None = None,
    context_mode: str | None = None,
    context_ids: dict[str, Any] | None = None,
    context_paragraph_text: str | None = None,
    usage_recorder: Any = None,
    linked_document_id: str | None = None,
    document_outline: dict[str, Any] | None = None,
    linked_document_content: str | None = None,
    app_tool_registry: Any = None,
) -> AsyncIterator[dict]:
    """SSE-Event-Generator für POST /app/chat/stream.

    Yields sse-starlette-kompatible dicts: {"data": json.dumps(...)}.
    Contract §5/§6: status / token / thinking / done / error.
    """
    # context_mode reserved for future retrieval scoping (paragraph/segment/free).
    _ = context_mode
    usage_recorder = usage_recorder or UsageRecorder()
    msg = message.strip()
    if not msg:
        raise ValueError("message must not be empty")
    personality_slug = personality.strip()
    if not personality_slug:
        raise ValueError("personality must not be empty")

    assistant_slug = settings.app_default_assistant_slug
    collection = load_assistant_rag_collection(assistant_slug)
    thinking_type = _thinking_type_for_mode(mode)

    history_rows = await talks.load_talk_turns(talk_id) if talk_id else []
    history_messages = _history_messages_from_turns(history_rows)

    ctx = context_ids or {}
    kontext_meta = ctx if ctx else None
    paragraph_text = await _resolve_context_paragraph_text(
        explicit=context_paragraph_text,
        context_ids=context_ids,
    )

    initial_state: dict[str, Any] = {
        "assistant_slug":   assistant_slug,
        "collection_name":  collection,
        "user_message":     msg,
        "messages":         history_messages,
        "retry_count":      0,
        "lemma_found":      False,
        "extracted_lemma":  "",
        "retrieval_plan":   [],
        "context_text":     "",
        "context_refs":     [],
        "sufficiency":      "",
        "citations":        [],
        "final_response":   "",
        "confidence_score": 0.0,
        "account_id":       user_id,
        "model":            model,
        "thinking_type":    thinking_type,
        # App-Chat bucht Usage selbst (nach create_talk_turn, mit echter
        # turn_id/talk_id-Verknüpfung) — unterdrückt die graph-interne,
        # thread_id-only Buchung in compose_answer/finalize.
        "skip_usage":       True,
        # Arbeitstext für compose_answer (nicht nur Tool-Loop) — Contract: ≤50k.
        "linked_document_content": linked_document_content or "",
        # Bezugstext aus „Philo fragen“ (Absatz).
        "context_paragraph_text": paragraph_text,
    }
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    try:
        async for ev in _run_graph_events(graph, initial_state, config):
            k = ev["kind"]
            if k == "status":
                yield {"data": json.dumps({"type": "status", "step": ev["step"], "label": ev["label"]})}

            elif k == "thinking":
                yield {"data": json.dumps({"type": "thinking", "content": ev["content"]})}

            elif k == "token":
                yield {"data": json.dumps({"type": "token", "content": ev["content"]})}

            elif k == "final":
                final_response = ev["final_response"]
                resolved_model = model or settings.deepseek_chat_model or "deepseek-v4-flash"
                usage_meta = ev.get("usage") or {}
                usage_payload = (
                    {
                        "model":             resolved_model,
                        "prompt_tokens":     usage_meta.get("input_tokens"),
                        "completion_tokens": usage_meta.get("output_tokens"),
                        "total_tokens":      usage_meta.get("total_tokens"),
                    }
                    if usage_meta
                    else None
                )

                saved = await talks.create_talk_turn(
                    user_id=user_id,
                    user_name=user_name or user_id,
                    collection=collection,
                    personality=personality_slug,
                    title=msg[:120],
                    user_message=msg,
                    assistant_message=final_response,
                    talk_id=talk_id,
                    kontext_meta=kontext_meta,
                    usage=usage_payload,
                )

                citations = ev.get("citations") or []
                if citations:
                    await talks.save_turn_references(saved["turn_id"], citations)

                used_tokens = estimate_context_tokens(
                    history_rows=[
                        *history_rows,
                        {"user_message": msg, "assistant_message": final_response},
                    ],
                    citations=citations,
                    linked_document_content=linked_document_content,
                    context_paragraph_text=paragraph_text,
                )

                tool_results: list[ToolResult] = []
                if app_tool_registry is not None:
                    try:
                        available_tools = app_tool_registry.list_tools(
                            mode=mode, linked_document_id=linked_document_id
                        )
                        if available_tools:
                            tools_schema = app_tool_registry.schemas_for_llm(available_tools)
                            tool_ctx = ToolContext(
                                user_id=user_id,
                                talk_id=saved["talk_id"],
                                mode=mode,
                                personality=personality_slug,
                                linked_document_id=linked_document_id,
                                document_content=linked_document_content,
                                document_outline=document_outline,
                            )
                            seed_messages = [
                                {"role": "system", "content": _tool_loop_system_prompt(document_outline)},
                                {"role": "user", "content": msg},
                                {
                                    "role": "system",
                                    "content": (
                                        "Deine Antwort an den Nutzer im Chat (bereits gesendet):\n\n"
                                        + final_response
                                        + "\n\nPrüfe jetzt anhand der Werkzeug-Regeln oben, ob eine "
                                        "Änderung am Arbeitstext nötig ist, und rufe bei Bedarf das "
                                        "passende Werkzeug auf."
                                    ),
                                },
                            ]
                            tool_results = await _run_tool_loop(
                                app_tool_registry, tool_ctx, tools_schema, seed_messages
                            )
                    except Exception:
                        logger.warning(
                            "App tool loop failed (talk_id=%s)", saved["talk_id"], exc_info=True
                        )
                        tool_results = []

                if usage_meta:
                    await usage_recorder.record(
                        account_id=user_id,
                        model=resolved_model,
                        provider="deepseek",
                        endpoint="app_chat",
                        prompt_tokens=usage_meta.get("input_tokens"),
                        completion_tokens=usage_meta.get("output_tokens"),
                        total_tokens=usage_meta.get("total_tokens"),
                        turn_id=saved["turn_id"],
                        talk_id=saved["talk_id"],
                    )

                yield {
                    "data": json.dumps({
                        "type":              "done",
                        "turn_id":           saved["turn_id"],
                        "talk_id":           saved["talk_id"],
                        "usage":             usage_payload,
                        "assistant_message": final_response,
                        "citations":         citations,
                        "confidence_score":  ev.get("confidence_score", 0.0),
                        "intent":            ev.get("intent", ""),
                        "sufficiency":       ev.get("sufficiency", ""),
                        "context_meta":      {
                            "used_tokens":  used_tokens,
                            "limit_tokens": DEEPSEEK_CONTEXT_LIMIT_TOKENS,
                        },
                        "tool_results":      [
                            {"tool_id": r.tool_id, "result_key": r.result_key, "payload": r.payload}
                            for r in tool_results
                        ],
                    })
                }
    except Exception:
        logger.exception("Error during app chat stream (talk_id=%s)", talk_id)
        yield {"data": json.dumps({"type": "error", "message": "Interner Fehler"})}


async def run_app_chat_once(
    graph: Any,
    talks: TalksPort,
    *,
    user_id: str,
    user_name: str,
    message: str,
    personality: str,
    talk_id: str | None = None,
    mode: str | None = None,
    model: str | None = None,
    context_mode: str | None = None,
    context_ids: dict[str, Any] | None = None,
    context_paragraph_text: str | None = None,
    usage_recorder: Any = None,
    linked_document_id: str | None = None,
    document_outline: dict[str, Any] | None = None,
    linked_document_content: str | None = None,
    app_tool_registry: Any = None,
) -> dict[str, str]:
    """Non-streaming variant for POST /app/chat (3a.5: gleicher Graph-Kern wie
    /app/chat/stream, Fallback bei SSE-Fehlern). Konsumiert denselben Event-Strom
    und verwirft status/token/thinking."""
    result: dict[str, str] | None = None
    async for sse in stream_app_chat(
        graph,
        talks,
        user_id=user_id,
        user_name=user_name,
        message=message,
        personality=personality,
        talk_id=talk_id,
        mode=mode,
        model=model,
        context_mode=context_mode,
        context_ids=context_ids,
        context_paragraph_text=context_paragraph_text,
        usage_recorder=usage_recorder,
        linked_document_id=linked_document_id,
        document_outline=document_outline,
        linked_document_content=linked_document_content,
        app_tool_registry=app_tool_registry,
    ):
        payload = json.loads(sse["data"])
        if payload["type"] == "done":
            result = {
                "talk_id": payload["talk_id"],
                "turn_id": payload["turn_id"],
                "reply":   payload["assistant_message"],
            }
        elif payload["type"] == "error":
            raise RuntimeError(payload.get("message", "Interner Fehler"))

    if result is None:
        raise RuntimeError("Chat stream ended without a result")
    return result
