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
from app.retrieval.services.usage_recorder import UsageRecorder

logger = logging.getLogger(__name__)

_MAX_HISTORY_TURNS = 8
_USAGE_COMMENT_RE = re.compile(r"\n?<!-- usage \{.*?\} -->\s*$")


def _strip_usage_comment(text: str) -> str:
    """Removes the `<!-- usage {...} -->` HTML comment compose_answer/finalize
    append to final_response (Iter.-1 telemetry marker) — must not leak into the
    persisted/user-facing assistant_message."""
    return _USAGE_COMMENT_RE.sub("", text or "")


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
    usage_recorder: Any = None,
) -> AsyncIterator[dict]:
    """SSE-Event-Generator für POST /app/chat/stream.

    Yields sse-starlette-kompatible dicts: {"data": json.dumps(...)}.
    Contract §5/§6: status / token / thinking / done / error.
    """
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
                        "tool_results":      [],
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
    usage_recorder: Any = None,
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
        usage_recorder=usage_recorder,
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
