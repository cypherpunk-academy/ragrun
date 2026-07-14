"""API for generate-prompt / execute-prompt (ASSISTANTS_CHAT_PLAN_V2)."""
from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config import settings
from app.retrieval.services.event_recorder import EventRecorder, enqueue_record_metadata_only
from app.retrieval.services.usage_recorder import UsageRecorder, enqueue_record_usage
from app.services.pricing_service import calculate_cost
from app.retrieval.services.action_prompt_service import (
    generate_prompt_id,
    get_prompt_state,
    list_actions,
    load_action_manifest,
    load_assistant_instruction,
    load_assistant_name,
    load_assistant_rag_collection,
    run_queries_and_fill_prompt,
    store_prompt_state,
)
from app.retrieval.services.providers import get_embedding_client, get_qdrant_client, get_deepseek_chat
from app.retrieval.utils.reference_evaluator import evaluate_chunk_relevance
from app.retrieval.utils.retrievers import snippet_author
from app.retrieval.models import RetrievedSnippet

logger = logging.getLogger(__name__)

CITATION_THRESHOLD = 0.3
_DEBUG_LOG_PATH = "/Users/michael/Reniets/Ai/ragkeep/.cursor/debug-6843d9.log"


def _debug_log(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    try:
        with open(_DEBUG_LOG_PATH, "a", encoding="utf-8") as _dbg:
            _dbg.write(
                json.dumps(
                    {
                        "sessionId": "6843d9",
                        "runId": "pre-fix",
                        "hypothesisId": hypothesis_id,
                        "location": location,
                        "message": message,
                        "data": data,
                        "timestamp": int(time.time() * 1000),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    except Exception:
        pass


def _author_for_reference(meta: dict, c: dict) -> str:
    thin = {k: v for k, v in meta.items() if k != "text"}
    au = snippet_author({"payload": thin}) if thin else None
    if isinstance(au, str) and au.strip():
        return au.strip()
    ac = c.get("author")
    if isinstance(ac, str) and ac.strip():
        return ac.strip()
    return ""

_CITATION_INDEX_RE = re.compile(r"\[(\d+)\]")


def _extract_cited_indices(response_text: str) -> set[int]:
    """Extract citation indices like [1], [7], [1][3] from the response text."""
    return {int(m.group(1)) for m in _CITATION_INDEX_RE.finditer(response_text)}


def _resolve_bracket_chunk_ids(
    cited_indices: set[int],
    chunk_index_map: list[dict],
    retrieved_serialized: list[dict],
) -> set[str]:
    """Resolve bracket-cited indices to primary-source chunk_ids.

    For talk chunks: expands to their referenced primary-source chunk_ids.
    For all other chunk types: uses the chunk_id directly.
    """
    payload_by_cid: dict[str, dict] = {}
    for s in retrieved_serialized:
        pl = s.get("payload") or {}
        inner = pl.get("payload") if isinstance(pl.get("payload"), dict) else pl
        cid = inner.get("chunk_id") if isinstance(inner, dict) else None
        if isinstance(cid, str) and cid:
            payload_by_cid[cid] = pl

    index_to_entry = {e["index"]: e for e in chunk_index_map}
    result: set[str] = set()
    for idx in cited_indices:
        entry = index_to_entry.get(idx)
        if not entry:
            continue
        cid = entry.get("chunk_id")
        if not cid:
            continue
        pl = payload_by_cid.get(cid, {})
        inner = pl.get("payload", pl) if isinstance(pl.get("payload"), dict) else pl
        chunk_type = inner.get("chunk_type", "") if isinstance(inner, dict) else ""
        if chunk_type == "talk":
            refs = inner.get("references") or []
            for r in refs:
                if isinstance(r, dict):
                    ref_cid = r.get("chunk_id")
                    if isinstance(ref_cid, str) and ref_cid:
                        result.add(ref_cid)
        else:
            result.add(cid)
    return result


# Reference text: chars per ref; keep in sync with action_prompt_service.CHUNK_TEXT_MAX_CHARS
REFERENCE_TEXT_MAX_CHARS = 2000

def _make_llm(streaming: bool = True) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.deepseek_chat_model or "deepseek-v4-flash",
        openai_api_key=settings.deepseek_api_key,
        openai_api_base=f"{str(settings.deepseek_base_url).rstrip('/')}/",
        temperature=0.3,
        max_tokens=1200,  # room for full answer + all citations (was 800, refs cut off)
        streaming=streaming,
        # deepseek-v4-flash defaults to thinking "enabled" if omitted — this path
        # never used reasoning mode under the old deepseek-chat name, so keep it off.
        model_kwargs={"thinking": {"type": "disabled"}},
    )

router = APIRouter(tags=["action-prompt"])


class GeneratePromptRequest(BaseModel):
    assistant_slug: str | None = Field(None, description="Defaults to path param if omitted")
    action_id: str = Field(..., description="Action ID (e.g. assistant-host)")
    user_prompt: str = Field(
        default="",
        description="May be empty when manifest allows-empty-prompt or requires_prompt is false",
    )
    thread_id: str | None = Field(None, description="Optional thread for conversation context")
    conversation_context: str | None = Field(
        None,
        description="Previous Q&A for follow-up context (built from turns by client)",
    )
    language: str = Field("de-DE", description="BCP 47 locale")
    context_chunk_ids: list[str] | None = Field(
        None,
        description="Pre-selected chunk IDs for mit-kontext mode — injected as {kontext} slot alongside normal retrieval",
    )


class ExecutePromptRequest(BaseModel):
    prompt_id: str = Field(..., description="UUID from generate-prompt response")
    modified_prompt: str | None = Field(None, description="Full replacement if user edited the prompt")
    stream: bool = Field(True, description="Stream tokens via SSE")


@router.get("/agent/{assistant_slug}/actions")
async def list_available_actions(assistant_slug: str) -> dict:
    """List all available actions (prompt types) for the assistant."""
    assistant_name = load_assistant_name(assistant_slug)
    actions = list_actions(assistant_name=assistant_name)
    return {"assistant_slug": assistant_slug, "actions": actions}


@router.post("/agent/{assistant_slug}/generate-prompt")
async def generate_prompt(assistant_slug: str, body: GeneratePromptRequest) -> dict:
    """
    Run Qdrant queries per action-manifest, fill prompt template, cache state.
    Returns filled_prompt and prompt_id for execute-prompt.
    """
    effective_slug = body.assistant_slug or assistant_slug

    collection_name = load_assistant_rag_collection(effective_slug)
    # #region agent log
    try:
        with open(
            "/Users/michael/Reniets/Ai/ragkeep/.cursor/debug-b2cd2e.log",
            "a",
            encoding="utf-8",
        ) as _dbg:
            _dbg.write(
                json.dumps(
                    {
                        "sessionId": "b2cd2e",
                        "timestamp": int(time.time() * 1000),
                        "hypothesisId": "H1",
                        "location": "action_prompt.py:generate_prompt",
                        "message": "resolved rag collection",
                        "data": {
                            "assistant_slug": effective_slug,
                            "collection_name": collection_name,
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    except Exception:
        pass
    # #endregion
    conversation_context = (body.conversation_context or "").strip()

    if not body.user_prompt.strip():
        try:
            manifest = load_action_manifest(body.action_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        allows_empty = bool(manifest.get("allows-empty-prompt"))
        requires_prompt = manifest.get("requires_prompt", True)
        if not allows_empty and requires_prompt:
            raise HTTPException(
                status_code=400,
                detail="user_prompt is required for this action unless allows-empty-prompt is set",
            )

    embedding_client = get_embedding_client()
    qdrant_client = get_qdrant_client()

    (
        system_prompt,
        action_filled,
        query_results,
        citations_metadata,
        context_refs,
        retrieved_serialized,
        direct_response,
        chunk_index_map,
        context_ref_snippets_serialized,
    ) = await run_queries_and_fill_prompt(
        action_id=body.action_id,
        assistant_slug=effective_slug,
        user_prompt=body.user_prompt,
        collection_name=collection_name,
        conversation_context=conversation_context,
        embedding_client=embedding_client,
        qdrant_client=qdrant_client,
        context_chunk_ids=body.context_chunk_ids or [],
    )

    prompt_id = generate_prompt_id()
    store_prompt_state(
        prompt_id=prompt_id,
        filled_prompt=system_prompt,
        instruction_prompt="",
        action_prompt_filled=action_filled,
        query_results=query_results,
        citations_metadata=citations_metadata,
        context_refs=context_refs,
        assistant_slug=effective_slug,
        action_id=body.action_id,
        collection_name=collection_name,
        user_prompt=body.user_prompt,
        retrieved_snippets=retrieved_serialized,
        chunk_index_map=chunk_index_map,
        context_ref_snippets=context_ref_snippets_serialized,
    )

    estimated_tokens = len((system_prompt + action_filled).split()) * 2  # rough
    expires = datetime.now(timezone.utc) + timedelta(minutes=30)

    instruction = load_assistant_instruction(effective_slug)
    out: dict = {
        "prompt_id": prompt_id,
        "action_id": body.action_id,
        "filled_prompt": action_filled,
        "action_prompt_filled": action_filled,
        "instruction_prompt": instruction,
        "query_results": query_results,
        "citations_metadata": citations_metadata,
        "chunk_index_map": chunk_index_map,
        "estimated_tokens": estimated_tokens,
        "expires_at": expires.isoformat(),
    }
    if direct_response is not None:
        out["direct_response"] = direct_response
        # Build references from lemma-lookup for direct_response (clarify-concept)
        lemma_res = query_results.get("lemma-lookup", {})
        chunk_ids = lemma_res.get("chunk_ids", [])
        refs_direct: list[dict] = []
        for cid in chunk_ids:
            c = next(
                (m for m in citations_metadata if m.get("chunk_id") == cid),
                {},
            )
            segment_title = c.get("segment_title", "") or body.user_prompt.strip()
            refs_direct.append({
                "chunk_id": cid,
                "description": segment_title,
                "relevance": 1.0,
            })
        text_by_id = {}
        for s in retrieved_serialized:
            pl = s.get("payload") or {}
            inner = pl.get("payload") if isinstance(pl.get("payload"), dict) else pl
            cid = (inner or {}).get("chunk_id") if isinstance(inner, dict) else None
            if isinstance(cid, str):
                text_by_id[cid] = (s.get("text") or "")[:REFERENCE_TEXT_MAX_CHARS]
        refs_enriched = []
        for r in refs_direct:
            c = next(
                (m for m in citations_metadata if m.get("chunk_id") == r["chunk_id"]),
                {},
            )
            refs_enriched.append({
                "chunk_id": r["chunk_id"],
                "description": r["description"],
                "relevance": r["relevance"],
                "chunk_type": c.get("chunk_type", ""),
                "source_title": c.get("source_title", ""),
                "segment_title": c.get("segment_title", ""),
                "author": c.get("author", ""),
                "text": text_by_id.get(r["chunk_id"], ""),
            })
        out["references"] = refs_enriched
        out["collection"] = collection_name
        if refs_direct:
            enqueue_record_metadata_only(
                EventRecorder(),
                endpoint="generate_prompt_direct",
                collection=collection_name,
                metadata={"references": refs_direct},
            )
    return out


@router.post("/agent/{assistant_slug}/execute-prompt")
async def execute_prompt(assistant_slug: str, body: ExecutePromptRequest, request: Request) -> EventSourceResponse:
    """
    Execute the cached prompt: call LLM with filled prompt, stream response, attach citations.
    Uses modified_prompt if provided (full replacement).
    """
    state = get_prompt_state(body.prompt_id)
    if not state:
        raise HTTPException(status_code=404, detail="prompt_id expired or not found")

    system_prompt = state["filled_prompt"]  # identity + personality mode (fixed per turn)
    user_message = body.modified_prompt if body.modified_prompt else state["action_prompt_filled"]
    retrieved_serialized = state["retrieved_snippets"]
    citations_metadata = state["citations_metadata"]
    chunk_index_map = state.get("chunk_index_map") or []
    collection_name = state.get("collection_name", assistant_slug)

    account_id = request.headers.get("X-Account-Id", "anonymous")
    llm = _make_llm(streaming=body.stream)
    chat_client = get_deepseek_chat()
    _model_name = settings.deepseek_chat_model or "deepseek-v4-flash"

    async def event_generator():
        yield {"data": json.dumps({"type": "start"})}
        response_text = ""
        usage_meta: dict = {}
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message),
            ]
            if body.stream:
                async for chunk in llm.astream(messages):
                    if chunk.content:
                        response_text += chunk.content
                        yield {"data": json.dumps({"type": "token", "content": chunk.content})}
                    if chunk.usage_metadata:
                        usage_meta = chunk.usage_metadata
            else:
                resp = await llm.ainvoke(messages)
                response_text = (resp.content or "").strip()
                usage_meta = resp.usage_metadata or {}
                yield {"data": json.dumps({"type": "token", "content": response_text})}

            # 1. Resolve bracket citations ([n]) to primary-source chunk_ids
            cited_indices = _extract_cited_indices(response_text)
            bracket_primary_cids = _resolve_bracket_chunk_ids(
                cited_indices, chunk_index_map, retrieved_serialized
            )

            # 2. Rebuild primary-source snippets from cache for relevance evaluation
            context_ref_snippets_serialized = state.get("context_ref_snippets") or []
            context_ref_snippets = [
                RetrievedSnippet(
                    text=s.get("text", ""),
                    score=float(s.get("score", 1.0)),
                    payload=s.get("payload") or {},
                )
                for s in context_ref_snippets_serialized
            ]

            # 3. Always evaluate chunk relevance on primary-source snippets
            eval_refs: list[dict] = []
            if context_ref_snippets and response_text:
                eval_refs = await evaluate_chunk_relevance(
                    generated_text=response_text,
                    retrieved_chunks=context_ref_snippets,
                    llm=chat_client,
                )
            eval_refs = [r for r in eval_refs if float(r.get("relevance", 0)) >= CITATION_THRESHOLD]

            # 4. Build fast-lookup maps
            #    chunk_score_by_id: fallback retrieval score from original Qdrant hits
            chunk_score_by_id: dict[str, float] = {}
            for s in retrieved_serialized:
                pl = s.get("payload") or {}
                inner = pl.get("payload") if isinstance(pl.get("payload"), dict) else pl
                cid = (inner or {}).get("chunk_id") if isinstance(inner, dict) else None
                if isinstance(cid, str):
                    chunk_score_by_id[cid] = float(s.get("score", 0))

            #    crs_meta_by_id: text + metadata from primary-source snippets
            crs_meta_by_id: dict[str, dict] = {}
            for s in context_ref_snippets_serialized:
                pl = s.get("payload") or {}
                inner = pl.get("payload", pl) if isinstance(pl.get("payload"), dict) else pl
                cid = inner.get("chunk_id") if isinstance(inner, dict) else None
                if isinstance(cid, str) and cid:
                    crs_meta_by_id[cid] = {"text": s.get("text", ""), **inner}

            # 5. Union: eval results + bracket-cited primary sources
            eval_by_cid = {r["chunk_id"]: r for r in eval_refs}
            all_primary_cids = set(eval_by_cid) | bracket_primary_cids

            references_enriched: list[dict] = []
            for cid in all_primary_cids:
                meta = crs_meta_by_id.get(cid, {})
                c = next((m for m in citations_metadata if m.get("chunk_id") == cid), {})
                eval_entry = eval_by_cid.get(cid)
                chosen_source_title = meta.get("source_title") or c.get("source_title", "")
                chosen_segment_title = meta.get("segment_title") or c.get("segment_title", "")
                chosen_description = (
                    (eval_entry or {}).get("description")
                    or c.get("segment_title")
                    or c.get("source_title", "")
                )
                # region agent log
                _debug_log(
                    "H1",
                    "action_prompt.py:references_enriched_loop",
                    "reference title candidates",
                    {
                        "chunk_id": cid,
                        "chunk_type_meta": meta.get("chunk_type"),
                        "meta_source_title": meta.get("source_title"),
                        "meta_segment_title": meta.get("segment_title"),
                        "meta_book_title": meta.get("book_title"),
                        "citation_source_title": c.get("source_title"),
                        "citation_segment_title": c.get("segment_title"),
                        "chosen_source_title": chosen_source_title,
                        "chosen_segment_title": chosen_segment_title,
                        "chosen_description": chosen_description,
                        "has_eval_description": bool((eval_entry or {}).get("description")),
                    },
                )
                # endregion
                if (
                    isinstance(chosen_source_title, str)
                    and isinstance(chosen_segment_title, str)
                    and chosen_source_title.strip()
                    and chosen_segment_title.strip()
                    and chosen_source_title.strip().lower() == chosen_segment_title.strip().lower()
                ):
                    # region agent log
                    _debug_log(
                        "H4",
                        "action_prompt.py:references_enriched_loop",
                        "detected duplicate source/segment title",
                        {
                            "chunk_id": cid,
                            "chunk_type_meta": meta.get("chunk_type"),
                            "source_title": chosen_source_title,
                            "segment_title": chosen_segment_title,
                            "book_title": meta.get("book_title") or c.get("book_title"),
                        },
                    )
                    # endregion
                references_enriched.append({
                    "chunk_id": cid,
                    "description": chosen_description,
                    "relevance": float(
                        (eval_entry or {}).get("relevance", chunk_score_by_id.get(cid, 0.5))
                    ),
                    "chunk_type": meta.get("chunk_type") or c.get("chunk_type", ""),
                    "source_title": chosen_source_title,
                    "segment_title": chosen_segment_title,
                    "author": _author_for_reference(meta, c),
                    "text": (meta.get("text") or "")[:REFERENCE_TEXT_MAX_CHARS],
                    "cited": cid in bracket_primary_cids,
                })
            # region agent log
            _debug_log(
                "H3",
                "action_prompt.py:execute_prompt",
                "references_enriched summary",
                {
                    "count": len(references_enriched),
                    "count_duplicate_titles": sum(
                        1
                        for r in references_enriched
                        if isinstance(r.get("source_title"), str)
                        and isinstance(r.get("segment_title"), str)
                        and r.get("source_title", "").strip().lower() == r.get("segment_title", "").strip().lower()
                        and r.get("source_title", "").strip() != ""
                    ),
                    "sample": [
                        {
                            "chunk_id": r.get("chunk_id"),
                            "chunk_type": r.get("chunk_type"),
                            "source_title": r.get("source_title"),
                            "segment_title": r.get("segment_title"),
                            "description": r.get("description"),
                        }
                        for r in references_enriched[:3]
                    ],
                },
            )
            # endregion

            enqueue_record_metadata_only(
                EventRecorder(),
                endpoint="execute_prompt",
                collection=collection_name,
                metadata={"references": references_enriched},
            )

            if usage_meta:
                _pt = usage_meta.get("input_tokens")
                _ct = usage_meta.get("output_tokens")
                _tt = usage_meta.get("total_tokens")
                _cost = calculate_cost(_model_name, _pt, _ct)
                enqueue_record_usage(
                    UsageRecorder(),
                    account_id=account_id,
                    endpoint="execute_prompt",
                    model=_model_name,
                    provider="deepseek",
                    prompt_tokens=_pt,
                    completion_tokens=_ct,
                    total_tokens=_tt,
                    extra=_cost if (_cost["cost_usd"] > 0) else None,
                )
                response_text += (
                    f'\n<!-- usage {json.dumps({"prompt_tokens": _pt, "completion_tokens": _ct, "total_tokens": _tt, "model": _model_name, "cost_usd": _cost["cost_usd"], "cost_eur": _cost["cost_eur"]})} -->'
                )

            yield {
                "data": json.dumps({
                    "type": "done",
                    "references": references_enriched,
                    "response": response_text,
                    "collection": collection_name,
                    "chunk_index_map": chunk_index_map,
                })
            }
        except Exception as exc:
            logger.exception("execute-prompt failed: %s", exc)
            yield {"data": json.dumps({"type": "error", "message": str(exc)})}

    return EventSourceResponse(event_generator())
