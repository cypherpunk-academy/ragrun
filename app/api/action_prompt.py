"""API for generate-prompt / execute-prompt (ASSISTANTS_CHAT_PLAN_V2)."""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config import settings
from app.retrieval.services.event_recorder import EventRecorder, enqueue_record_metadata_only
from app.retrieval.services.action_prompt_service import (
    generate_prompt_id,
    get_prompt_state,
    list_actions,
    load_action_manifest,
    load_assistant_instruction,
    run_queries_and_fill_prompt,
    store_prompt_state,
)
from app.retrieval.services.providers import get_embedding_client, get_qdrant_client, get_deepseek_chat
from app.retrieval.utils.reference_evaluator import evaluate_chunk_relevance
from app.retrieval.models import RetrievedSnippet

logger = logging.getLogger(__name__)

CITATION_THRESHOLD = 0.3

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

    For essay/talk chunks: expands to their referenced primary-source chunk_ids.
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
        if chunk_type in ("essay", "talk"):
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
        model=settings.deepseek_chat_model or "deepseek-chat",
        openai_api_key=settings.deepseek_api_key,
        openai_api_base=f"{str(settings.deepseek_base_url).rstrip('/')}/",
        temperature=0.3,
        max_tokens=1200,  # room for full answer + all citations (was 800, refs cut off)
        streaming=streaming,
    )

router = APIRouter(tags=["action-prompt"])


class GeneratePromptRequest(BaseModel):
    assistant_slug: str | None = Field(None, description="Defaults to path param if omitted")
    action_id: str = Field(..., description="Action ID (e.g. general-question)")
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


class ExecutePromptRequest(BaseModel):
    prompt_id: str = Field(..., description="UUID from generate-prompt response")
    modified_prompt: str | None = Field(None, description="Full replacement if user edited the prompt")
    stream: bool = Field(True, description="Stream tokens via SSE")


@router.get("/agent/{assistant_slug}/actions")
async def list_available_actions(assistant_slug: str) -> dict:
    """List all available actions (prompt types) for the assistant."""
    actions = list_actions()
    return {"assistant_slug": assistant_slug, "actions": actions}


@router.post("/agent/{assistant_slug}/generate-prompt")
async def generate_prompt(assistant_slug: str, body: GeneratePromptRequest) -> dict:
    """
    Run Qdrant queries per action-manifest, fill prompt template, cache state.
    Returns filled_prompt and prompt_id for execute-prompt.
    """
    effective_slug = body.assistant_slug or assistant_slug

    # Match assistant_chat: collection = assistant_slug (or use assistant-manifest rag-collection)
    collection_name = effective_slug
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
        full_system,
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
    )

    prompt_id = generate_prompt_id()
    store_prompt_state(
        prompt_id=prompt_id,
        filled_prompt=full_system,
        instruction_prompt="",  # included in filled_prompt
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

    estimated_tokens = len(full_system.split()) * 2  # rough
    expires = datetime.now(timezone.utc) + timedelta(minutes=30)

    instruction = load_assistant_instruction(effective_slug)
    out: dict = {
        "prompt_id": prompt_id,
        "action_id": body.action_id,
        "filled_prompt": full_system,
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
                "source_title": c.get("source_title", ""),
                "segment_title": c.get("segment_title", ""),
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
async def execute_prompt(assistant_slug: str, body: ExecutePromptRequest) -> EventSourceResponse:
    """
    Execute the cached prompt: call LLM with filled prompt, stream response, attach citations.
    Uses modified_prompt if provided (full replacement).
    """
    state = get_prompt_state(body.prompt_id)
    if not state:
        raise HTTPException(status_code=404, detail="prompt_id expired or not found")

    system_prompt = body.modified_prompt if body.modified_prompt else state["filled_prompt"]
    user_prompt = state["user_prompt"]
    retrieved_serialized = state["retrieved_snippets"]
    citations_metadata = state["citations_metadata"]
    chunk_index_map = state.get("chunk_index_map") or []
    collection_name = state.get("collection_name", assistant_slug)

    llm = _make_llm(streaming=body.stream)
    chat_client = get_deepseek_chat()

    async def event_generator():
        yield {"data": json.dumps({"type": "start"})}
        response_text = ""
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
            if body.stream:
                async for chunk in llm.astream(messages):
                    if chunk.content:
                        response_text += chunk.content
                        yield {"data": json.dumps({"type": "token", "content": chunk.content})}
            else:
                resp = await llm.ainvoke(messages)
                response_text = (resp.content or "").strip()
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
                references_enriched.append({
                    "chunk_id": cid,
                    "description": (
                        (eval_entry or {}).get("description")
                        or c.get("segment_title")
                        or c.get("source_title", "")
                    ),
                    "relevance": float(
                        (eval_entry or {}).get("relevance", chunk_score_by_id.get(cid, 0.5))
                    ),
                    "source_title": meta.get("source_title") or c.get("source_title", ""),
                    "segment_title": meta.get("segment_title") or c.get("segment_title", ""),
                    "text": (meta.get("text") or "")[:REFERENCE_TEXT_MAX_CHARS],
                    "cited": cid in bracket_primary_cids,
                })

            enqueue_record_metadata_only(
                EventRecorder(),
                endpoint="execute_prompt",
                collection=collection_name,
                metadata={"references": references_enriched},
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
