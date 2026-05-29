"""Chain: retrieve book chunks -> LLM typology JSON -> optional verify -> references."""
from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any, Awaitable, Callable, Mapping, Sequence

from app.infra.deepseek_client import DeepSeekClient
from app.infra.embedding_client import EmbeddingClient
from app.infra.qdrant_client import QdrantClient
from app.retrieval.models import TypologyExplainResult
from app.retrieval.prompts.typology_explain import (
    build_typology_extract_messages,
    build_typology_verify_messages,
)
from app.retrieval.services.graph_event_recorder import GraphEventRecorder
from app.retrieval.utils.reference_evaluator import evaluate_chunk_relevance
from app.retrieval.utils.retrievers import (
    build_context,
    filter_snippets_by_typology_source,
    hybrid_retrieve,
    rerank_by_embedding,
    typology_dense_retrieve,
)
from app.retrieval.utils.retry import retry_async

logger = logging.getLogger(__name__)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _parse_llm_json_object(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("empty model response")
    m = _JSON_FENCE_RE.search(raw)
    if m:
        raw = m.group(1).strip()
    return json.loads(raw)


def _is_retryable_llm_error(exc: Exception) -> bool:
    message = str(exc).lower()
    if "deepseek returned empty content" in message:
        return True
    if "deepseek returned no choices" in message:
        return True
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    return bool(status_code and status_code >= 500)


async def _chat_json(
    chat_client: DeepSeekClient,
    messages: Sequence[Mapping[str, str]],
    *,
    temperature: float,
    max_tokens: int,
    operation: str,
    retries: int = 3,
    verbose: bool = False,
) -> tuple[dict[str, Any], list[Mapping[str, str]]]:
    async def _once() -> tuple[dict[str, Any], list[Mapping[str, str]]]:
        if verbose:
            logger.warning(
                "Prompt for %s:\n%s",
                operation,
                "\n\n".join(f"[{m.get('role')}] {m.get('content', '')}" for m in messages),
            )
        result_obj = await chat_client.chat(
            list(messages), temperature=temperature, max_tokens=max_tokens
        )
        return _parse_llm_json_object(result_obj.content), list(messages)

    return await retry_async(
        _once,
        retries=max(0, int(retries)),
        base_delay=1.0,
        max_delay=8.0,
        jitter=0.2,
        retry_on=_is_retryable_llm_error,
        logger=logger,
        operation=operation,
    )


def _merge_aliases(name: str, request_aliases: Sequence[str], extra: Any) -> list[str]:
    order: list[str] = []
    seen_cf: set[str] = set()
    for raw in (name, *request_aliases, *(extra if isinstance(extra, list) else [])):
        if not isinstance(raw, str):
            continue
        s = raw.strip()
        if not s:
            continue
        k = s.casefold()
        if k not in seen_cf:
            seen_cf.add(k)
            order.append(s)
    return order if order else [name.strip()]


async def run_typology_explain_chain(
    *,
    name: str,
    aliases: list[str],
    source_ids: list[str],
    source_chunk_types: list[str],
    collection: str,
    embedding_client: EmbeddingClient,
    qdrant_client: QdrantClient,
    chat_client: DeepSeekClient,
    source_chunks: list[str] | None = None,
    known_members: list[str] | None = None,
    event_recorder: GraphEventRecorder | None = None,
    verbose: bool = False,
    llm_retries: int = 3,
    progress_callback: Callable[[str, str], Awaitable[None]] | None = None,
    run_verify: bool = True,
) -> TypologyExplainResult:
    """
    Retrieve from books filtered by source_id UUIDs, ask LLM for structured typology, optionally verify members.
    `source_chunks` is reserved for future chunk-id pinning; currently unused.
    """

    del source_chunks  # reserved

    tname = (name or "").strip()
    if not tname:
        raise ValueError("name is required")
    sid_list = [s.strip() for s in source_ids if isinstance(s, str) and s.strip()]
    if not sid_list:
        raise ValueError("source_ids must contain at least one book UUID")

    ct_list = (
        [c.strip() for c in source_chunk_types if isinstance(c, str) and c.strip()]
        if source_chunk_types
        else ["book", "secondary_book"]
    )

    graph_event_id = uuid.uuid4()
    graph_name = "typology_explain"
    recorder = event_recorder or GraphEventRecorder()

    async def _progress(step: str, label: str) -> None:
        if progress_callback is not None:
            try:
                await progress_callback(step, label)
            except Exception:
                logger.debug("progress_callback failed for step=%s", step)

    async def _record_event(step: str, **kwargs: object) -> None:
        await recorder.record_event(
            graph_event_id=graph_event_id,
            graph_name=graph_name,
            step=step,
            concept=tname,
            collection=collection,
            **kwargs,
        )

    query_bits = [tname, *[a for a in aliases if isinstance(a, str) and a.strip()][:12]]
    query_text = ". ".join(query_bits)

    await _progress("typology_retrieval", "Quellen für Typologie abrufen…")

    reranked = await typology_dense_retrieve(
        query=query_text,
        k=14,
        worldview=None,
        book_types=ct_list,
        collection=collection,
        embedding_client=embedding_client,
        qdrant_client=qdrant_client,
        chunk_types=ct_list,
        source_ids=sid_list,
    )

    if len(reranked) < 6:
        wide = await typology_dense_retrieve(
            query=query_text,
            k=28,
            worldview=None,
            book_types=ct_list,
            collection=collection,
            embedding_client=embedding_client,
            qdrant_client=qdrant_client,
            chunk_types=ct_list,
            source_ids=sid_list,
        )
        reranked = wide

    if not reranked and hasattr(qdrant_client, "search_sparse_points"):
        try:
            h = await hybrid_retrieve(
                query=query_text,
                k_dense=24,
                k_sparse=24,
                k_fused=40,
                worldview=None,
                book_types=ct_list,
                collection=collection,
                embedding_client=embedding_client,
                qdrant_client=qdrant_client,
            )
            reranked = filter_snippets_by_typology_source(
                h,
                chunk_types=ct_list,
                source_ids=sid_list,
            )
        except Exception:
            logger.exception("typology hybrid retrieval failed")

    if not reranked:
        raise ValueError(
            f"Typology retrieval returned no hits for source_ids={sid_list}; "
            "check that the books are uploaded to the collection"
        )

    reranked = await rerank_by_embedding(
        query=query_text, snippets=reranked, embedding_client=embedding_client, k_final=12
    )

    context, refs = build_context(reranked, max_chars=14_000)
    await _record_event(
        "typology_retrieval",
        query_text=query_text,
        context_refs=refs,
        retrieval_mode="dense+filter",
        metadata={
            "source_ids": sid_list,
            "chunk_types": ct_list,
        },
        errors=None,
    )

    await _progress("typology_extract", "Typologie extrahieren…")
    extract_messages = build_typology_extract_messages(
        name=tname,
        aliases=list(aliases),
        chunks=context,
        known_members=list(known_members) if known_members else None,
    )
    data, extract_prompt_messages = await _chat_json(
        chat_client,
        extract_messages,
        temperature=0.2,
        max_tokens=4_000,
        operation="typology_extract",
        retries=llm_retries,
        verbose=verbose,
    )
    await _record_event(
        "typology_extract",
        prompt_messages=extract_prompt_messages,
        response_text=json.dumps(data, ensure_ascii=False),
        context_refs=refs,
    )

    members_raw = data.get("members")
    if not isinstance(members_raw, list):
        raise ValueError('model JSON must include "members" as a list')
    members = [str(m).strip() for m in members_raw if str(m).strip()]

    explanation = data.get("explanation")
    if not isinstance(explanation, str) or not explanation.strip():
        raise ValueError('model JSON must include non-empty "explanation"')

    merged_aliases = _merge_aliases(tname, aliases, data.get("aliases_extra"))

    if run_verify and members:
        await _progress("typology_verify", "Mitglieder gegen Quellen prüfen…")
        verify_messages = build_typology_verify_messages(name=tname, members=members, chunks=context)
        try:
            vdata, verify_prompt_messages = await _chat_json(
                chat_client,
                verify_messages,
                temperature=0.1,
                max_tokens=1_200,
                operation="typology_verify",
                retries=llm_retries,
                verbose=verbose,
            )
            await _record_event(
                "typology_verify",
                prompt_messages=verify_prompt_messages,
                response_text=json.dumps(vdata, ensure_ascii=False),
                context_refs=refs,
            )
            unsupported = vdata.get("unsupported_members")
            if isinstance(unsupported, list):
                bad = {str(x).strip().casefold() for x in unsupported if str(x).strip()}
                members = [m for m in members if m.casefold() not in bad]
        except Exception as exc:
            logger.warning("typology verify step failed: %s", exc)

    references = await evaluate_chunk_relevance(
        generated_text=explanation,
        retrieved_chunks=reranked,
        llm=chat_client,
        max_chunks=20,
    )

    return TypologyExplainResult(
        name=tname,
        members=members,
        explanation=explanation.strip(),
        aliases=merged_aliases,
        references=references,
        graph_event_id=str(graph_event_id),
    )
