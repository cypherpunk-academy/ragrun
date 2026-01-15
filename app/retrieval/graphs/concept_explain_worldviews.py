"""LangGraph-like orchestration for concept explain worldviews."""
from __future__ import annotations

import asyncio
import logging
import operator
import uuid
from dataclasses import dataclass, field
from typing import Iterable, List, Mapping, Optional, Sequence

from app.config import settings
from app.infra.deepseek_client import DeepSeekClient
from app.infra.embedding_client import EmbeddingClient
from app.infra.qdrant_client import QdrantClient
from app.retrieval.models import ConceptExplainWorldviewsResult, RetrievedSnippet, WorldviewAnswer
from app.retrieval.prompts.concept_explain_worldviews import (
    build_philo_explain_prompt,
    build_worldview_how_prompt,
    build_worldview_what_prompt,
)
from app.retrieval.services.graph_event_recorder import GraphEventRecorder
from app.retrieval.telemetry import retrieval_telemetry
from app.retrieval.utils.retrievers import (
    build_context,
    dense_retrieve,
    hybrid_retrieve,
    rerank_by_embedding,
    sparse_retrieve,
)

logger = logging.getLogger(__name__)

@dataclass(slots=True)
class RetrievalConfig:
    k_base_concept: int = 10
    k_base_context1: int = 5
    k_base_context2: int = 10
    k_final_concept: int = 6
    k_final_context1: int = 4
    k_final_context2: int = 6
    widen_concept: int = 18
    widen_context1: int = 10
    widen_context2: int = 18
    hybrid_k_dense: int = 12
    hybrid_k_sparse: int = 30
    hybrid_k_fused: int = 30


@dataclass(slots=True)
class RetrievalOutcome:
    hits: list[RetrievedSnippet] = field(default_factory=list)
    widened_hits: list[RetrievedSnippet] = field(default_factory=list)
    mode: str = "dense"


def _should_widen(snippets: Sequence[RetrievedSnippet], k_final: int, min_chars: int = 400) -> bool:
    if len(snippets) < k_final:
        return True
    total_chars = sum(len(s.text) for s in snippets)
    return total_chars < min_chars


async def _retrieve_with_widen(
    *,
    query: str,
    worldview: str | None,
    book_types: Iterable[str],
    collection: str,
    embedding_client: EmbeddingClient,
    qdrant_client: QdrantClient,
    cfg: RetrievalConfig,
    k_base: int,
    widen_to: int,
    k_final: int,
    hybrid: bool,
    sparse_only: bool = False,
) -> RetrievalOutcome:
    """Retrieve with optional widening and hybrid fallback."""

    async def _run_path(mode: str) -> tuple[list[RetrievedSnippet], list[RetrievedSnippet], str]:
        if mode == "sparse":
            mode_label = "sparse"
            hits = await sparse_retrieve(
                query=query,
                k=k_base,
                worldview=worldview,
                book_types=book_types,
                collection=collection,
                qdrant_client=qdrant_client,
            )
        elif mode == "hybrid":
            mode_label = "hybrid"
            hits = await hybrid_retrieve(
                query=query,
                k_dense=cfg.hybrid_k_dense,
                k_sparse=cfg.hybrid_k_sparse,
                k_fused=cfg.hybrid_k_fused,
                worldview=worldview,
                book_types=book_types,
                collection=collection,
                embedding_client=embedding_client,
                qdrant_client=qdrant_client,
                force_sparse=False,
            )
        else:
            mode_label = "dense"
            hits = await dense_retrieve(
                query=query,
                k=k_base,
                worldview=worldview,
                book_types=book_types,
                collection=collection,
                embedding_client=embedding_client,
                qdrant_client=qdrant_client,
            )

        reranked = await rerank_by_embedding(
            query=query, snippets=hits, embedding_client=embedding_client, k_final=k_final
        )

        if _should_widen(reranked, k_final):
            if mode == "sparse":
                widened_hits = await sparse_retrieve(
                    query=query,
                    k=widen_to,
                    worldview=worldview,
                    book_types=book_types,
                    collection=collection,
                    qdrant_client=qdrant_client,
                )
            else:
                widened_hits = await dense_retrieve(
                    query=query,
                    k=widen_to,
                    worldview=worldview,
                    book_types=book_types,
                    collection=collection,
                    embedding_client=embedding_client,
                    qdrant_client=qdrant_client,
                )
            reranked = await rerank_by_embedding(
                query=query, snippets=widened_hits, embedding_client=embedding_client, k_final=k_final
            )
            return reranked, widened_hits, mode_label

        return reranked, [], mode_label

    base_mode = "sparse" if sparse_only else ("hybrid" if bool(hybrid and settings.use_hybrid_retrieval) else "dense")
    base_hits, base_widened, base_mode_label = await _run_path(base_mode)
    chosen = RetrievalOutcome(hits=base_hits, widened_hits=base_widened, mode=base_mode_label)

    if sparse_only and not chosen.hits:
        raise ValueError("Sparse retrieval returned no hits")

    if (
        not sparse_only
        and settings.hybrid_fallback_on_thin
        and base_mode != "hybrid"
        and settings.use_hybrid_retrieval
        and _should_widen(base_hits, k_final)
    ):
        hybrid_hits, hybrid_widened, _ = await _run_path("hybrid")
        if _is_better_candidate(hybrid_hits, base_hits):
            return RetrievalOutcome(
                hits=hybrid_hits, widened_hits=hybrid_widened, mode=f"{base_mode_label}->hybrid"
            )

    return chosen


async def run_concept_explain_worldviews_graph(
    *,
    concept: str,
    worldviews: Sequence[str],
    collection: str,
    embedding_client: EmbeddingClient,
    qdrant_client: QdrantClient,
    reasoning_client: DeepSeekClient,
    chat_client: DeepSeekClient,
    cfg: RetrievalConfig | None = None,
    hybrid: bool | None = None,
    max_concurrency: int = 4,
    event_recorder: GraphEventRecorder | None = None,
) -> ConceptExplainWorldviewsResult:
    if not concept or not concept.strip():
        raise ValueError("concept is required")
    if not worldviews:
        raise ValueError("worldviews must not be empty")

    cfg = cfg or RetrievalConfig()
    short_concept = _is_short_query(
        concept,
        max_words=settings.hybrid_short_concept_max_words,
        max_chars=settings.hybrid_short_concept_max_chars,
    )
    hybrid_requested = settings.use_hybrid_retrieval if hybrid is None else hybrid
    if hybrid is None and settings.hybrid_prefer_short_concepts and short_concept:
        hybrid_requested = True
    hybrid = hybrid_requested
    graph_id = str(uuid.uuid4())
    graph_event_id = uuid.uuid4()
    graph_name = "concept_explain_worldviews"
    recorder = event_recorder or GraphEventRecorder()

    async def _record_event(step: str, **kwargs: object) -> None:
        await recorder.record_event(
            graph_event_id=graph_event_id,
            graph_name=graph_name,
            step=step,
            concept=concept,
            **kwargs,
        )

    # Step 1: concept explanation
    concept_outcome = await _retrieve_with_widen(
        query=concept,
        worldview=None,
        book_types=["primary"],
        collection=collection,
        embedding_client=embedding_client,
        qdrant_client=qdrant_client,
        cfg=cfg,
        k_base=cfg.k_base_concept,
        widen_to=cfg.widen_concept,
        k_final=cfg.k_final_concept,
        hybrid=hybrid,
        sparse_only=True,
    )
    concept_context, concept_refs = build_context(concept_outcome.hits)
    concept_errors = ["no_sparse_hits"] if not concept_outcome.hits else None
    philo_messages = build_philo_explain_prompt(concept=concept, context=concept_context)
    await _record_event(
        "concept_retrieval",
        query_text=concept,
        context_refs=concept_refs,
        context_text=concept_context,
        retrieval_mode=concept_outcome.mode,
        metadata={
            "k_base": cfg.k_base_concept,
            "widen_to": cfg.widen_concept,
            "k_final": cfg.k_final_concept,
            "widened": bool(concept_outcome.widened_hits),
            "hybrid": hybrid,
            "sparse_only": True,
            "empty_hits": not concept_outcome.hits,
        },
        errors=concept_errors,
    )
    if not concept_outcome.hits:
        raise ValueError("Sparse concept retrieval returned no hits; aborting graph")

    concept_explanation = await reasoning_client.chat(philo_messages, temperature=0.2, max_tokens=320)
    await _record_event(
        "concept_reasoning",
        prompt_messages=philo_messages,
        response_text=concept_explanation,
        context_refs=concept_refs,
        context_text=concept_context,
    )

    results_map: dict[str, WorldviewAnswer] = {}
    retrieval_modes: dict[str, dict[str, str]] = {}
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _per_worldview(wv: str) -> WorldviewAnswer:
        async with semaphore:
            errors: list[str] = []
            # context1: primary only
            ctx1_outcome = await _retrieve_with_widen(
                query=concept_explanation,
                worldview=wv,
                book_types=["primary"],
                collection=collection,
                embedding_client=embedding_client,
                qdrant_client=qdrant_client,
                cfg=cfg,
                k_base=cfg.k_base_context1,
                widen_to=cfg.widen_context1,
                k_final=cfg.k_final_context1,
                hybrid=hybrid,
            )
            ctx1_text, ctx1_refs = build_context(ctx1_outcome.hits)
            await _record_event(
                "wv_context1_retrieval",
                worldview=wv,
                query_text=concept_explanation,
                context_refs=ctx1_refs,
                context_text=ctx1_text,
                retrieval_mode=ctx1_outcome.mode,
                metadata={
                    "k_base": cfg.k_base_context1,
                    "widen_to": cfg.widen_context1,
                    "k_final": cfg.k_final_context1,
                    "widened": bool(ctx1_outcome.widened_hits),
                    "hybrid": hybrid,
                },
            )

            # what step
            what_prompt = build_worldview_what_prompt(
                concept_explanation=concept_explanation,
                worldview_description=wv,
                context=ctx1_text,
            )
            main_points = await chat_client.chat(what_prompt, temperature=0.3, max_tokens=320)
            await _record_event(
                "wv_what",
                worldview=wv,
                prompt_messages=what_prompt,
                response_text=main_points,
                context_refs=ctx1_refs,
                context_text=ctx1_text,
            )

            # context2: primary + secondary
            ctx2_outcome = await _retrieve_with_widen(
                query=concept_explanation,
                worldview=wv,
                book_types=["primary", "secondary"],
                collection=collection,
                embedding_client=embedding_client,
                qdrant_client=qdrant_client,
                cfg=cfg,
                k_base=cfg.k_base_context2,
                widen_to=cfg.widen_context2,
                k_final=cfg.k_final_context2,
                hybrid=hybrid,
            )
            ctx2_text, ctx2_refs = build_context(ctx2_outcome.hits)

            how_prompt = build_worldview_how_prompt(
                concept_explanation=concept_explanation,
                worldview_description=wv,
                main_points=main_points,
                context=ctx2_text,
            )
            how_details = await chat_client.chat(how_prompt, temperature=0.3, max_tokens=420)

            sufficiency = _assess_sufficiency(ctx2_outcome.hits, ctx2_text)
            if _should_widen(ctx2_outcome.hits, cfg.k_final_context2):
                sufficiency = "low"

            # Collect errors if widened but still thin
            if not ctx2_outcome.hits:
                errors.append("no_context2_hits")
            if not ctx1_outcome.hits:
                errors.append("no_context1_hits")

            await _record_event(
                "wv_context2_retrieval",
                worldview=wv,
                query_text=concept_explanation,
                context_refs=ctx2_refs,
                context_text=ctx2_text,
                retrieval_mode=ctx2_outcome.mode,
                sufficiency=sufficiency,
                metadata={
                    "k_base": cfg.k_base_context2,
                    "widen_to": cfg.widen_context2,
                    "k_final": cfg.k_final_context2,
                    "widened": bool(ctx2_outcome.widened_hits),
                    "hybrid": hybrid,
                },
            )

            await _record_event(
                "wv_how",
                worldview=wv,
                prompt_messages=how_prompt,
                response_text=how_details,
                context_refs=ctx2_refs,
                context_text=ctx2_text,
                sufficiency=sufficiency,
                errors=errors or None,
            )

            retrieval_modes[wv] = {
                "context1": ctx1_outcome.mode,
                "context2": ctx2_outcome.mode,
            }
            return WorldviewAnswer(
                worldview=wv,
                main_points=main_points,
                how_details=how_details,
                context1_refs=ctx1_refs,
                context2_refs=ctx2_refs,
                sufficiency=sufficiency,
                errors=errors or None,
            )

    tasks = [asyncio.create_task(_per_worldview(wv)) for wv in worldviews]
    answers = await asyncio.gather(*tasks, return_exceptions=True)
    for ans in answers:
        if isinstance(ans, Exception):
            logger.exception("worldview branch failed", exc_info=ans)
            continue
        # Reducer pattern: last-write-wins merge keyed by worldview.
        results_map = operator.or_(results_map, {ans.worldview: ans})

    # Persist/telemetry
    await retrieval_telemetry.record_worldviews(
        trace_id=None,
        graph_id=graph_id,
        concept=concept,
        worldviews=len(worldviews),
        metadata={
            "k_base_concept": cfg.k_base_concept,
            "k_base_context1": cfg.k_base_context1,
            "k_base_context2": cfg.k_base_context2,
            "k_final_concept": cfg.k_final_concept,
            "k_final_context1": cfg.k_final_context1,
            "k_final_context2": cfg.k_final_context2,
            "widened_concept": bool(concept_outcome.widened_hits),
            "hybrid": hybrid,
            "hybrid_short_concept": short_concept,
            "hybrid_fallback": settings.hybrid_fallback_on_thin,
            "retrieval_modes": {"concept": concept_outcome.mode, "contexts": retrieval_modes},
            "max_concurrency": max_concurrency,
            "sufficiency_counts": _sufficiency_hist(results_map.values()),
        },
    )

    # Reducer already implicit by list accumulation; use operator.or_ pattern if stateful dict required
    return ConceptExplainWorldviewsResult(
        concept=concept,
        concept_explanation=concept_explanation,
        worldviews=list(results_map.values()),
        context_refs=concept_refs,
        graph_event_id=str(graph_event_id),
    )


def _assess_sufficiency(ctx_hits: Sequence[RetrievedSnippet], ctx_text: str) -> str:
    if not ctx_hits or len(ctx_text) < 200:
        return "insufficient"
    if len(ctx_hits) < 3 or len(ctx_text) < 600:
        return "low"
    if len(ctx_text) > 2000:
        return "high"
    return "medium"


def _sufficiency_hist(worldviews: Iterable[WorldviewAnswer]) -> Mapping[str, int]:
    hist: dict[str, int] = {}
    for w in worldviews:
        key = w.sufficiency or "unknown"
        hist[key] = hist.get(key, 0) + 1
    return hist


def _is_short_query(text: str, *, max_words: int, max_chars: int) -> bool:
    """Heuristic: short/one-word concepts benefit from hybrid retrieval."""
    stripped = text.strip()
    if not stripped:
        return False
    words = stripped.split()
    return len(words) <= max_words or len(stripped) <= max_chars


def _snippets_total_chars(snippets: Sequence[RetrievedSnippet]) -> int:
    return sum(len(s.text) for s in snippets)


def _is_better_candidate(candidate: Sequence[RetrievedSnippet], current: Sequence[RetrievedSnippet]) -> bool:
    """Prefer candidate if it has more coverage (chars) or more hits."""
    if len(candidate) > len(current):
        return True
    return _snippets_total_chars(candidate) > _snippets_total_chars(current)
