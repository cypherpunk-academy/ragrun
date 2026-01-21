"""LangGraph-like orchestration for translating a base explanation into worldviews (Sigrid).

This graph:
- retrieves per-worldview context1/context2 using Qdrant (primary / primary+secondary)
- runs a WHAT step (main points) and a HOW step (final translated explanation)
- persists best-effort graph events via GraphEventRecorder

Important:
- Fails fast if per-worldview prompt files are missing for any requested worldview.
"""

from __future__ import annotations

import asyncio
import logging
import operator
import uuid
from typing import Sequence

from app.config import settings
from app.infra.deepseek_client import DeepSeekClient
from app.infra.embedding_client import EmbeddingClient
from app.infra.qdrant_client import QdrantClient
from app.retrieval.graphs.concept_explain_worldviews import (  # reuse stable retrieval helpers
    RetrievalConfig,
    _assess_sufficiency,  # noqa: SLF001
    _chat_with_retry,  # noqa: SLF001
    _retrieve_with_widen,  # noqa: SLF001
    _should_widen,  # noqa: SLF001
)
from app.retrieval.models import TranslateToWorldviewResult, WorldviewAnswer
from app.retrieval.prompts.sigrid_von_gleich_worldviews import (
    build_chat_messages,
    ensure_worldview_prompts_exist,
    render_worldview_how,
    render_worldview_what,
)
from app.retrieval.services.graph_event_recorder import GraphEventRecorder
from app.retrieval.utils.retrievers import build_context

logger = logging.getLogger(__name__)


async def run_translate_to_worldview_graph(
    *,
    text: str,
    worldviews: Sequence[str],
    collection: str,
    embedding_client: EmbeddingClient,
    qdrant_client: QdrantClient,
    chat_client: DeepSeekClient,
    concept: str | None = None,
    cfg: RetrievalConfig | None = None,
    hybrid: bool | None = None,
    max_concurrency: int = 4,
    event_recorder: GraphEventRecorder | None = None,
    verbose: bool = False,
    llm_retries: int = 3,
) -> TranslateToWorldviewResult:
    base_text = (text or "").strip()
    if not base_text:
        raise ValueError("text is required")
    if not worldviews:
        raise ValueError("worldviews must not be empty")

    # Fail fast: ensure prompt files exist for all requested worldviews.
    for wv in worldviews:
        ensure_worldview_prompts_exist(worldview=wv)

    # Reuse concept_explain_worldviews RetrievalConfig defaults + settings finals.
    if cfg is None:
        defaults = RetrievalConfig()
        k_final_context1 = settings.cewv_k_final_context1
        k_final_context2 = settings.cewv_k_final_context2
        cfg = RetrievalConfig(
            k_base_concept=defaults.k_base_concept,
            k_base_context1=max(defaults.k_base_context1, k_final_context1),
            k_base_context2=max(defaults.k_base_context2, k_final_context2),
            k_final_concept=defaults.k_final_concept,
            k_final_context1=k_final_context1,
            k_final_context2=k_final_context2,
            widen_concept=defaults.widen_concept,
            widen_context1=max(defaults.widen_context1, k_final_context1 * 3),
            widen_context2=max(defaults.widen_context2, k_final_context2 * 3),
            hybrid_k_dense=defaults.hybrid_k_dense,
            hybrid_k_sparse=defaults.hybrid_k_sparse,
            hybrid_k_fused=defaults.hybrid_k_fused,
        )

    # Hybrid selection: default to global setting unless explicitly overridden.
    hybrid_effective = settings.use_hybrid_retrieval if hybrid is None else bool(hybrid)

    graph_event_id = uuid.uuid4()
    graph_name = "translate_to_worldview"
    recorder = event_recorder or GraphEventRecorder()

    # GraphEventRecorder requires a non-empty concept field.
    concept_for_events = (concept or "translate_to_worldview").strip() or "translate_to_worldview"

    async def _record_event(step: str, **kwargs: object) -> None:
        await recorder.record_event(
            graph_event_id=graph_event_id,
            graph_name=graph_name,
            step=step,
            concept=concept_for_events,
            **kwargs,
        )

    results_map: dict[str, WorldviewAnswer] = {}
    semaphore = asyncio.Semaphore(max(1, int(max_concurrency)))

    async def _per_worldview(wv: str) -> WorldviewAnswer:
        async with semaphore:
            errors: list[str] = []

            # context1: primary only
            ctx1_outcome = await _retrieve_with_widen(
                query=base_text,
                worldview=wv,
                book_types=["primary"],
                collection=collection,
                embedding_client=embedding_client,
                qdrant_client=qdrant_client,
                cfg=cfg,
                k_base=cfg.k_base_context1,
                widen_to=cfg.widen_context1,
                k_final=cfg.k_final_context1,
                hybrid=hybrid_effective,
            )
            ctx1_text, ctx1_refs = build_context(ctx1_outcome.hits)
            await _record_event(
                "wv_context1_retrieval",
                worldview=wv,
                query_text=base_text,
                context_refs=ctx1_refs,
                context_text=ctx1_text,
                retrieval_mode=ctx1_outcome.mode,
                metadata={
                    "k_base": cfg.k_base_context1,
                    "widen_to": cfg.widen_context1,
                    "k_final": cfg.k_final_context1,
                    "widened": bool(ctx1_outcome.widened_hits),
                    "hybrid": hybrid_effective,
                },
            )

            # WHAT step
            what_user = render_worldview_what(
                worldview=wv, concept_explanation=base_text, context1_k5=ctx1_text
            )
            what_messages = build_chat_messages(user_content=what_user)
            main_points, what_prompt_messages = await _chat_with_retry(
                chat_client,
                what_messages,
                temperature=0.1,
                max_tokens=380,
                operation=f"wv_what:{wv}",
                verbose=verbose,
            )
            await _record_event(
                "wv_what",
                worldview=wv,
                prompt_messages=what_prompt_messages,
                response_text=main_points,
                context_refs=ctx1_refs,
                context_text=ctx1_text,
            )

            # context2: primary + secondary
            ctx2_outcome = await _retrieve_with_widen(
                query=base_text,
                worldview=wv,
                book_types=["primary", "secondary"],
                collection=collection,
                embedding_client=embedding_client,
                qdrant_client=qdrant_client,
                cfg=cfg,
                k_base=cfg.k_base_context2,
                widen_to=cfg.widen_context2,
                k_final=cfg.k_final_context2,
                hybrid=hybrid_effective,
            )
            ctx2_text, ctx2_refs = build_context(ctx2_outcome.hits)
            sufficiency = _assess_sufficiency(ctx2_outcome.hits, ctx2_text)
            if _should_widen(ctx2_outcome.hits, cfg.k_final_context2):
                sufficiency = "low"

            if not ctx2_outcome.hits:
                errors.append("no_context2_hits")
            if not ctx1_outcome.hits:
                errors.append("no_context1_hits")

            await _record_event(
                "wv_context2_retrieval",
                worldview=wv,
                query_text=base_text,
                context_refs=ctx2_refs,
                context_text=ctx2_text,
                retrieval_mode=ctx2_outcome.mode,
                sufficiency=sufficiency,
                metadata={
                    "k_base": cfg.k_base_context2,
                    "widen_to": cfg.widen_context2,
                    "k_final": cfg.k_final_context2,
                    "widened": bool(ctx2_outcome.widened_hits),
                    "hybrid": hybrid_effective,
                },
            )

            # HOW step
            how_user = render_worldview_how(
                worldview=wv,
                concept_explanation=base_text,
                context2_k10=ctx2_text,
                main_points=main_points,
            )
            how_messages = build_chat_messages(user_content=how_user)
            how_details, how_prompt_messages = await _chat_with_retry(
                chat_client,
                how_messages,
                temperature=0.3,
                max_tokens=520,
                operation=f"wv_how:{wv}",
                verbose=verbose,
                min_chars=600,
                retries=max(0, int(llm_retries)),
            )
            await _record_event(
                "wv_how",
                worldview=wv,
                prompt_messages=how_prompt_messages,
                response_text=how_details,
                context_refs=ctx2_refs,
                context_text=ctx2_text,
                sufficiency=sufficiency,
                errors=errors or None,
            )

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
        results_map = operator.or_(results_map, {ans.worldview: ans})

    # If everything failed, surface the failure to callers.
    if not results_map:
        raise ValueError("translate_to_worldview produced no results")

    return TranslateToWorldviewResult(
        input_text=base_text,
        worldviews=list(results_map.values()),
        graph_event_id=str(graph_event_id),
    )

