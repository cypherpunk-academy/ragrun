"""Thin entrypoint for concept_explain_worldviews graph."""
from __future__ import annotations

from typing import Sequence

from app.infra.deepseek_client import DeepSeekClient
from app.infra.embedding_client import EmbeddingClient
from app.infra.qdrant_client import QdrantClient
from app.retrieval.graphs.concept_explain_worldviews import (
    RetrievalConfig,
    run_concept_explain_worldviews_graph,
)
from app.retrieval.models import ConceptExplainWorldviewsResult


async def run_concept_explain_worldviews_chain(
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
) -> ConceptExplainWorldviewsResult:
    return await run_concept_explain_worldviews_graph(
        concept=concept,
        worldviews=worldviews,
        collection=collection,
        embedding_client=embedding_client,
        qdrant_client=qdrant_client,
        reasoning_client=reasoning_client,
        chat_client=chat_client,
        cfg=cfg,
        hybrid=hybrid,
        max_concurrency=max_concurrency,
    )
