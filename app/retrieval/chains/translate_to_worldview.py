"""Thin entrypoint for translate_to_worldview graph."""

from __future__ import annotations

from typing import Sequence

from app.infra.deepseek_client import DeepSeekClient
from app.infra.embedding_client import EmbeddingClient
from app.infra.qdrant_client import QdrantClient
from app.retrieval.graphs.concept_explain_worldviews import RetrievalConfig
from app.retrieval.graphs.translate_to_worldview import run_translate_to_worldview_graph
from app.retrieval.models import TranslateToWorldviewResult


async def run_translate_to_worldview_chain(
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
    verbose: bool = False,
    llm_retries: int = 3,
) -> TranslateToWorldviewResult:
    return await run_translate_to_worldview_graph(
        text=text,
        worldviews=worldviews,
        collection=collection,
        embedding_client=embedding_client,
        qdrant_client=qdrant_client,
        chat_client=chat_client,
        concept=concept,
        cfg=cfg,
        hybrid=hybrid,
        max_concurrency=max_concurrency,
        verbose=verbose,
        llm_retries=llm_retries,
    )

