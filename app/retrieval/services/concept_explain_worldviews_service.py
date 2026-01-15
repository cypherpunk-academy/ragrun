"""Service layer for concept explain worldviews."""
from __future__ import annotations

from typing import Sequence

from app.retrieval.chains.concept_explain_worldviews import (
    RetrievalConfig,
    run_concept_explain_worldviews_chain,
)
from app.retrieval.models import ConceptExplainWorldviewsResult
from app.infra.deepseek_client import DeepSeekClient
from app.infra.embedding_client import EmbeddingClient
from app.infra.qdrant_client import QdrantClient


class ConceptExplainWorldviewsService:
    """Coordinates retrieval and generation for concept explain per-worldview."""

    def __init__(
        self,
        *,
        embedding_client: EmbeddingClient,
        qdrant_client: QdrantClient,
        reasoning_client: DeepSeekClient,
        chat_client: DeepSeekClient,
        collection: str = "philo-von-freisinn",
        max_concurrency: int = 4,
        hybrid: bool | None = None,
        cfg: RetrievalConfig | None = None,
    ) -> None:
        self.embedding_client = embedding_client
        self.qdrant_client = qdrant_client
        self.reasoning_client = reasoning_client
        self.chat_client = chat_client
        self.collection = collection
        self.max_concurrency = max_concurrency
        self.hybrid = hybrid
        self.cfg = cfg

    async def explain(self, *, concept: str, worldviews: Sequence[str]) -> ConceptExplainWorldviewsResult:
        return await run_concept_explain_worldviews_chain(
            concept=concept,
            worldviews=worldviews,
            collection=self.collection,
            embedding_client=self.embedding_client,
            qdrant_client=self.qdrant_client,
            reasoning_client=self.reasoning_client,
            chat_client=self.chat_client,
            cfg=self.cfg,
            hybrid=self.hybrid,
            max_concurrency=self.max_concurrency,
        )
