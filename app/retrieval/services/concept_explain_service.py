"""Service layer for concept explanations."""
from __future__ import annotations

from app.retrieval.graphs.philo_von_freisinn import run_concept_explain_graph
from app.retrieval.models import ConceptExplainResult
from app.services.deepseek_client import DeepSeekClient
from app.services.embedding_client import EmbeddingClient
from app.services.qdrant_client import QdrantClient


class ConceptExplainService:
    """Retrieval + generation pipeline for concept explanations."""

    def __init__(
        self,
        *,
        embedding_client: EmbeddingClient,
        qdrant_client: QdrantClient,
        deepseek_client: DeepSeekClient,
        collection: str = "philo-von-freisinn",
        k: int = 10,
    ) -> None:
        self.embedding_client = embedding_client
        self.qdrant_client = qdrant_client
        self.deepseek_client = deepseek_client
        self.collection = collection
        self.k = k

    async def explain(self, concept: str) -> ConceptExplainResult:
        return await run_concept_explain_graph(
            concept=concept,
            collection=self.collection,
            k=self.k,
            embedding_client=self.embedding_client,
            qdrant_client=self.qdrant_client,
            deepseek_client=self.deepseek_client,
        )


