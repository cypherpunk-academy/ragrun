"""Service layer for concept explanations."""
from __future__ import annotations

from app.retrieval.graphs.philo_von_freisinn import run_concept_explain_graph
from app.retrieval.models import ConceptExplainResult
from app.retrieval.services.retrieval_logging import (
    RetrievalLoggingRepository,
    enqueue_log_concept_explain,
)
from app.infra.deepseek_client import DeepSeekClient
from app.infra.embedding_client import EmbeddingClient
from app.infra.qdrant_client import QdrantClient


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
        logging_repo: RetrievalLoggingRepository | None = None,
        branch: str = "concept-explain",
    ) -> None:
        self.embedding_client = embedding_client
        self.qdrant_client = qdrant_client
        self.deepseek_client = deepseek_client
        self.collection = collection
        self.k = k
        self.logging_repo = logging_repo or RetrievalLoggingRepository()
        self.branch = branch

    async def explain(self, concept: str) -> ConceptExplainResult:
        result = await run_concept_explain_graph(
            concept=concept,
            collection=self.collection,
            k=self.k,
            embedding_client=self.embedding_client,
            qdrant_client=self.qdrant_client,
            deepseek_client=self.deepseek_client,
        )

        enqueue_log_concept_explain(
            repository=self.logging_repo,
            concept=result.concept,
            branch=self.branch,
            collection=self.collection,
            answer=result.answer,
            primary=result.retrieved,
            expanded=result.expanded,
        )

        return result


