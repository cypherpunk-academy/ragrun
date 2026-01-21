"""Service layer for authentic concept explanation (Steiner-first)."""
from __future__ import annotations

from app.infra.deepseek_client import DeepSeekClient
from app.infra.embedding_client import EmbeddingClient
from app.infra.qdrant_client import QdrantClient
from app.retrieval.chains.authentic_concept_explain import (
    VerifyRetrievalConfig,
    run_authentic_concept_explain_chain,
)
from app.retrieval.models import AuthenticConceptExplainResult


class AuthenticConceptExplainService:
    def __init__(
        self,
        *,
        embedding_client: EmbeddingClient,
        qdrant_client: QdrantClient,
        chat_client: DeepSeekClient,
        collection: str = "philo-von-freisinn",
        cfg: VerifyRetrievalConfig | None = None,
    ) -> None:
        self.embedding_client = embedding_client
        self.qdrant_client = qdrant_client
        self.chat_client = chat_client
        self.collection = collection
        self.cfg = cfg

    async def explain(
        self, *, concept: str, verbose: bool = False, llm_retries: int = 3
    ) -> AuthenticConceptExplainResult:
        return await run_authentic_concept_explain_chain(
            concept=concept,
            collection=self.collection,
            embedding_client=self.embedding_client,
            qdrant_client=self.qdrant_client,
            chat_client=self.chat_client,
            cfg=self.cfg,
            verbose=verbose,
            llm_retries=llm_retries,
        )

