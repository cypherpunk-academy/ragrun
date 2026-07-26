"""Service layer for typology explanation (book-filtered RAG + LLM)."""
from __future__ import annotations

from app.infra.deepseek_client import DeepSeekClient
from app.infra.embedding_client import EmbeddingClient
from app.infra.qdrant_client import QdrantClient
from app.retrieval.chains.typology_explain import run_typology_explain_chain
from app.retrieval.models import TypologyExplainResult


class TypologyExplainService:
    def __init__(
        self,
        *,
        embedding_client: EmbeddingClient,
        qdrant_client: QdrantClient,
        chat_client: DeepSeekClient,
        collection: str = "philo-von-freisinn",
    ) -> None:
        self.embedding_client = embedding_client
        self.qdrant_client = qdrant_client
        self.chat_client = chat_client
        self.collection = collection

    async def explain(
        self,
        *,
        name: str,
        aliases: list[str],
        source_ids: list[str],
        source_chunk_types: list[str] | None = None,
        source_chunks: list[str] | None = None,
        known_members: list[str] | None = None,
        verbose: bool = False,
        llm_retries: int = 3,
        run_verify: bool = True,
    ) -> TypologyExplainResult:
        return await run_typology_explain_chain(
            name=name,
            aliases=list(aliases or []),
            source_ids=source_ids,
            source_chunk_types=list(source_chunk_types or ["book", "secondary_book"]),
            collection=self.collection,
            embedding_client=self.embedding_client,
            qdrant_client=self.qdrant_client,
            chat_client=self.chat_client,
            source_chunks=source_chunks,
            known_members=list(known_members) if known_members else None,
            verbose=verbose,
            llm_retries=llm_retries,
            run_verify=run_verify,
        )
