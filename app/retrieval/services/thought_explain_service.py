"""Service layer for thought explanation."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from app.infra.deepseek_client import DeepSeekClient
from app.infra.embedding_client import EmbeddingClient
from app.infra.qdrant_client import QdrantClient
from app.retrieval.graphs.thought_explain_graph import run_thought_explain_graph
from app.retrieval.models import ThoughtExplainResult


class ThoughtExplainService:
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
        thought_id: str,
        text: str,
        quote: str,
        theme: str,
        alignment: str,
        writing_style: str = "",
        references: Sequence[Mapping[str, Any]] | None = None,
        verbose: bool = False,
        llm_retries: int = 3,
    ) -> ThoughtExplainResult:
        return await run_thought_explain_graph(
            thought_id=thought_id,
            text=text,
            quote=quote,
            theme=theme,
            alignment=alignment,
            writing_style=writing_style,
            collection=self.collection,
            assistant_slug=self.collection,
            embedding_client=self.embedding_client,
            qdrant_client=self.qdrant_client,
            chat_client=self.chat_client,
            incoming_references=references,
            verbose=verbose,
            llm_retries=llm_retries,
        )
