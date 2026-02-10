"""Service layer for essay_completion graph."""
from __future__ import annotations

from app.infra.deepseek_client import DeepSeekClient
from app.infra.embedding_client import EmbeddingClient
from app.infra.qdrant_client import QdrantClient
from app.retrieval.chains.essay_completion import run_essay_completion_chain
from app.retrieval.models import EssayCompletionResult


class EssayCompletionService:
    def __init__(
        self,
        *,
        embedding_client: EmbeddingClient,
        qdrant_client: QdrantClient,
        chat_client: DeepSeekClient,
    ) -> None:
        self.embedding_client = embedding_client
        self.qdrant_client = qdrant_client
        self.chat_client = chat_client

    async def complete(
        self,
        *,
        assistant: str,
        essay_slug: str,
        essay_title: str,
        mood_index: int,
        mood_name: str | None,
        current_text: str | None,
        current_header: str | None = None,
        previous_parts: str | None = None,
        k: int = 5,
        verbose: bool = False,
        llm_retries: int = 3,
        force: bool = False,
    ) -> EssayCompletionResult:
        return await run_essay_completion_chain(
            assistant=assistant,
            essay_slug=essay_slug,
            essay_title=essay_title,
            mood_index=mood_index,
            mood_name=mood_name,
            current_text=current_text,
            current_header=current_header,
            previous_parts=previous_parts,
            k=k,
            embedding_client=self.embedding_client,
            qdrant_client=self.qdrant_client,
            chat_client=self.chat_client,
            verbose=verbose,
            llm_retries=llm_retries,
            force=force,
        )
