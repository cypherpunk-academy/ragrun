"""Service layer for essay:create graph."""
from __future__ import annotations

from typing import Sequence

from app.infra.deepseek_client import DeepSeekClient
from app.infra.embedding_client import EmbeddingClient
from app.infra.qdrant_client import QdrantClient
from app.retrieval.chains.essay_create import run_essay_create_chain
from app.retrieval.models import EssayCreateResult


class EssayCreateService:
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

    async def create(
        self,
        *,
        assistant: str,
        essay_slug: str,
        essay_title: str,
        pitch_steps: Sequence[str],
        k: int = 5,
        verbose: bool = False,
    ) -> EssayCreateResult:
        return await run_essay_create_chain(
            assistant=assistant,
            essay_slug=essay_slug,
            essay_title=essay_title,
            pitch_steps=pitch_steps,
            k=k,
            embedding_client=self.embedding_client,
            qdrant_client=self.qdrant_client,
            chat_client=self.chat_client,
            verbose=verbose,
        )

