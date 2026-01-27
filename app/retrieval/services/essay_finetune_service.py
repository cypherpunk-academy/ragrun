"""Service layer for essay:finetune graph."""
from __future__ import annotations

from app.infra.deepseek_client import DeepSeekClient
from app.infra.embedding_client import EmbeddingClient
from app.infra.qdrant_client import QdrantClient
from app.retrieval.chains.essay_finetune import run_essay_finetune_chain
from app.retrieval.models import EssayFinetuneResult


class EssayFinetuneService:
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

    async def finetune(
        self,
        *,
        assistant: str,
        essay_slug: str,
        essay_title: str,
        essay_text: str,
        instruction: str,
        k: int = 5,
        verbose: bool = False,
    ) -> EssayFinetuneResult:
        return await run_essay_finetune_chain(
            assistant=assistant,
            essay_slug=essay_slug,
            essay_title=essay_title,
            essay_text=essay_text,
            instruction=instruction,
            k=k,
            embedding_client=self.embedding_client,
            qdrant_client=self.qdrant_client,
            chat_client=self.chat_client,
            verbose=verbose,
        )

