"""Service layer for translating a base explanation into Sigrid worldviews."""

from __future__ import annotations

from typing import Sequence

from app.infra.deepseek_client import DeepSeekClient
from app.infra.embedding_client import EmbeddingClient
from app.infra.qdrant_client import QdrantClient
from app.retrieval.chains.translate_to_worldview import run_translate_to_worldview_chain
from app.retrieval.graphs.concept_explain_worldviews import RetrievalConfig
from app.retrieval.models import TranslateToWorldviewResult


class TranslateToWorldviewService:
    def __init__(
        self,
        *,
        embedding_client: EmbeddingClient,
        qdrant_client: QdrantClient,
        chat_client: DeepSeekClient,
        collection: str = "sigrid-von-gleich",
        max_concurrency: int = 4,
        hybrid: bool | None = None,
        cfg: RetrievalConfig | None = None,
    ) -> None:
        self.embedding_client = embedding_client
        self.qdrant_client = qdrant_client
        self.chat_client = chat_client
        self.collection = collection
        self.max_concurrency = max_concurrency
        self.hybrid = hybrid
        self.cfg = cfg

    async def translate(
        self,
        *,
        text: str,
        worldviews: Sequence[str],
        concept: str | None = None,
        verbose: bool = False,
        llm_retries: int = 3,
    ) -> TranslateToWorldviewResult:
        return await run_translate_to_worldview_chain(
            text=text,
            worldviews=worldviews,
            collection=self.collection,
            embedding_client=self.embedding_client,
            qdrant_client=self.qdrant_client,
            chat_client=self.chat_client,
            concept=concept,
            cfg=self.cfg,
            hybrid=self.hybrid,
            max_concurrency=self.max_concurrency,
            verbose=verbose,
            llm_retries=llm_retries,
        )

