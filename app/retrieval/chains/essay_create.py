"""Thin entrypoint for essay_create graph."""
from __future__ import annotations

from typing import Sequence

from app.infra.deepseek_client import DeepSeekClient
from app.infra.embedding_client import EmbeddingClient
from app.infra.qdrant_client import QdrantClient
from app.retrieval.graphs.essay_create import run_essay_create_graph
from app.retrieval.models import EssayCreateResult


async def run_essay_create_chain(
    *,
    assistant: str,
    essay_slug: str,
    essay_title: str,
    pitch_steps: Sequence[str],
    k: int,
    embedding_client: EmbeddingClient,
    qdrant_client: QdrantClient,
    chat_client: DeepSeekClient,
    verbose: bool = False,
) -> EssayCreateResult:
    return await run_essay_create_graph(
        assistant=assistant,
        essay_slug=essay_slug,
        essay_title=essay_title,
        pitch_steps=pitch_steps,
        k=k,
        embedding_client=embedding_client,
        qdrant_client=qdrant_client,
        chat_client=chat_client,
        verbose=verbose,
    )

