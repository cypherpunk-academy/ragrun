"""Thin entrypoint for essay_finetune graph."""
from __future__ import annotations

from app.infra.deepseek_client import DeepSeekClient
from app.infra.embedding_client import EmbeddingClient
from app.infra.qdrant_client import QdrantClient
from app.retrieval.graphs.essay_finetune import run_essay_finetune_graph
from app.retrieval.models import EssayFinetuneResult


async def run_essay_finetune_chain(
    *,
    assistant: str,
    essay_slug: str,
    essay_title: str,
    essay_text: str,
    instruction: str,
    k: int,
    embedding_client: EmbeddingClient,
    qdrant_client: QdrantClient,
    chat_client: DeepSeekClient,
    verbose: bool = False,
) -> EssayFinetuneResult:
    return await run_essay_finetune_graph(
        assistant=assistant,
        essay_slug=essay_slug,
        essay_title=essay_title,
        essay_text=essay_text,
        instruction=instruction,
        k=k,
        embedding_client=embedding_client,
        qdrant_client=qdrant_client,
        chat_client=chat_client,
        verbose=verbose,
    )

