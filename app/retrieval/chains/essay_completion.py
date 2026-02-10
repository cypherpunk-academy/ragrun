"""Thin entrypoint for essay_completion graph."""
from __future__ import annotations

from app.infra.deepseek_client import DeepSeekClient
from app.infra.embedding_client import EmbeddingClient
from app.infra.qdrant_client import QdrantClient
from app.retrieval.graphs.essay_completion import run_essay_completion_graph
from app.retrieval.models import EssayCompletionResult


async def run_essay_completion_chain(
    *,
    assistant: str,
    essay_slug: str,
    essay_title: str,
    mood_index: int,
    mood_name: str | None,
    current_text: str | None,
    current_header: str | None,
    previous_parts: str | None,
    k: int,
    embedding_client: EmbeddingClient,
    qdrant_client: QdrantClient,
    chat_client: DeepSeekClient,
    verbose: bool = False,
    llm_retries: int = 3,
    force: bool = False,
) -> EssayCompletionResult:
    return await run_essay_completion_graph(
        assistant=assistant,
        essay_slug=essay_slug,
        essay_title=essay_title,
        mood_index=mood_index,
        mood_name=mood_name,
        current_text=current_text,
        current_header=current_header,
        previous_parts=previous_parts,
        k=k,
        embedding_client=embedding_client,
        qdrant_client=qdrant_client,
        chat_client=chat_client,
        verbose=verbose,
        llm_retries=llm_retries,
        force=force,
    )
