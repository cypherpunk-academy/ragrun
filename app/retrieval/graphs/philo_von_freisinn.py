"""Graph wrapper for the philo-von-freisinn agent."""
from __future__ import annotations

from app.retrieval.chains.concept_explain import run_concept_explain_chain
from app.retrieval.models import ConceptExplainResult
from app.infra.deepseek_client import DeepSeekClient
from app.infra.embedding_client import EmbeddingClient
from app.infra.qdrant_client import QdrantClient


async def run_concept_explain_graph(
    *,
    concept: str,
    collection: str,
    k: int,
    embedding_client: EmbeddingClient,
    qdrant_client: QdrantClient,
    deepseek_client: DeepSeekClient,
) -> ConceptExplainResult:
    # Thin wrapper; a real graph could add branching, retries, telemetry, etc.
    return await run_concept_explain_chain(
        concept=concept,
        collection=collection,
        k=k,
        embedding_client=embedding_client,
        qdrant_client=qdrant_client,
        deepseek_client=deepseek_client,
    )


