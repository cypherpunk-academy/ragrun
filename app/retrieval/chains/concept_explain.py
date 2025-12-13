"""Chain for concept explanation retrieval and generation."""
from __future__ import annotations

from typing import Iterable, Sequence

from app.retrieval.models import ConceptExplainResult, RetrievedSnippet
from app.retrieval.prompts.philo_von_freisinn import build_concept_explain_prompt
from app.services.deepseek_client import DeepSeekClient
from app.services.embedding_client import EmbeddingClient
from app.services.qdrant_client import QdrantClient


SUMMARY_TYPES = {
    "chapter_summary",
    "talk_summary",
    "essay_summary",
    "explanation_summary",
}


async def _embed(concept: str, embedding_client: EmbeddingClient) -> Sequence[float]:
    result = await embedding_client.embed_texts([concept])
    return result.embeddings[0]


def _qdrant_filter_for_source(source_id: str) -> dict[str, object]:
    # Ingestion stores source_id as a flat payload field.
    return {"must": [{"key": "source_id", "match": {"value": source_id}}]}


async def _expand_summaries(
    hits: Iterable[RetrievedSnippet],
    *,
    collection: str,
    qdrant_client: QdrantClient,
) -> list[RetrievedSnippet]:
    expanded: list[RetrievedSnippet] = []
    def _payload(hit: RetrievedSnippet) -> dict[str, object]:
        p = hit.payload or {}
        inner = p.get("payload")
        return dict(inner) if isinstance(inner, dict) else dict(p)  # best-effort

    source_ids = {
        str(_payload(h).get("source_id") or "")
        for h in hits
        if _payload(h).get("chunk_type") in SUMMARY_TYPES
    }
    for src in source_ids:
        if not src:
            continue
        scroll = await qdrant_client.scroll_points(
            collection,
            filter_=_qdrant_filter_for_source(src),
            limit=256,
            with_payload=True,
            with_vectors=False,
        )
        for item in scroll:
            payload = item.get("payload", {})
            text = payload.get("text") or ""
            if not text:
                continue
            expanded.append(
                RetrievedSnippet(
                    text=text,
                    score=float(item.get("score") or 0.0),
                    payload=item,
                )
            )
    return expanded


async def run_concept_explain_chain(
    *,
    concept: str,
    collection: str,
    k: int,
    embedding_client: EmbeddingClient,
    qdrant_client: QdrantClient,
    deepseek_client: DeepSeekClient,
) -> ConceptExplainResult:
    vector = await _embed(concept, embedding_client)
    search_results = await qdrant_client.search_points(
        collection, vector=vector, limit=k, with_payload=True
    )
    primary_hits: list[RetrievedSnippet] = []
    for item in search_results:
        payload = item.get("payload", {})
        text = payload.get("text") or ""
        if not text:
            continue
        score = float(item.get("score") or 0.0)
        primary_hits.append(RetrievedSnippet(text=text, score=score, payload=item))

    expanded_hits = await _expand_summaries(primary_hits, collection=collection, qdrant_client=qdrant_client)

    messages = build_concept_explain_prompt(concept, primary_hits, expanded_hits)
    answer = await deepseek_client.chat(messages, temperature=0.2, max_tokens=300)

    return ConceptExplainResult(
        concept=concept,
        answer=answer,
        retrieved=primary_hits,
        expanded=expanded_hits,
    )


