"""Service: Referenzen zu einem vorhandenen Text nachträglich ermitteln.

Wird von ragprep (`rag:augment:references`) genutzt, um Begriff- und Typologie-Chunks
nachzurüsten, die ohne `references` erzeugt wurden. Der Text (Begriffserklärung bzw.
Typologie-Langtext) dient als Suchanfrage; die Treffer aus Primärbüchern und Vorträgen
werden mit demselben Evaluator bewertet wie bei der Erzeugung.
"""
from __future__ import annotations

import logging
from typing import Any, Mapping

from app.infra.deepseek_client import DeepSeekClient
from app.infra.embedding_client import EmbeddingClient
from app.infra.qdrant_client import QdrantClient
from app.retrieval.models import RetrievedSnippet
from app.retrieval.services.action_prompt_service import load_assistant_embedding_prefixes
from app.retrieval.utils.reference_evaluator import evaluate_chunk_relevance
from app.retrieval.utils.retrievers import dense_retrieve

logger = logging.getLogger(__name__)

DEFAULT_K_BOOKS = 8
DEFAULT_K_LECTURES = 4
MAX_QUERY_CHARS = 2_000


def _snippet_chunk_id(snippet: RetrievedSnippet) -> str | None:
    payload = snippet.payload
    if not isinstance(payload, Mapping):
        return None
    inner = payload.get("payload")
    p = inner if isinstance(inner, Mapping) else payload
    cid = p.get("chunk_id")
    return cid.strip() if isinstance(cid, str) and cid.strip() else None


def _dedupe(snippets: list[RetrievedSnippet], exclude: set[str]) -> list[RetrievedSnippet]:
    seen: set[str] = set()
    out: list[RetrievedSnippet] = []
    for snippet in snippets:
        cid = _snippet_chunk_id(snippet)
        if cid is None or cid in exclude or cid in seen:
            continue
        seen.add(cid)
        out.append(snippet)
    return out


async def compute_chunk_references(
    *,
    text: str,
    collection: str,
    assistant: str,
    embedding_client: EmbeddingClient,
    qdrant_client: QdrantClient,
    chat_client: DeepSeekClient,
    exclude_chunk_ids: list[str] | None = None,
    k_books: int = DEFAULT_K_BOOKS,
    k_lectures: int = DEFAULT_K_LECTURES,
) -> list[dict[str, Any]]:
    """Gibt `[{chunk_id, description, relevance}]` für `text` zurück (leer, wenn nichts passt)."""
    body = (text or "").strip()
    if not body:
        raise ValueError("text is required")
    query = body[:MAX_QUERY_CHARS]

    _prefix_passage, query_prefix = load_assistant_embedding_prefixes(assistant)

    hits_books = await dense_retrieve(
        query=query,
        k=k_books,
        worldview=None,
        book_types=["primary"],
        collection=collection,
        embedding_client=embedding_client,
        qdrant_client=qdrant_client,
        query_prefix=query_prefix,
    )
    hits_lectures = await dense_retrieve(
        query=query,
        k=k_lectures,
        worldview=None,
        book_types=["talk", "talk_summary"],
        collection=collection,
        embedding_client=embedding_client,
        qdrant_client=qdrant_client,
        query_prefix=query_prefix,
    )

    exclude = {c.strip() for c in (exclude_chunk_ids or []) if isinstance(c, str) and c.strip()}
    hits = _dedupe(list(hits_books) + list(hits_lectures), exclude)
    if not hits:
        logger.warning("chunk-references: keine Treffer für Text (%s Zeichen)", len(body))
        return []

    return await evaluate_chunk_relevance(
        generated_text=body,
        retrieved_chunks=hits,
        llm=chat_client,
        max_chunks=k_books + k_lectures,
    )
