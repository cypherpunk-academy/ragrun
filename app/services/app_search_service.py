"""Hybrid search for /app/search."""
from __future__ import annotations

from typing import Any, Iterable

from app.config import settings
from app.core.providers import get_embedding_client, get_qdrant_client
from app.retrieval.services.action_prompt_service import load_assistant_embedding_prefixes, load_assistant_rag_collection
from app.retrieval.utils.retrievers import hybrid_retrieve
from app.retrieval.models import RetrievedSnippet

_APP_TYPE_TO_CHUNK_TYPES: dict[str, list[str]] = {
    "text": ["book", "secondary_book", "talk"],
    "concept": ["begriff", "typology"],
    "quote": ["quote", "quote_explanation"],
    "chapter_summary": ["chapter_summary"],
}


def _resolve_chunk_types(types: Iterable[str] | None) -> list[str]:
    if not types:
        return [
            "book",
            "secondary_book",
            "talk",
            "chapter_summary",
            "quote",
            "quote_explanation",
            "begriff",
            "typology",
        ]
    out: list[str] = []
    for t in types:
        key = (t or "").strip().lower()
        if not key:
            continue
        mapped = _APP_TYPE_TO_CHUNK_TYPES.get(key)
        if mapped:
            out.extend(mapped)
        else:
            out.append(key)
    return list(dict.fromkeys(out))


def _meta(snippet: RetrievedSnippet) -> dict[str, Any]:
    payload = snippet.payload
    inner = payload.get("payload") if isinstance(payload.get("payload"), dict) else payload
    return dict(inner) if isinstance(inner, dict) else {}


def _snippet_text(text: str, limit: int = 240) -> str:
    t = (text or "").strip().replace("\n", " ")
    if len(t) <= limit:
        return t
    return t[: limit - 1].rstrip() + "…"


async def app_search(
    *,
    query: str,
    types: list[str] | None = None,
    limit: int = 20,
    collection: str | None = None,
) -> list[dict[str, Any]]:
    q = query.strip()
    if not q:
        return []

    assistant_slug = settings.app_default_assistant_slug
    coll = (collection or "").strip() or load_assistant_rag_collection(assistant_slug)
    chunk_types = _resolve_chunk_types(types)
    k = max(1, min(limit, 50))

    prefix_passage, prefix_query = load_assistant_embedding_prefixes(assistant_slug)
    _ = prefix_passage  # search uses query prefix only

    snippets = await hybrid_retrieve(
        query=q,
        k_dense=k,
        k_sparse=k,
        k_fused=k,
        worldview=None,
        book_types=chunk_types,
        collection=coll,
        embedding_client=get_embedding_client(),
        qdrant_client=get_qdrant_client(),
        query_prefix=prefix_query or None,
        force_sparse=settings.use_hybrid_retrieval,
    )

    results: list[dict[str, Any]] = []
    for snip in snippets:
        meta = _meta(snip)
        chunk_id = meta.get("chunk_id") or snip.payload.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id:
            continue
        source_id = meta.get("source_id")
        text_val = snip.text or ""
        chunk_type = meta.get("chunk_type")
        item: dict[str, Any] = {
            "chunk_id": chunk_id,
            "source_id": str(source_id or ""),
            "segment_id": meta.get("segment_id"),
            "paragraph_id": meta.get("paragraph_id"),
            "title": meta.get("book_title") or meta.get("source_title"),
            "segment_title": meta.get("segment_title"),
            "snippet": _snippet_text(text_val),
            "text": text_val,
            "score": float(snip.score),
            "chunk_type": chunk_type,
            "source_type": meta.get("source_type"),
            "author": meta.get("author"),
            "book_title": meta.get("book_title"),
        }
        if chunk_type in ("quote", "quote_explanation"):
            item["quote_text"] = text_val
            item["quote_author"] = meta.get("author")
        results.append(item)
    return results
