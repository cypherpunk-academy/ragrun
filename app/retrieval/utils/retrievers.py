"""Retrieval helpers for concept-explain-worldviews."""
from __future__ import annotations

import logging
from typing import Iterable, List, Mapping, Sequence, Tuple

from app.config import settings
from app.infra.embedding_client import EmbeddingClient
from app.infra.qdrant_client import QdrantClient
from app.retrieval.models import RetrievedSnippet

logger = logging.getLogger(__name__)


def _extract_chunk_id(payload: Mapping[str, object]) -> str | None:
    inner = payload.get("payload")
    if isinstance(inner, Mapping):
        payload = inner
    cid = payload.get("chunk_id")
    return cid if isinstance(cid, str) else None


async def embed_text(text: str, embedding_client: EmbeddingClient) -> Sequence[float]:
    result = await embedding_client.embed_texts([text])
    return result.embeddings[0]


def payload_filter(worldview: str | None, book_types: Iterable[str]) -> Mapping[str, object]:
    """
    Build a Qdrant payload filter for worldview- and book-type scoped retrieval.

    Note: older ingestions stored the type under `chunk_type` (book/secondary_book)
    rather than `book_type`. We prefer `chunk_type` to avoid empty results.
    """

    must: list[Mapping[str, object]] = []

    if worldview:
        must.append({"key": "worldviews", "match": {"value": worldview}})

    # Map logical book_types to the stored chunk_type values.
    chunk_types: list[str] = []
    for bt in book_types:
        if not bt:
            continue
        if bt == "primary":
            chunk_types.append("book")
        elif bt == "secondary":
            chunk_types.append("secondary_book")
        else:
            # Allow direct chunk_type passthrough (e.g., already "book")
            chunk_types.append(bt)

    if chunk_types:
        if len(chunk_types) == 1:
            must.append({"key": "chunk_type", "match": {"value": chunk_types[0]}})
        else:
            must.append({"key": "chunk_type", "match": {"any": chunk_types}})

    return {"must": must} if must else {}


async def dense_retrieve(
    *,
    query: str,
    k: int,
    worldview: str | None,
    book_types: Iterable[str],
    collection: str,
    embedding_client: EmbeddingClient,
    qdrant_client: QdrantClient,
) -> list[RetrievedSnippet]:
    vector = await embed_text(query, embedding_client)
    hits = await qdrant_client.search_points(
        collection,
        vector=vector,
        limit=k,
        filter_=payload_filter(worldview, book_types),
        with_payload=True,
    )
    out: list[RetrievedSnippet] = []
    for item in hits:
        payload = item.get("payload", {}) if isinstance(item, Mapping) else {}
        text = payload.get("text") or ""
        if not isinstance(text, str) or not text.strip():
            continue
        score = float(item.get("score") or 0.0)
        out.append(RetrievedSnippet(text=text, score=score, payload=item))
    return out


async def sparse_retrieve(
    *,
    query: str,
    k: int,
    worldview: str | None,
    book_types: Iterable[str],
    collection: str,
    qdrant_client: QdrantClient,
) -> list[RetrievedSnippet]:
    """Sparse-only BM25 retrieval with payload filtering."""

    if not hasattr(qdrant_client, "search_sparse_points"):
        raise RuntimeError("Sparse retrieval not supported by qdrant client")

    try:
        sparse_hits = await qdrant_client.search_sparse_points(  # type: ignore[attr-defined]
            collection=collection,
            text=query,
            limit=k,
            filter_=payload_filter(worldview, book_types),
        )
    except Exception:
        logger.exception("Sparse retrieval failed")
        raise

    out: list[RetrievedSnippet] = []
    for item in sparse_hits or []:
        payload = item.get("payload", {}) if isinstance(item, Mapping) else {}
        text_val = payload.get("text") or ""
        if not isinstance(text_val, str) or not text_val.strip():
            continue
        score = float(item.get("score") or 0.0)
        out.append(RetrievedSnippet(text=text_val, score=score, payload=item))
    return out


async def hybrid_retrieve(
    *,
    query: str,
    k_dense: int,
    k_sparse: int,
    k_fused: int,
    worldview: str | None,
    book_types: Iterable[str],
    collection: str,
    embedding_client: EmbeddingClient,
    qdrant_client: QdrantClient,
    force_sparse: bool = False,
) -> list[RetrievedSnippet]:
    """Hybrid dense+BM25 with RRF; falls back to dense-only if sparse unavailable."""

    dense_hits = await dense_retrieve(
        query=query,
        k=k_dense,
        worldview=worldview,
        book_types=book_types,
        collection=collection,
        embedding_client=embedding_client,
        qdrant_client=qdrant_client,
    )

    if (not settings.use_hybrid_retrieval and not force_sparse) or k_sparse <= 0:
        return dense_hits

    sparse_hits: list[RetrievedSnippet] = []
    if hasattr(qdrant_client, "search_sparse_points"):
        try:
            sparse = await qdrant_client.search_sparse_points(  # type: ignore[attr-defined]
                collection=collection,
                text=query,
                limit=k_sparse,
                filter_=payload_filter(worldview, book_types),
            )
            for item in sparse or []:
                payload = item.get("payload", {}) if isinstance(item, Mapping) else {}
                text_val = payload.get("text") or ""
                if not isinstance(text_val, str) or not text_val.strip():
                    continue
                score = float(item.get("score") or 0.0)
                sparse_hits.append(RetrievedSnippet(text=text_val, score=score, payload=item))
        except Exception:
            logger.warning("Hybrid enabled but sparse search failed; falling back to dense-only", exc_info=True)
            return dense_hits
    else:
        logger.info("Hybrid enabled but no sparse retriever available; using dense-only")
        return dense_hits

    fused = _rrf_fuse(dense_hits, sparse_hits, k_fused=k_fused)
    return fused


def _rrf_fuse(
    dense_hits: list[RetrievedSnippet],
    sparse_hits: list[RetrievedSnippet],
    *,
    k_rrf: int = 60,
    k_fused: int = 30,
) -> list[RetrievedSnippet]:
    scores: dict[str, float] = {}
    seen: dict[str, RetrievedSnippet] = {}

    def _bump(items: list[RetrievedSnippet], weight: float = 1.0) -> None:
        for rank, item in enumerate(items, start=1):
            payload = item.payload if isinstance(item.payload, Mapping) else {}
            chunk_id = _extract_chunk_id(payload) or f"idx-{id(item)}-{rank}"
            if chunk_id not in seen:
                seen[chunk_id] = item
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight * (1.0 / (k_rrf + rank))

    _bump(dense_hits, 1.0)
    _bump(sparse_hits, 1.0)

    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    fused: list[RetrievedSnippet] = []
    for chunk_id, _ in ordered[:k_fused]:
        fused.append(seen[chunk_id])
    return fused


async def rerank_by_embedding(
    *,
    query: str,
    snippets: list[RetrievedSnippet],
    embedding_client: EmbeddingClient,
    k_final: int,
) -> list[RetrievedSnippet]:
    """Simple embedding-based reranker."""

    if not snippets:
        return []

    query_vec = await embed_text(query, embedding_client)
    texts = [s.text for s in snippets]
    embeddings = await embedding_client.embed_texts(texts)
    scored: list[Tuple[float, RetrievedSnippet]] = []
    for emb, snippet in zip(embeddings.embeddings, snippets):
        score = _dot(query_vec, emb)
        scored.append((score, snippet))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [s for _, s in scored[:k_final]]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))


def build_context(snippets: Iterable[RetrievedSnippet], max_chars: int = 12000) -> Tuple[str, List[str]]:
    """Render a joined context string and collect chunk_ids for telemetry."""
    parts: list[str] = []
    refs: list[str] = []
    total = 0
    for snip in snippets:
        text = snip.text.strip()
        if not text:
            continue
        cid = _extract_chunk_id(snip.payload) or ""
        if cid:
            refs.append(cid)
        snippet = text
        if total + len(snippet) > max_chars:
            break
        parts.append(snippet)
        total += len(snippet)
    return "\n\n".join(parts), refs
