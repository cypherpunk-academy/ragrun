"""Retrieval helpers for concept-explain-worldviews."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Iterable, List, Mapping, Sequence, Tuple, cast

from app.config import settings
from app.infra.embedding_client import EmbeddingClient
from app.infra.qdrant_client import QdrantClient
from app.retrieval.models import RetrievedSnippet

logger = logging.getLogger(__name__)

# region agent log
_AGENT_DEBUG_LOG_PATH = "/Users/michael/Reniets/Ai/ragkeep/.cursor/debug-d085a7.log"


def _agent_debug_ndjson(hypothesis_id: str, location: str, data: dict) -> None:
    try:
        with open(_AGENT_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "sessionId": "d085a7",
                        "timestamp": int(time.time() * 1000),
                        "hypothesisId": hypothesis_id,
                        "location": location,
                        "message": "typology_retrieval_debug",
                        "data": data,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    except Exception:
        pass


# endregion


def _filter_and_cap_hits(raw_items: list, k: int, *, score_key: str = "score") -> list[RetrievedSnippet]:
    """Over-fetch and filter: keep only chunks with text >= min_chunk_chars, return up to k."""
    out: list[RetrievedSnippet] = []
    min_chars = getattr(settings, "min_chunk_chars", 80)
    for item in raw_items:
        if len(out) >= k:
            break
        payload = item.get("payload", {}) if isinstance(item, Mapping) else {}
        text = payload.get("text") or ""
        if not isinstance(text, str):
            continue
        t = text.strip()
        if not t or len(t) < min_chars:
            continue
        score = float(item.get(score_key, 0.0) or 0.0)
        out.append(RetrievedSnippet(text=text, score=score, payload=item))
    return out


def _extract_chunk_id(payload: Mapping[str, object]) -> str | None:
    inner = payload.get("payload")
    if isinstance(inner, Mapping):
        payload = inner
    cid = payload.get("chunk_id")
    return cid if isinstance(cid, str) else None


async def embed_text(text: str, embedding_client: EmbeddingClient) -> Sequence[float]:
    result = await embedding_client.embed_texts([text])
    return result.embeddings[0]


def payload_filter(
    worldview: str | None,
    book_types: Iterable[str],
    author: str | None = None,
) -> Mapping[str, object]:
    """
    Build a Qdrant payload filter for worldview-, book-type- and author-scoped retrieval.

    - If worldview is set: only chunks where worldviews contains that value.
    - If worldview is None: only chunks where worldviews is null (general, non-worldview-specific).
    - If author is set: only chunks where author matches (e.g. "Rudolf Steiner").

    Note: older ingestions stored the type under `chunk_type` (book/secondary_book)
    rather than `book_type`. We prefer `chunk_type` to avoid empty results.
    """

    must: list[Mapping[str, object]] = []

    if worldview:
        must.append({"key": "worldviews", "match": {"value": worldview}})
    else:
        # Match both null and empty array (ingestion may store either)
        must.append({
            "should": [
                {"is_null": {"key": "worldviews"}},
                {"is_empty": {"key": "worldviews"}},
            ]
        })

    if author:
        must.append({"key": "author", "match": {"value": author}})

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
    author: str | None = None,
    diagnostic_tag: str | None = None,
) -> list[RetrievedSnippet]:
    mult = getattr(settings, "retrieval_overfetch_multiplier", 2)
    limit = max(k * mult, k + 30)
    vector = await embed_text(query, embedding_client)
    pf = payload_filter(worldview, book_types, author=author)
    hits = await qdrant_client.search_points(
        collection,
        vector=vector,
        limit=limit,
        filter_=pf,
        with_payload=True,
    )
    items = list(hits) if hits else []
    capped = _filter_and_cap_hits(items, k)
    # region agent log
    if diagnostic_tag:
        try:
            _agent_debug_ndjson(
                "H_dense_cap",
                "retrievers.py:dense_retrieve",
                {
                    "tag": diagnostic_tag,
                    "collection": collection,
                    "qdrant_hits": len(items),
                    "after_min_chars_cap": len(capped),
                    "min_chunk_chars": getattr(settings, "min_chunk_chars", 80),
                    "payload_filter": pf,
                },
            )
        except Exception:
            pass
    # endregion
    return capped


async def sparse_retrieve(
    *,
    query: str,
    k: int,
    worldview: str | None,
    book_types: Iterable[str],
    collection: str,
    qdrant_client: QdrantClient,
    author: str | None = None,
) -> list[RetrievedSnippet]:
    """Sparse-only BM25 retrieval with payload filtering."""

    if not hasattr(qdrant_client, "search_sparse_points"):
        raise RuntimeError("Sparse retrieval not supported by qdrant client")

    mult = getattr(settings, "retrieval_overfetch_multiplier", 2)
    limit = max(k * mult, k + 30)
    try:
        sparse_raw = await qdrant_client.search_sparse_points(  # type: ignore[attr-defined]
            collection=collection,
            text=query,
            limit=limit,
            filter_=payload_filter(worldview, book_types, author=author),
        )
    except Exception:
        logger.exception("Sparse retrieval failed")
        raise

    items = list(sparse_raw or [])
    return _filter_and_cap_hits(items, k)


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
    author: str | None = None,
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
        author=author,
    )

    if (not settings.use_hybrid_retrieval and not force_sparse) or k_sparse <= 0:
        return dense_hits

    sparse_hits: list[RetrievedSnippet] = []
    if hasattr(qdrant_client, "search_sparse_points"):
        mult = getattr(settings, "retrieval_overfetch_multiplier", 2)
        limit_sparse = max(k_sparse * mult, k_sparse + 30)
        try:
            sparse = await qdrant_client.search_sparse_points(  # type: ignore[attr-defined]
                collection=collection,
                text=query,
                limit=limit_sparse,
                filter_=payload_filter(worldview, book_types, author=author),
            )
            sparse_hits = _filter_and_cap_hits(list(sparse or []), k_sparse)
        except Exception:
            logger.warning("Hybrid enabled but sparse search failed; falling back to dense-only", exc_info=True)
            return dense_hits
    else:
        logger.info("Hybrid enabled but no sparse retriever available; using dense-only")
        return dense_hits

    fused = _rrf_fuse(dense_hits, sparse_hits, k_fused=k_fused)
    return fused


async def hybrid_retrieve_quote_parallel(
    *,
    query: str,
    k_per_branch: int,
    k_fused: int,
    collection: str,
    embedding_client: EmbeddingClient,
    qdrant_client: QdrantClient,
) -> list[RetrievedSnippet]:
    """Zwei parallele Suchen: quote + book/secondary_book (author=Rudolf Steiner)."""

    async def search_quote() -> list[RetrievedSnippet]:
        return await hybrid_retrieve(
            query=query,
            k_dense=k_per_branch,
            k_sparse=k_per_branch,
            k_fused=k_per_branch,
            worldview=None,
            book_types=["quote"],
            collection=collection,
            embedding_client=embedding_client,
            qdrant_client=qdrant_client,
        )

    async def search_steiner_books() -> list[RetrievedSnippet]:
        return await hybrid_retrieve(
            query=query,
            k_dense=k_per_branch,
            k_sparse=k_per_branch,
            k_fused=k_per_branch,
            worldview=None,
            book_types=["book", "secondary_book"],
            author="Rudolf Steiner",
            collection=collection,
            embedding_client=embedding_client,
            qdrant_client=qdrant_client,
        )

    quote_hits, book_hits = await asyncio.gather(search_quote(), search_steiner_books())
    return _rrf_fuse(quote_hits, book_hits, k_fused=k_fused)


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


def _snippet_flat_payload(payload: Mapping[str, object]) -> Mapping[str, object]:
    inner = payload.get("payload")
    return cast(Mapping[str, object], inner) if isinstance(inner, Mapping) else payload


def snippet_source_title(payload: Mapping[str, object]) -> str:
    """Resolve `source_title` from a Qdrant hit payload (flat or nested metadata)."""

    p = _snippet_flat_payload(payload)
    md = p.get("metadata")
    if isinstance(md, Mapping):
        st = md.get("source_title")
        if isinstance(st, str):
            return st
    st = p.get("source_title")
    return st if isinstance(st, str) else ""


def snippet_author(payload: Mapping[str, object]) -> str | None:
    p = _snippet_flat_payload(payload)
    md = p.get("metadata")
    if isinstance(md, Mapping):
        a = md.get("author")
        if isinstance(a, str) and a.strip():
            return a.strip()
    a = p.get("author")
    return a.strip() if isinstance(a, str) and a.strip() else None


def snippet_chunk_type_value(payload: Mapping[str, object]) -> str | None:
    p = _snippet_flat_payload(payload)
    md = p.get("metadata")
    if isinstance(md, Mapping):
        ct = md.get("chunk_type")
        if isinstance(ct, str) and ct.strip():
            return ct.strip()
    ct = p.get("chunk_type")
    return ct.strip() if isinstance(ct, str) and ct.strip() else None


def snippet_source_id(payload: Mapping[str, object]) -> str:
    """Resolve `source_id` from a Qdrant hit payload (aligned with rag_chunks.source_id)."""

    p = _snippet_flat_payload(payload)
    md = p.get("metadata")
    if isinstance(md, Mapping):
        sid = md.get("source_id")
        if isinstance(sid, str) and sid.strip():
            return sid.strip()
    sid = p.get("source_id")
    return sid.strip() if isinstance(sid, str) and sid.strip() else ""


def _normalize_person_name(name: str) -> str:
    return " ".join(name.lower().replace("_", " ").split())


def _source_title_matches_book_prefix(title: str, book_prefix: str) -> bool:
    t = title.strip()
    b = book_prefix.strip()
    if not b:
        return True
    # Direct match
    if t.startswith(b) or t == b:
        return True
    # Book-id format: "Author#Titel#GA" → extract title part (underscores → spaces)
    parts = b.split("#")
    if len(parts) >= 2:
        title_part = parts[1].replace("_", " ")
        if title_part and (t == title_part or t.startswith(title_part)):
            return True
    return False


def filter_snippets_by_typology_source(
    snippets: List[RetrievedSnippet],
    *,
    chunk_types: List[str] | None,
    source_ids: List[str],
) -> List[RetrievedSnippet]:
    """
    Post-filter retrieval hits by source_id (exact UUID match) and optional chunk_type allow-list.
    """

    id_set = {s.strip() for s in source_ids if isinstance(s, str) and s.strip()}
    ct_allow = (
        {c.strip() for c in chunk_types if isinstance(c, str) and c.strip()}
        if chunk_types
        else None
    )

    out: List[RetrievedSnippet] = []
    for snip in snippets:
        payload = snip.payload if isinstance(snip.payload, Mapping) else {}
        if ct_allow:
            ct = snippet_chunk_type_value(payload)
            if not ct or ct not in ct_allow:
                continue
        if id_set:
            sid = snippet_source_id(payload)
            if sid not in id_set:
                continue
        out.append(snip)
    return out


def _diagnose_typology_post_filter(
    snippets: List[RetrievedSnippet],
    *,
    book_prefixes: List[str],
    authors: List[str] | None,
    chunk_types: List[str] | None,
    source_ids: List[str] | None,
) -> dict[str, object]:
    """Count first-failure drops vs filter_snippets_by_typology_source (debug)."""

    id_set = (
        {s.strip() for s in source_ids if isinstance(s, str) and s.strip()}
        if source_ids
        else set()
    )
    prefixes = [p.strip() for p in book_prefixes if p and p.strip()]
    author_allow = (
        {_normalize_person_name(a) for a in authors if isinstance(a, str) and a.strip()}
        if authors
        else None
    )
    ct_allow = (
        {c.strip() for c in chunk_types if isinstance(c, str) and c.strip()}
        if chunk_types
        else None
    )
    drop_ct = 0
    drop_au = 0
    drop_sid = 0
    drop_prefix = 0
    kept = 0
    sample_sid: list[str] = []
    sample_ct: list[str | None] = []
    sample_wv: list[object] = []
    for snip in snippets:
        payload = snip.payload if isinstance(snip.payload, Mapping) else {}
        pflat = _snippet_flat_payload(payload)
        if len(sample_wv) < 5:
            md = pflat.get("metadata") if isinstance(pflat.get("metadata"), Mapping) else None
            wv = md.get("worldviews") if isinstance(md, Mapping) else pflat.get("worldviews")
            sample_wv.append(wv)
        sid = snippet_source_id(payload)
        if len(sample_sid) < 10:
            sample_sid.append(sid)
        ct = snippet_chunk_type_value(payload)
        if len(sample_ct) < 10:
            sample_ct.append(ct)
        if ct_allow:
            if not ct or ct not in ct_allow:
                drop_ct += 1
                continue
        if author_allow:
            au = snippet_author(payload)
            if au and _normalize_person_name(au) not in author_allow:
                drop_au += 1
                continue
        if id_set:
            if sid not in id_set:
                drop_sid += 1
                continue
        else:
            st = snippet_source_title(payload)
            if prefixes:
                if not any(_source_title_matches_book_prefix(st, pref) for pref in prefixes):
                    drop_prefix += 1
                    continue
        kept += 1
    return {
        "n_snippets": len(snippets),
        "drop_chunk_type": drop_ct,
        "drop_author": drop_au,
        "drop_source_id": drop_sid,
        "drop_prefix": drop_prefix,
        "kept": kept,
        "sample_source_ids": sample_sid,
        "sample_chunk_types": sample_ct,
        "sample_worldviews": sample_wv,
        "id_set_size": len(id_set),
    }


async def typology_dense_retrieve(
    *,
    query: str,
    k: int,
    worldview: str | None,
    book_types: Iterable[str],
    collection: str,
    embedding_client: EmbeddingClient,
    qdrant_client: QdrantClient,
    source_ids: List[str],
    chunk_types: List[str] | None = None,
) -> List[RetrievedSnippet]:
    """
    Dense search with over-fetch, then post-filter by source_id UUID and chunk_type.
    """

    if k <= 0:
        return []

    multipliers = (3, 6, 12, 24)
    last_filtered: List[RetrievedSnippet] = []

    for mult in multipliers:
        fetch_k = min(max(k * mult, k + 20), 400)
        raw = await dense_retrieve(
            query=query,
            k=fetch_k,
            worldview=worldview,
            book_types=book_types,
            collection=collection,
            embedding_client=embedding_client,
            qdrant_client=qdrant_client,
            diagnostic_tag=f"typology_dense_mult{mult}",
        )
        filtered = filter_snippets_by_typology_source(
            raw,
            chunk_types=chunk_types,
            source_ids=source_ids,
        )
        last_filtered = filtered
        if len(filtered) >= min(k, 6) or mult == multipliers[-1]:
            break

    return last_filtered[:k]


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


def build_context_numbered(
    snippets: Iterable[RetrievedSnippet],
    start_index: int = 1,
    max_chars: int = 12000,
    include_score: bool = False,
) -> Tuple[str, List[str], List[Tuple[int, str, str, Mapping[str, object]]]]:
    """Like build_context but prefixes each chunk with [N]. Returns (context, refs, index_map).
    index_map: list of (index, chunk_id, text, payload_meta) for citation linking.
    If include_score, each chunk is prefixed with (r: X.X) for retrieval relevance."""
    parts: list[str] = []
    refs: list[str] = []
    index_map: list[Tuple[int, str, str, Mapping[str, object]]] = []
    total = 0
    for idx, snip in enumerate(snippets):
        text = snip.text.strip()
        if not text:
            continue
        num = start_index + len(parts)
        payload = snip.payload if isinstance(snip.payload, Mapping) else {}
        meta = payload.get("payload", payload) if isinstance(payload.get("payload"), Mapping) else payload
        if not isinstance(meta, dict):
            meta = {}
        cid = _extract_chunk_id(snip.payload) or ""
        if cid:
            refs.append(cid)
        index_map.append((num, cid, text, meta))
        if include_score:
            chunk_type = snippet_chunk_type_value(payload) or ""
            # Plain text only — rp ragChat colors (r: …) and chunk_type in the CLI; embedded
            # ANSI resets here would turn the rest of the line white after the orange label.
            ct_str = f", {chunk_type}" if chunk_type else ""
            score_str = f" (r: {snip.score:.1f}{ct_str})"
        else:
            score_str = ""
        prefixed = f"[{num}]{score_str} {text}"
        if total + len(prefixed) > max_chars:
            break
        parts.append(prefixed)
        total += len(prefixed)
    return "\n\n".join(parts), refs, index_map
