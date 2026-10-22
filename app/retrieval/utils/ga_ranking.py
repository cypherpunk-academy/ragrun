"""Ranking pipeline for the Qdrant collection ``rudolf-steiner-ga``.

Stages: deep hybrid candidates, RRF fusion, title signal (pool plus catalog
path), optional cross-encoder, then one hit per lecture and a cap per GA volume.
``hybrid_retrieve`` is left unchanged; other chains still call it.

``search_ga`` and ``search_ga_summaries`` share these weights. The title score
reads ``vortragstitel`` (or ``segment_title`` on book volumes), not the chunk
body, so summaries do not need a separate weight: the title signal is the same
even when the summary text already contains the title.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from app.config import settings
from app.retrieval.models import RetrievedSnippet
from app.retrieval.utils.retrievers import (
    _extract_chunk_id,
    _filter_and_cap_hits,
    _rrf_fuse,
    dense_retrieve,
    embed_text,
    payload_filter,
    sparse_retrieve,
)

logger = logging.getLogger(__name__)

K_RRF = 60
_TITLE_PATH_LIMIT = 10
_CATALOG_TTL_S = 300.0

_FOLD = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"})
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)

_STOPWORDS = frozenset(
    {
        "der", "die", "das", "den", "dem", "des",
        "ein", "eine", "einer", "einem", "einen", "eines",
        "und", "oder", "aber", "denn", "doch", "nur", "auch", "noch", "schon", "nicht",
        "so", "wie", "als", "wenn", "weil", "dass", "ob",
        "in", "im", "ins", "an", "am", "ans", "auf", "aufs", "aus", "bei", "beim",
        "mit", "von", "vom", "vor", "nach", "zu", "zum", "zur", "ueber", "unter", "durch",
        "fuer", "gegen", "ohne", "um", "bis",
        "ist", "sind", "war", "wird", "werden", "sein", "hat", "haben",
        "sich", "es", "er", "sie", "wir", "ihr", "ihnen", "ihre", "seine", "seinem", "seinen",
        "dieser", "diese", "dieses", "diesem", "diesen",
        "jener", "jene", "jenes",
        "was", "wer", "wo", "wem", "wen",
        "the", "and", "for", "with", "from", "that", "this",
    }
)

_catalog_cache: tuple[float, list[dict[str, str]]] | None = None


@dataclass
class GaRankedHit:
    snippet: RetrievedSnippet
    meta: dict[str, Any]
    rrf_norm: float
    title_score: float
    score: float
    ordinal: int
    group: str
    lecture_hits: int = 1
    rerank: float | None = None


def fold_text(text: str) -> str:
    """Lowercase, fold umlauts and ß, drop punctuation."""
    folded = (text or "").lower().translate(_FOLD)
    folded = _PUNCT.sub(" ", folded)
    return re.sub(r"\s+", " ", folded).strip()


def query_tokens(query: str) -> list[str]:
    """Query tokens after folding, stopword removal, and the length cutoff."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in fold_text(query).split():
        if raw in _STOPWORDS or len(raw) < 3 or raw in seen:
            continue
        seen.add(raw)
        out.append(raw)
    return out


def token_matches(token: str, word: str) -> bool:
    """Exact match, or a prefix either way when the shorter side has 4+ characters.

    ``karma`` matches ``karmabildung``. ``gesetze`` does not match
    ``gesetzmäßigkeiten``; ``gesetz`` does.
    """
    if token == word:
        return True
    if len(token) >= 4 and word.startswith(token):
        return True
    if len(word) >= 4 and token.startswith(word):
        return True
    return False


def title_score(query: str, title: str) -> float:
    """Share of query tokens found in the title. Exact title equality is 1."""
    folded_query = fold_text(query)
    folded_title = fold_text(title)
    if folded_query and folded_query == folded_title:
        return 1.0
    tokens = query_tokens(query)
    if not tokens or not folded_title:
        return 0.0
    words = folded_title.split()
    hits = sum(1 for token in tokens if any(token_matches(token, word) for word in words))
    return hits / len(tokens)


def normalize_rrf(score: float, *, k_rrf: int = K_RRF) -> float:
    """Map an RRF score onto 0..1. Rank 1 in both lists is 1."""
    peak = 2.0 / (k_rrf + 1)
    if peak <= 0:
        return 0.0
    return float(score) / peak


def minmax_norm(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi <= lo:
        return [1.0 for _ in values]
    span = hi - lo
    return [(float(v) - lo) / span for v in values]


def snippet_meta(snippet: RetrievedSnippet) -> dict[str, Any]:
    payload = snippet.payload if isinstance(snippet.payload, Mapping) else {}
    inner = payload.get("payload") if isinstance(payload.get("payload"), Mapping) else payload
    meta = dict(inner) if isinstance(inner, Mapping) else {}
    if not meta.get("text") and snippet.text:
        meta["text"] = snippet.text
    if not meta.get("chunk_id"):
        cid = _extract_chunk_id(payload if isinstance(payload, Mapping) else {})
        if cid:
            meta["chunk_id"] = cid
    return meta


def group_key(meta: Mapping[str, Any], ordinal: int) -> str:
    lecture_id = str(meta.get("lecture_id") or "").strip()
    if lecture_id:
        return f"l:{lecture_id}"
    segment_id = str(meta.get("segment_id") or "").strip()
    if segment_id:
        return f"s:{segment_id}"
    chunk_id = str(meta.get("chunk_id") or "").strip()
    if chunk_id:
        return f"c:{chunk_id}"
    return f"o:{ordinal}"


def _ga_parts(raw: object) -> set[str]:
    return {part.strip().lower() for part in str(raw or "").split(",") if part.strip()}


def _chunk_id_of(snippet: RetrievedSnippet, ordinal: int) -> str:
    meta = snippet_meta(snippet)
    cid = str(meta.get("chunk_id") or "").strip()
    return cid or f"o:{ordinal}"


def _score_snippet(
    snippet: RetrievedSnippet,
    ordinal: int,
    *,
    query: str,
    rrf_norm: float,
    title_weight: float,
) -> GaRankedHit:
    meta = snippet_meta(snippet)
    title = str(meta.get("vortragstitel") or meta.get("segment_title") or "")
    t_score = title_score(query, title)
    return GaRankedHit(
        snippet=snippet,
        meta=meta,
        rrf_norm=rrf_norm,
        title_score=t_score,
        score=rrf_norm + title_weight * t_score,
        ordinal=ordinal,
        group=group_key(meta, ordinal),
    )


def prepare_stage4(
    fused: Sequence[RetrievedSnippet],
    extras: Sequence[RetrievedSnippet],
    *,
    query: str,
    title_weight: float,
    k_rrf: int = K_RRF,
) -> list[GaRankedHit]:
    """Normalize RRF, add the title signal, count chunks per lecture, sort."""
    items: list[GaRankedHit] = []
    seen: set[str] = set()
    for snippet in fused:
        cid = _chunk_id_of(snippet, len(items))
        if cid in seen:
            continue
        seen.add(cid)
        items.append(
            _score_snippet(
                snippet,
                len(items),
                query=query,
                rrf_norm=normalize_rrf(snippet.score, k_rrf=k_rrf),
                title_weight=title_weight,
            )
        )
    for snippet in extras:
        cid = _chunk_id_of(snippet, len(items))
        if cid in seen:
            continue
        seen.add(cid)
        items.append(
            _score_snippet(
                snippet,
                len(items),
                query=query,
                rrf_norm=0.0,
                title_weight=title_weight,
            )
        )
    counts: dict[str, int] = {}
    for item in items:
        counts[item.group] = counts.get(item.group, 0) + 1
    for item in items:
        item.lecture_hits = counts[item.group]
    items.sort(key=lambda hit: (-hit.score, hit.ordinal))
    return items


def finish_stage4(
    items: list[GaRankedHit],
    *,
    rerank_norms: Sequence[float] | None,
    rerank_top: int,
    group_by_lecture: bool,
    max_per_ga: int,
    apply_ga_cap: bool,
    limit: int,
) -> list[GaRankedHit]:
    """Optional rerank mix, then grouping, volume cap, and the caller limit."""
    if rerank_norms:
        top = items[: max(0, rerank_top)]
        for item, norm in zip(top, rerank_norms):
            item.rerank = float(norm)
            item.score = 0.6 * item.rerank + 0.4 * item.score
        items.sort(key=lambda hit: (-hit.score, hit.ordinal))

    if group_by_lecture:
        chosen: list[GaRankedHit] = []
        seen_groups: set[str] = set()
        for item in items:
            if item.group in seen_groups:
                continue
            seen_groups.add(item.group)
            chosen.append(item)
    else:
        chosen = list(items)

    if apply_ga_cap and max_per_ga > 0:
        per_ga: dict[str, int] = {}
        capped: list[GaRankedHit] = []
        for item in chosen:
            ga = str(item.meta.get("ga") or "")
            used = per_ga.get(ga, 0)
            if used >= max_per_ga:
                continue
            per_ga[ga] = used + 1
            capped.append(item)
        chosen = capped

    if limit < 0:
        return chosen
    return chosen[:limit]


def rank_ga_pool(
    fused: Sequence[RetrievedSnippet],
    extras: Sequence[RetrievedSnippet] = (),
    *,
    query: str,
    title_weight: float,
    group_by_lecture: bool = True,
    max_per_ga: int = 3,
    apply_ga_cap: bool = True,
    limit: int = 10,
    rerank_norms: Sequence[float] | None = None,
    rerank_top: int = 50,
    k_rrf: int = K_RRF,
) -> list[GaRankedHit]:
    """Rank an already fused pool. Used by the pipeline and by unit tests."""
    items = prepare_stage4(
        fused,
        extras,
        query=query,
        title_weight=title_weight,
        k_rrf=k_rrf,
    )
    return finish_stage4(
        items,
        rerank_norms=rerank_norms,
        rerank_top=rerank_top,
        group_by_lecture=group_by_lecture,
        max_per_ga=max_per_ga,
        apply_ga_cap=apply_ga_cap,
        limit=limit,
    )


def select_title_lectures(
    rows: Sequence[Mapping[str, Any]],
    query: str,
    *,
    exclude: set[str],
    gas: Sequence[str] | None,
    limit: int = _TITLE_PATH_LIMIT,
) -> list[dict[str, Any]]:
    """Up to ``limit`` catalog lectures whose titles match and that are not in the pool."""
    if limit <= 0 or not fold_text(query):
        return []
    allowed = {part.lower() for part in gas} if gas else None
    scored: list[tuple[float, str, dict[str, Any]]] = []
    for row in rows:
        lecture_id = str(row.get("id") or "").strip()
        title = str(row.get("vortragstitel") or "").strip()
        if not lecture_id or not title or lecture_id in exclude:
            continue
        if allowed is not None and not (_ga_parts(row.get("ga")) & allowed):
            continue
        score = title_score(query, title)
        if score <= 0:
            continue
        scored.append(
            (
                score,
                lecture_id,
                {
                    "id": lecture_id,
                    "ga": str(row.get("ga") or ""),
                    "vortragstitel": title,
                    "title_score": score,
                },
            )
        )
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [row for _, _, row in scored[:limit]]


def _load_catalog_titles() -> list[dict[str, str]]:
    """Lecture id, GA and title from ``rag_lecture_catalog``. Cached for a few minutes.

    ``has_chunks`` is not filtered: in the current catalog that flag is false for
    every row, including lectures that are embedded in Qdrant.
    """
    global _catalog_cache
    now = time.monotonic()
    if _catalog_cache is not None and now - _catalog_cache[0] < _CATALOG_TTL_S:
        return _catalog_cache[1]

    from sqlalchemy import text

    from app.db.session import get_engine

    with get_engine().connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT id, ga, vortragstitel
                FROM rag_lecture_catalog
                WHERE vortragstitel IS NOT NULL
                  AND btrim(vortragstitel) <> ''
                """
            )
        ).mappings().all()
    loaded = [
        {
            "id": str(row["id"]),
            "ga": str(row["ga"] or ""),
            "vortragstitel": str(row["vortragstitel"] or ""),
        }
        for row in rows
    ]
    _catalog_cache = (now, loaded)
    return loaded


async def find_title_lectures(
    query: str,
    *,
    exclude: set[str],
    gas: Sequence[str] | None,
    limit: int = _TITLE_PATH_LIMIT,
) -> list[dict[str, Any]]:
    """Catalog lookup. Missing table or a failed query: warning, empty list."""
    try:
        rows = await asyncio.to_thread(_load_catalog_titles)
    except Exception as exc:
        logger.warning("GA title path skipped: %s", exc)
        return []
    return select_title_lectures(rows, query, exclude=exclude, gas=gas, limit=limit)


async def rerank_texts(
    query: str,
    texts: Sequence[str],
    *,
    url: str,
    timeout_s: float,
) -> list[float] | None:
    """Cross-encoder scores normalized to 0..1. Failure or timeout returns None."""
    if not texts:
        return None
    if not url:
        logger.warning(
            "GA rerank enabled but RAGRUN_GA_SEARCH_RERANK_URL is empty; continuing without rerank"
        )
        return None
    import httpx

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_s)) as client:
            response = await client.post(url, json={"query": query, "texts": list(texts)})
            response.raise_for_status()
            data = response.json()
        raw = data.get("scores") if isinstance(data, dict) else None
        if not isinstance(raw, list) or len(raw) != len(texts):
            logger.warning("GA rerank response unusable; continuing without rerank")
            return None
        return minmax_norm([float(value) for value in raw])
    except Exception as exc:
        logger.warning("GA rerank skipped: %s", exc)
        return None


def _pool_lecture_ids(fused: Sequence[RetrievedSnippet]) -> set[str]:
    found: set[str] = set()
    for snippet in fused:
        lecture_id = str(snippet_meta(snippet).get("lecture_id") or "").strip()
        if lecture_id:
            found.add(lecture_id)
    return found


def _with_catalog_fields(snippet: RetrievedSnippet, lecture: Mapping[str, Any]) -> RetrievedSnippet:
    """Fill title, lecture id and GA when the Qdrant payload left them empty."""
    meta = snippet_meta(snippet)
    title = str(lecture.get("vortragstitel") or "").strip()
    lecture_id = str(lecture.get("id") or "").strip()
    ga = str(lecture.get("ga") or "").strip()
    if meta.get("vortragstitel") or meta.get("segment_title"):
        title = ""
    if str(meta.get("lecture_id") or "").strip():
        lecture_id = ""
    if str(meta.get("ga") or "").strip():
        ga = ""
    if not title and not lecture_id and not ga:
        return snippet
    payload = dict(snippet.payload) if isinstance(snippet.payload, Mapping) else {}
    inner = payload.get("payload")
    if isinstance(inner, Mapping):
        copied = dict(inner)
        if title:
            copied["vortragstitel"] = title
        if lecture_id:
            copied["lecture_id"] = lecture_id
        if ga:
            copied["ga"] = ga
        payload = dict(payload)
        payload["payload"] = copied
    else:
        if title:
            payload["vortragstitel"] = title
        if lecture_id:
            payload["lecture_id"] = lecture_id
        if ga:
            payload["ga"] = ga
    return RetrievedSnippet(text=snippet.text, score=snippet.score, payload=payload)


async def _chunks_for_title_lectures(
    lectures: Sequence[Mapping[str, Any]],
    *,
    query_vector: Sequence[float],
    chunk_types: Sequence[str],
    extra_must: Sequence[Mapping[str, Any]] | None,
    collection: str,
    qdrant_client: Any,
) -> list[RetrievedSnippet]:
    base_must = list(extra_must or [])

    async def one(lecture: Mapping[str, Any]) -> RetrievedSnippet | None:
        lecture_id = str(lecture.get("id") or "").strip()
        if not lecture_id:
            return None
        must = list(base_must)
        must.append({"key": "lecture_id", "match": {"value": lecture_id}})
        try:
            raw = await qdrant_client.search_points(
                collection,
                query_vector,
                limit=8,
                filter_=payload_filter(None, chunk_types, extra_must=must),
                with_payload=True,
            )
        except Exception as exc:
            logger.warning("GA title path chunk lookup failed for %s: %s", lecture_id, exc)
            return None
        capped = _filter_and_cap_hits(list(raw or []), 1)
        if not capped:
            return None
        return _with_catalog_fields(capped[0], lecture)

    found = await asyncio.gather(*(one(lecture) for lecture in lectures))
    return [snippet for snippet in found if snippet is not None]


async def ga_search_pipeline(
    *,
    query: str,
    chunk_types: Sequence[str],
    extra_must: Sequence[Mapping[str, Any]] | None,
    collection: str,
    embedding_client: Any,
    qdrant_client: Any,
    sparse_embedder: Any,
    force_sparse: bool,
    query_prefix: str,
    limit: int,
    candidates: int,
    pool: int,
    title_weight: float,
    use_title_path: bool,
    group_by_lecture: bool,
    max_per_ga: int,
    apply_ga_cap: bool,
    gas: Sequence[str] | None,
    rerank: bool,
    rerank_top: int,
    rerank_timeout_s: float,
    rerank_url: str,
) -> list[GaRankedHit]:
    """Full GA ranking. The query vector is embedded once and reused."""
    vector = await embed_text(query, embedding_client, query_prefix=query_prefix)

    async def _sparse() -> list[RetrievedSnippet]:
        if not force_sparse and not settings.use_hybrid_retrieval:
            return []
        try:
            return await sparse_retrieve(
                query=query,
                k=candidates,
                worldview=None,
                book_types=chunk_types,
                collection=collection,
                qdrant_client=qdrant_client,
                sparse_embedder=sparse_embedder,
                extra_must=extra_must,
            )
        except Exception as exc:
            logger.warning("GA sparse search failed; dense only: %s", exc)
            return []

    dense_hits, sparse_hits = await asyncio.gather(
        dense_retrieve(
            query=query,
            k=candidates,
            worldview=None,
            book_types=chunk_types,
            collection=collection,
            embedding_client=embedding_client,
            qdrant_client=qdrant_client,
            query_prefix=query_prefix,
            extra_must=extra_must,
            query_vector=vector,
        ),
        _sparse(),
    )
    fused = _rrf_fuse(dense_hits, sparse_hits, k_rrf=K_RRF, k_fused=pool)

    extras: list[RetrievedSnippet] = []
    if use_title_path and title_weight > 0:
        try:
            lectures = await find_title_lectures(
                query,
                exclude=_pool_lecture_ids(fused),
                gas=gas,
            )
            extras = await _chunks_for_title_lectures(
                lectures,
                query_vector=vector,
                chunk_types=chunk_types,
                extra_must=extra_must,
                collection=collection,
                qdrant_client=qdrant_client,
            )
        except Exception as exc:
            logger.warning("GA title path skipped: %s", exc)
            extras = []

    items = prepare_stage4(fused, extras, query=query, title_weight=title_weight)
    norms: list[float] | None = None
    if rerank and items:
        top = items[: max(0, rerank_top)]
        norms = await rerank_texts(
            query,
            [hit.snippet.text for hit in top],
            url=rerank_url,
            timeout_s=rerank_timeout_s,
        )
    return finish_stage4(
        items,
        rerank_norms=norms,
        rerank_top=rerank_top,
        group_by_lecture=group_by_lecture,
        max_per_ga=max_per_ga,
        apply_ga_cap=apply_ga_cap,
        limit=limit,
    )
