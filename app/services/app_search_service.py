"""Hybrid search for /app/search."""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Iterable

from sqlalchemy import text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

from app.config import settings
from app.core.providers import get_embedding_client, get_qdrant_client
from app.retrieval.services.action_prompt_service import load_assistant_embedding_prefixes, load_assistant_rag_collection
from app.mcp_server.typology_tools import typology_fields_for_search_result
from app.retrieval.utils.retrievers import hybrid_retrieve
from app.retrieval.models import RetrievedSnippet
from app.services.corpus_search_policy import (
    apply_quote_hit_policy,
    chunk_types_for_title_search,
)

_APP_TYPE_TO_CHUNK_TYPES: dict[str, list[str]] = {
    "text": ["book", "secondary_book"],
    "concept": ["begriff", "typology"],
    "quote": ["quote", "quote_explanation"],
    "chapter_summary": ["chapter_summary"],
    "curated_text": ["curated_text"],
    # Only meaningful with collection=rudolf-steiner-ga (augment GA-Angaben fallback).
    "ga": ["ga_text"],
}

ALLOWED_APP_SEARCH_TYPES = frozenset(_APP_TYPE_TO_CHUNK_TYPES.keys())

_NAVIGATION_CHUNK_TYPES = frozenset({
    "book",
    "secondary_book",
    "quote",
    "quote_explanation",
})

_BODY_CHUNK_TYPES = frozenset({"book", "secondary_book"})


def _normalize_catalog_quotes(raw: Any) -> list[dict[str, Any]] | None:
    """Validate metadata.quote_marks list for body hits; None if empty/invalid."""
    if not isinstance(raw, list) or not raw:
        return None
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        qid = item.get("id")
        pid = item.get("paragraph_id")
        span = item.get("span")
        if not isinstance(qid, str) or not qid.strip():
            continue
        if not isinstance(pid, str) or not pid.strip():
            continue
        if not isinstance(span, dict):
            continue
        try:
            start = int(span["start"])
            end = int(span["end"])
        except (KeyError, TypeError, ValueError):
            continue
        author = item.get("author")
        if not isinstance(author, str) or not author.strip():
            continue
        out.append(
            {
                "id": qid.strip(),
                "author": author.strip(),
                "paragraph_id": pid.strip(),
                "span": {"start": start, "end": end},
            }
        )
    return out or None


async def _enrich_catalog_quotes(
    results: list[dict[str, Any]],
    engine: Engine,
) -> None:
    """Attach catalog quote marks from rag_chunks.metadata (SSOT after quotes:mark)."""
    body_ids = [
        str(r.get("chunk_id") or "")
        for r in results
        if str(r.get("chunk_type") or "") in _BODY_CHUNK_TYPES and r.get("chunk_id")
    ]
    body_ids = list(dict.fromkeys(cid for cid in body_ids if cid))
    if not body_ids:
        return

    def _query() -> dict[str, Any]:
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT chunk_id,
                           CASE
                             WHEN jsonb_typeof(metadata->'quote_marks') = 'array'
                               THEN metadata->'quote_marks'
                             WHEN jsonb_typeof(metadata->'quotes') = 'array'
                               THEN metadata->'quotes'
                             ELSE NULL
                           END AS quote_marks
                    FROM rag_chunks
                    WHERE chunk_id = ANY(:ids)
                      AND deprecated_at IS NULL
                    """
                ),
                {"ids": body_ids},
            ).mappings().all()
        return {str(r["chunk_id"]): r.get("quote_marks") for r in rows}

    by_id = await asyncio.to_thread(_query)
    for r in results:
        if str(r.get("chunk_type") or "") not in _BODY_CHUNK_TYPES:
            continue
        cid = str(r.get("chunk_id") or "")
        normalized = _normalize_catalog_quotes(by_id.get(cid))
        if normalized:
            r["quotes"] = normalized
        else:
            r.pop("quotes", None)


def _resolve_chunk_types(types: Iterable[str] | None) -> list[str]:
    if not types:
        return [
            "book",
            "secondary_book",
            "chapter_summary",
            "quote",
            "quote_explanation",
            "begriff",
            "typology",
            "curated_text",
        ]
    out: list[str] = []
    for t in types:
        key = (t or "").strip().lower()
        if not key:
            continue
        mapped = _APP_TYPE_TO_CHUNK_TYPES.get(key)
        if mapped:
            out.extend(mapped)
        # Unknown app types are ignored here; MCP validates and errors separately.
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


def _chunk_ids_for_paragraph_lookup(results: list[dict[str, Any]]) -> list[str]:
    """All chunk types that navigate to a paragraph via app_paragraph_chunk."""
    return [
        str(r["chunk_id"])
        for r in results
        if r.get("chunk_id")
        and str(r.get("chunk_type") or "") in _NAVIGATION_CHUNK_TYPES
    ]


async def _lookup_paragraph_ids(
    chunk_ids: list[str],
    engine: Engine,
) -> dict[str, str]:
    """Returns {chunk_id: paragraph_id} for the lowest-numbered active paragraph of each chunk."""
    if not chunk_ids:
        return {}

    def _query() -> dict[str, str]:
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT DISTINCT ON (apc.chunk_id)
                        apc.chunk_id,
                        apc.paragraph_id
                    FROM app_paragraph_chunk apc
                    JOIN rag_paragraphs rp ON rp.id::text = apc.paragraph_id::text
                    WHERE apc.chunk_id = ANY(:ids)
                      AND rp.deprecated_at IS NULL
                    ORDER BY apc.chunk_id, rp.paragraph_number ASC
                    """
                ),
                {"ids": chunk_ids},
            ).mappings().all()
        return {str(r["chunk_id"]): str(r["paragraph_id"]) for r in rows}

    return await asyncio.to_thread(_query)


async def _lookup_source_ids_for_paragraphs(
    paragraph_ids: list[str],
    engine: Engine,
) -> dict[str, str]:
    """Returns {paragraph_id: source_id} from rag_paragraphs for active paragraphs."""
    if not paragraph_ids:
        return {}

    def _query() -> dict[str, str]:
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT id::text AS paragraph_id, source_id
                    FROM rag_paragraphs
                    WHERE id::text = ANY(:ids)
                      AND deprecated_at IS NULL
                    """
                ),
                {"ids": paragraph_ids},
            ).mappings().all()
        return {str(r["paragraph_id"]): str(r["source_id"]) for r in rows}

    return await asyncio.to_thread(_query)


def _parent_id_from_meta(meta: dict[str, Any]) -> str | None:
    parent_id = meta.get("parent_id")
    if isinstance(parent_id, str) and parent_id.strip():
        return parent_id.strip()
    return None


def _quote_para_from_metadata(meta: dict[str, Any]) -> int | None:
    para = meta.get("paragraph")
    if para is not None:
        return int(para)
    return None


def _apply_quote_nav_fields_from_meta(hit: dict[str, Any], meta: dict[str, Any]) -> None:
    """Copy paragraph navigation fields from chunk metadata onto a search hit."""
    para = _quote_para_from_metadata(meta)
    if para is not None:
        hit["paragraph_number"] = para
    paragraph_id = meta.get("paragraph_id")
    if isinstance(paragraph_id, str) and paragraph_id.strip():
        hit["paragraph_id"] = paragraph_id.strip()
    quote_span = meta.get("quote_span")
    if isinstance(quote_span, dict):
        start = quote_span.get("start")
        end = quote_span.get("end")
        if isinstance(start, int) and isinstance(end, int):
            hit["quote_span"] = {"start": start, "end": end}
    if "quote_verified" in meta:
        hit["quote_verified"] = bool(meta.get("quote_verified"))


def _apply_parent_quote_hit(hit: dict[str, Any], parent: dict[str, Any]) -> None:
    """Replace a quote_explanation search hit with its parent quote chunk."""
    parent_meta = parent.get("metadata")
    if not isinstance(parent_meta, dict):
        parent_meta = {}

    text_val = str(parent.get("text") or "")
    hit["chunk_id"] = str(parent["chunk_id"])
    hit["chunk_type"] = "quote"
    if parent.get("source_id"):
        hit["source_id"] = str(parent["source_id"])

    hit["text"] = text_val
    hit["snippet"] = _snippet_text(text_val)
    hit["quote_text"] = text_val

    for key, target in (
        ("segment_id", "segment_id"),
        ("segment_title", "segment_title"),
        ("author", "quote_author"),
        ("author", "author"),
        ("book_title", "book_title"),
        ("source_title", "title"),
    ):
        val = parent_meta.get(key)
        if val is not None and str(val).strip():
            hit[target] = val

    _apply_quote_nav_fields_from_meta(hit, parent_meta)

    hit.pop("_parent_id", None)
    hit.pop("parent_id", None)


_MAX_SUMMARY_RESULTS = 3


def _cap_summary_results(
    results: list[dict[str, Any]],
    max_summaries: int = _MAX_SUMMARY_RESULTS,
) -> list[dict[str, Any]]:
    """Keep at most *max_summaries* chapter_summary hits, preserving order."""
    out: list[dict[str, Any]] = []
    summary_count = 0
    for r in results:
        if str(r.get("chunk_type") or "") == "chapter_summary":
            summary_count += 1
            if summary_count > max_summaries:
                continue
        out.append(r)
    return out


def _dedupe_results_by_chunk_id(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one hit per chunk_id, preferring the highest score."""
    best: dict[str, dict[str, Any]] = {}
    for r in results:
        cid = str(r.get("chunk_id") or "")
        if not cid:
            continue
        prev = best.get(cid)
        if prev is None or float(r.get("score", 0)) > float(prev.get("score", 0)):
            best[cid] = r

    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for r in results:
        cid = str(r.get("chunk_id") or "")
        if not cid or cid in seen:
            continue
        if best.get(cid) is r:
            out.append(r)
            seen.add(cid)
    return out


async def _resolve_quote_explanations(
    results: list[dict[str, Any]],
    engine: Engine,
) -> None:
    """Swap quote_explanation hits for parent quote text — explanations are retrieval-only."""
    expl_indices = [
        i
        for i, r in enumerate(results)
        if str(r.get("chunk_type") or "") == "quote_explanation"
    ]
    if not expl_indices:
        return

    parent_ids = [
        str(results[i]["_parent_id"])
        for i in expl_indices
        if results[i].get("_parent_id")
    ]
    parents: dict[str, dict[str, Any]] = {}
    if parent_ids:
        unique_ids = list(dict.fromkeys(parent_ids))

        def _query() -> dict[str, dict[str, Any]]:
            with engine.connect() as conn:
                rows = conn.execute(
                    text(
                        """
                        SELECT chunk_id, text, metadata, source_id, chunk_type
                        FROM rag_chunks
                        WHERE chunk_id = ANY(:ids)
                        """
                    ),
                    {"ids": unique_ids},
                ).mappings().all()
            return {str(r["chunk_id"]): dict(r) for r in rows}

        parents = await asyncio.to_thread(_query)

    remove_indices: list[int] = []
    for i in expl_indices:
        hit = results[i]
        parent_id = str(hit.get("_parent_id") or "")
        parent = parents.get(parent_id)
        if parent is None or str(parent.get("chunk_type") or "") != "quote":
            remove_indices.append(i)
            logger.warning(
                "dropping quote_explanation hit %s — parent quote %s not found",
                hit.get("chunk_id"),
                parent_id or "(missing parent_id)",
            )
            continue
        _apply_parent_quote_hit(hit, parent)

    for i in reversed(remove_indices):
        results.pop(i)

    resolved = _dedupe_results_by_chunk_id(results)
    results.clear()
    results.extend(resolved)


def _set_navigation_error(hit: dict[str, Any], message: str) -> None:
    hit["navigation_error"] = message
    hit.pop("paragraph_id", None)


async def _title_match_search(
    query: str,
    chunk_types: list[str],
    engine: Engine,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Find chunks whose segment_title or source_title matches the query (ILIKE).

    Returns result dicts in the same shape as the hybrid search output, with a
    synthetic high score so title matches rank above typical RRF scores.
    """
    q_like = f"%{query}%"

    def _query() -> list[dict[str, Any]]:
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT chunk_id, source_id, chunk_type, text, metadata
                    FROM rag_chunks
                    WHERE chunk_type = ANY(:types)
                      AND (
                        metadata->>'segment_title' ILIKE :q
                        OR metadata->>'source_title' ILIKE :q
                      )
                    ORDER BY
                      CASE WHEN LOWER(metadata->>'segment_title') = LOWER(:exact) THEN 0
                           WHEN LOWER(metadata->>'source_title') = LOWER(:exact) THEN 1
                           ELSE 2
                      END,
                      updated_at DESC
                    LIMIT :lim
                    """
                ),
                {"types": chunk_types, "q": q_like, "exact": query, "lim": limit},
            ).mappings().all()
        return [dict(r) for r in rows]

    return await asyncio.to_thread(_query)


def _title_hit_to_result(row: dict[str, Any], query: str) -> dict[str, Any]:
    """Convert a rag_chunks row from _title_match_search to a search result dict."""
    meta = row.get("metadata") or {}
    if isinstance(meta, str):
        import json as _json
        try:
            meta = _json.loads(meta)
        except Exception:
            meta = {}

    chunk_type = str(row.get("chunk_type") or meta.get("chunk_type") or "")
    text_val = str(row.get("text") or "")
    source_title = meta.get("source_title")
    segment_title = meta.get("segment_title")
    book_title = meta.get("book_title")
    if (
        not book_title
        and isinstance(source_title, str) and source_title.strip()
        and isinstance(segment_title, str) and segment_title.strip()
        and source_title.strip() != segment_title.strip()
    ):
        book_title = source_title.strip()

    author = meta.get("author")
    if isinstance(author, str) and author.strip().lower() in (
        "philo-von-freisinn", "philo von freisinn",
    ):
        author = None

    # Exact title match → highest score; partial → still above typical RRF (~0.03).
    seg_lower = (segment_title or "").strip().lower()
    q_lower = query.strip().lower()
    if seg_lower == q_lower:
        score = 1.0
    elif q_lower in seg_lower:
        score = 0.8
    else:
        score = 0.5

    item: dict[str, Any] = {
        "chunk_id": str(row.get("chunk_id") or ""),
        "source_id": str(row.get("source_id") or meta.get("source_id") or ""),
        "segment_id": meta.get("segment_id"),
        "paragraph_id": meta.get("paragraph_id"),
        "title": book_title or source_title,
        "segment_title": segment_title,
        "snippet": _snippet_text(text_val),
        "text": text_val,
        "score": score,
        "chunk_type": chunk_type,
        "source_type": meta.get("source_type"),
        "author": author,
        "book_title": book_title,
        "venue": meta.get("venue"),
        "lecture_date": meta.get("lecture_date"),
        "vortragstitel": meta.get("vortragstitel"),
    }
    if chunk_type == "chapter_summary":
        parent_id = _parent_id_from_meta(meta)
        if parent_id:
            item["parent_id"] = parent_id
        source_index = meta.get("source_index")
        if source_index is None:
            source_index = meta.get("segment_index")
        if isinstance(source_index, (int, float)) and not isinstance(source_index, bool):
            item["source_index"] = int(source_index)
    if chunk_type in ("quote", "quote_explanation"):
        item["quote_text"] = text_val
        item["quote_author"] = meta.get("author")
        _apply_quote_nav_fields_from_meta(item, meta)
    if chunk_type in ("begriff", "typology"):
        refs = meta.get("references")
        if isinstance(refs, list) and refs:
            item["references"] = refs
    item.update(typology_fields_for_search_result(meta))

    return item


async def app_search(
    *,
    query: str,
    types: list[str] | None = None,
    limit: int = 20,
    collection: str | None = None,
    engine: Engine | None = None,
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

    # Run hybrid vector search and DB title-match search in parallel.
    # Skip title search for very long queries (unlikely to match a title).
    _MAX_TITLE_QUERY_LEN = 120

    hybrid_coro = hybrid_retrieve(
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

    title_types = chunk_types_for_title_search(chunk_types)
    run_title_search = (
        engine is not None and len(q) <= _MAX_TITLE_QUERY_LEN and bool(title_types)
    )
    if run_title_search:
        title_coro = _title_match_search(q, title_types, engine, limit=k)
        hybrid_out, title_out = await asyncio.gather(
            hybrid_coro, title_coro, return_exceptions=True
        )
        if isinstance(hybrid_out, BaseException):
            logger.exception(
                "hybrid_retrieve failed (%s): %s",
                type(hybrid_out).__name__,
                hybrid_out,
            )
            if isinstance(title_out, BaseException):
                raise hybrid_out
            snippets, title_rows = [], title_out
        elif isinstance(title_out, BaseException):
            logger.exception(
                "title search failed (%s): %s",
                type(title_out).__name__,
                title_out,
            )
            snippets, title_rows = hybrid_out, []
        else:
            snippets, title_rows = hybrid_out, title_out
    else:
        snippets = await hybrid_coro
        title_rows = []

    # Build title-match results and collect their chunk_ids for dedup.
    title_results: list[dict[str, Any]] = []
    title_chunk_ids: set[str] = set()
    for row in title_rows:
        item = _title_hit_to_result(row, q)
        cid = item.get("chunk_id")
        if cid:
            title_chunk_ids.add(cid)
            title_results.append(item)

    results: list[dict[str, Any]] = []
    for snip in snippets:
        meta = _meta(snip)
        chunk_id = meta.get("chunk_id") or snip.payload.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id:
            continue
        # Skip hybrid hits already covered by title search (will be merged later).
        if chunk_id in title_chunk_ids:
            continue
        source_id = meta.get("source_id")
        text_val = snip.text or ""
        chunk_type = meta.get("chunk_type")
        source_title = meta.get("source_title")
        segment_title = meta.get("segment_title")
        book_title = meta.get("book_title")
        if (
            not book_title
            and isinstance(source_title, str)
            and source_title.strip()
            and isinstance(segment_title, str)
            and segment_title.strip()
            and source_title.strip() != segment_title.strip()
        ):
            book_title = source_title.strip()
        author = meta.get("author")
        if isinstance(author, str) and author.strip().lower() in (
            "philo-von-freisinn",
            "philo von freisinn",
        ):
            author = None
        item: dict[str, Any] = {
            "chunk_id": chunk_id,
            "source_id": str(source_id or ""),
            "segment_id": meta.get("segment_id"),
            "paragraph_id": meta.get("paragraph_id"),
            "title": book_title or source_title,
            "segment_title": segment_title,
            "snippet": _snippet_text(text_val),
            "text": text_val,
            "score": float(snip.score),
            "chunk_type": chunk_type,
            "source_type": meta.get("source_type"),
            "author": author,
            "book_title": book_title,
            "venue": meta.get("venue"),
            "lecture_date": meta.get("lecture_date"),
            "vortragstitel": meta.get("vortragstitel"),
        }
        if chunk_type in ("quote", "quote_explanation"):
            item["quote_text"] = text_val
            item["quote_author"] = meta.get("author")
            _apply_quote_nav_fields_from_meta(item, meta)
        if chunk_type == "quote_explanation":
            parent_id = _parent_id_from_meta(meta)
            if parent_id:
                item["_parent_id"] = parent_id
        if chunk_type == "chapter_summary":
            parent_id = _parent_id_from_meta(meta)
            if parent_id:
                item["parent_id"] = parent_id
            source_index = meta.get("source_index")
            if source_index is None:
                source_index = meta.get("segment_index")
            if isinstance(source_index, (int, float)) and not isinstance(source_index, bool):
                item["source_index"] = int(source_index)
        if chunk_type in ("begriff", "typology"):
            refs = meta.get("references")
            if isinstance(refs, list) and refs:
                item["references"] = refs
        if chunk_type == "ga_text":
            for key in (
                "ga",
                "lecture_id",
                "vortragstitel",
                "page_start",
                "page_end",
                "segment_id",
            ):
                if meta.get(key) is not None and meta.get(key) != "":
                    item[key] = meta[key]
        item.update(typology_fields_for_search_result(meta))
        results.append(item)

    async def _enrich_navigation(rows: list[dict[str, Any]]) -> None:
        if engine is None or not rows:
            return
        await _resolve_quote_explanations(rows, engine)
        await _enrich_catalog_quotes(rows, engine)
        chunk_ids = _chunk_ids_for_paragraph_lookup(rows)
        pid_map = await _lookup_paragraph_ids(chunk_ids, engine)
        for r in rows:
            if not r.get("paragraph_id"):
                r["paragraph_id"] = pid_map.get(str(r.get("chunk_id") or ""))

        nav_paragraph_ids = [
            str(r["paragraph_id"])
            for r in rows
            if r.get("paragraph_id")
            and str(r.get("chunk_type") or "") in _NAVIGATION_CHUNK_TYPES
        ]
        if nav_paragraph_ids:
            para_source_map = await _lookup_source_ids_for_paragraphs(
                nav_paragraph_ids, engine
            )
            for r in rows:
                pid = r.get("paragraph_id")
                if pid and str(r.get("chunk_type") or "") in _NAVIGATION_CHUNK_TYPES:
                    resolved_sid = para_source_map.get(str(pid))
                    if resolved_sid:
                        r["source_id"] = resolved_sid

        for r in rows:
            r.pop("_parent_id", None)
            chunk_type = str(r.get("chunk_type") or "")
            if chunk_type not in _NAVIGATION_CHUNK_TYPES:
                continue
            if not r.get("paragraph_id"):
                _set_navigation_error(
                    r,
                    "Kein Absatz für diesen Treffer verfügbar — bitte Ingest neu ausführen.",
                )

    # Title-match hits were previously skipped — lecture titles then had
    # no paragraph_id / citation (F-5).
    await _enrich_navigation(results)
    await _enrich_navigation(title_results)

    # Merge title-match results: prepend (sorted by title relevance), then hybrid.
    # Title-match scores (0.5–1.0) are not comparable to RRF scores (~0.01–0.03),
    # so we prepend structurally rather than mixing by score.
    if title_results:
        title_results.sort(key=lambda r: float(r.get("score", 0)), reverse=True)
        merged = (title_results + results)[:k]
    else:
        merged = results

    merged = _cap_summary_results(merged)
    return apply_quote_hit_policy(merged)


# ---------------------------------------------------------------------------
# Catalog quotes mode (search_corpus quotes=author|quoted)
# ---------------------------------------------------------------------------

_QUOTES_KINDS = frozenset({"author", "quoted"})


def _normalize_author_key(name: str) -> str:
    return " ".join((name or "").split()).strip().lower()


def _authors_match(a: str, b: str) -> bool:
    ka = _normalize_author_key(a)
    kb = _normalize_author_key(b)
    return bool(ka) and bool(kb) and ka == kb


def _mark_matches_quotes_kind(
    mark_author: str,
    book_author: str | None,
    quotes_kind: str,
) -> bool:
    book = (book_author or "").strip()
    if quotes_kind == "author":
        return bool(book) and _authors_match(mark_author, book)
    # quoted: mark author differs from book author (or book author missing)
    if not book:
        return True
    return not _authors_match(mark_author, book)


def _extract_span_text(text_raw: str, span: dict[str, Any]) -> str:
    try:
        start = int(span["start"])
        end = int(span["end"])
    except (KeyError, TypeError, ValueError):
        return ""
    if start < 0 or end <= start or end > len(text_raw):
        return ""
    return text_raw[start:end].strip()


async def _load_quote_marks_and_book_authors(
    chunk_ids: list[str],
    engine: Engine,
) -> dict[str, dict[str, Any]]:
    """chunk_id → {book_author, quote_marks}."""

    def _query() -> dict[str, dict[str, Any]]:
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT chunk_id::text AS chunk_id,
                           COALESCE(metadata->>'author', '') AS book_author,
                           CASE
                             WHEN jsonb_typeof(metadata->'quote_marks') = 'array'
                               THEN metadata->'quote_marks'
                             WHEN jsonb_typeof(metadata->'quotes') = 'array'
                               THEN metadata->'quotes'
                             ELSE '[]'::jsonb
                           END AS quote_marks
                    FROM rag_chunks
                    WHERE chunk_id = ANY(:ids)
                      AND deprecated_at IS NULL
                      AND chunk_type = ANY(:types)
                    """
                ),
                {"ids": chunk_ids, "types": list(_BODY_CHUNK_TYPES)},
            ).mappings().all()
        out: dict[str, dict[str, Any]] = {}
        for r in rows:
            cid = str(r["chunk_id"] or "")
            if not cid:
                continue
            out[cid] = {
                "book_author": str(r.get("book_author") or "").strip(),
                "quote_marks": _normalize_catalog_quotes(r.get("quote_marks")) or [],
            }
        return out

    return await asyncio.to_thread(_query)


async def _load_paragraph_texts(
    paragraph_ids: list[str],
    engine: Engine,
) -> dict[str, str]:
    ids = list(dict.fromkeys(p for p in paragraph_ids if p))
    if not ids:
        return {}

    def _query() -> dict[str, str]:
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT id::text AS id, text_raw
                    FROM rag_paragraphs
                    WHERE id = ANY(:ids)
                      AND deprecated_at IS NULL
                    """
                ),
                {"ids": ids},
            ).mappings().all()
        return {
            str(r["id"]): str(r.get("text_raw") or "")
            for r in rows
            if r.get("id")
        }

    return await asyncio.to_thread(_query)


async def list_catalog_quote_authors(
    *,
    quotes_kind: str,
    engine: Engine,
) -> list[str]:
    """Distinct mark authors for active body chunks with metadata.quotes = kind."""
    kind = (quotes_kind or "").strip().lower()
    if kind not in _QUOTES_KINDS:
        return []

    def _query() -> list[str]:
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT DISTINCT trim(mark->>'author') AS author
                    FROM rag_chunks c,
                         jsonb_array_elements(
                           CASE
                             WHEN jsonb_typeof(c.metadata->'quote_marks') = 'array'
                               THEN c.metadata->'quote_marks'
                             ELSE '[]'::jsonb
                           END
                         ) AS mark
                    WHERE c.deprecated_at IS NULL
                      AND c.chunk_type = ANY(:types)
                      AND c.metadata->>'quotes' = :kind
                      AND coalesce(trim(mark->>'author'), '') <> ''
                    ORDER BY 1
                    """
                ),
                {"types": list(_BODY_CHUNK_TYPES), "kind": kind},
            ).mappings().all()
        authors = [str(r["author"]).strip() for r in rows if r.get("author")]
        if kind == "author":
            return authors
        # quoted: keep authors that differ from typical band authors is already
        # encoded by metadata.quotes='quoted' on the chunk.
        return authors

    return await asyncio.to_thread(_query)


async def app_search_catalog_quotes(
    *,
    query: str,
    quotes_kind: str,
    author: str | None = None,
    limit: int = 10,
    collection: str | None = None,
    engine: Engine | None = None,
) -> dict[str, Any]:
    """Semantic search over body chunks that carry catalog quote marks.

    Returns ``{hits: [{author, text, return_url}], authors: list|null}``.
    When hits is empty, ``authors`` lists available mark authors for ``quotes_kind``.
    """
    from app.mcp_server.citation import passage_return_url

    kind = (quotes_kind or "").strip().lower()
    if kind not in _QUOTES_KINDS:
        return {
            "error": "quotes must be 'author' or 'quoted'",
            "code": "invalid_quotes",
        }

    q = (query or "").strip()
    if not q:
        return {"error": "query is required", "code": "empty_query"}

    if engine is None:
        return {"error": "database engine required", "code": "no_engine"}

    author_filter = (author or "").strip() or None
    k = max(1, min(int(limit), 20))
    fetch_k = min(50, max(k * 3, k + 10))

    assistant_slug = settings.app_default_assistant_slug
    coll = (collection or "").strip() or load_assistant_rag_collection(assistant_slug)
    _prefix_passage, prefix_query = load_assistant_embedding_prefixes(assistant_slug)

    extra_must: list[dict[str, object]] = [
        {"key": "quotes", "match": {"value": kind}},
    ]

    snippets = await hybrid_retrieve(
        query=q,
        k_dense=fetch_k,
        k_sparse=fetch_k,
        k_fused=fetch_k,
        worldview=None,
        book_types=["book", "secondary_book"],
        collection=coll,
        embedding_client=get_embedding_client(),
        qdrant_client=get_qdrant_client(),
        query_prefix=prefix_query or None,
        force_sparse=settings.use_hybrid_retrieval,
        extra_must=extra_must,
    )

    chunk_scores: list[tuple[str, float]] = []
    seen_cids: set[str] = set()
    for snip in snippets:
        meta = _meta(snip)
        cid = str(meta.get("chunk_id") or snip.payload.get("chunk_id") or "").strip()
        if not cid or cid in seen_cids:
            continue
        seen_cids.add(cid)
        chunk_scores.append((cid, float(snip.score)))

    hits: list[dict[str, Any]] = []
    if chunk_scores:
        by_chunk = await _load_quote_marks_and_book_authors(
            [c for c, _ in chunk_scores],
            engine,
        )
        para_ids: list[str] = []
        pending: list[tuple[float, dict[str, Any], str]] = []
        # (score, mark, book_author) candidates before text load
        for cid, score in chunk_scores:
            info = by_chunk.get(cid) or {}
            book_author = str(info.get("book_author") or "")
            marks = info.get("quote_marks") or []
            if not isinstance(marks, list):
                continue
            for mark in marks:
                if not isinstance(mark, dict):
                    continue
                mark_author = str(mark.get("author") or "").strip()
                if not mark_author:
                    continue
                if not _mark_matches_quotes_kind(mark_author, book_author, kind):
                    continue
                if author_filter and not _authors_match(mark_author, author_filter):
                    continue
                pid = str(mark.get("paragraph_id") or "").strip()
                span = mark.get("span")
                if not pid or not isinstance(span, dict):
                    continue
                para_ids.append(pid)
                pending.append((score, mark, mark_author))

        texts = await _load_paragraph_texts(para_ids, engine)
        for score, mark, mark_author in pending:
            pid = str(mark.get("paragraph_id") or "").strip()
            span = mark.get("span")
            if not isinstance(span, dict):
                continue
            raw = texts.get(pid) or ""
            quote_text = _extract_span_text(raw, span)
            if not quote_text:
                continue
            return_url = passage_return_url(pid)
            if not return_url:
                continue
            hits.append(
                {
                    "author": mark_author,
                    "text": quote_text,
                    "return_url": return_url,
                    "_score": score,
                }
            )
            if len(hits) >= k:
                break

    if hits:
        slim = [
            {
                "author": h["author"],
                "text": h["text"],
                "return_url": h["return_url"],
            }
            for h in hits[:k]
        ]
        return {"hits": slim, "authors": None}

    authors = await list_catalog_quote_authors(quotes_kind=kind, engine=engine)
    return {"hits": [], "authors": authors}
