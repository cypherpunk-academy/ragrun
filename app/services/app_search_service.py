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
from app.retrieval.utils.retrievers import hybrid_retrieve
from app.retrieval.models import RetrievedSnippet

_APP_TYPE_TO_CHUNK_TYPES: dict[str, list[str]] = {
    "text": ["book", "secondary_book", "talk"],
    "concept": ["begriff", "typology"],
    "quote": ["quote", "quote_explanation"],
    "chapter_summary": ["chapter_summary"],
}

_NAVIGATION_CHUNK_TYPES = frozenset({
    "book",
    "secondary_book",
    "talk",
    "quote",
    "quote_explanation",
})


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
                    JOIN rag_paragraphs rp ON rp.id = apc.paragraph_id
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
        results.append(item)

    if engine is not None and results:
        await _resolve_quote_explanations(results, engine)

    if engine is not None and results:
        chunk_ids = _chunk_ids_for_paragraph_lookup(results)
        pid_map = await _lookup_paragraph_ids(chunk_ids, engine)
        for r in results:
            if not r.get("paragraph_id"):
                r["paragraph_id"] = pid_map.get(str(r.get("chunk_id") or ""))

    # Resolve source_id from rag_paragraphs for all navigation results.
    # The chunk's source_id may differ from the paragraph's source_id (e.g. quote chunks
    # stored under a collection-level source_id). Using the paragraph's source_id ensures
    # the app can load the correct paragraphs from its local database.
    if engine is not None and results:
        nav_paragraph_ids = [
            str(r["paragraph_id"])
            for r in results
            if r.get("paragraph_id")
            and str(r.get("chunk_type") or "") in _NAVIGATION_CHUNK_TYPES
        ]
        if nav_paragraph_ids:
            para_source_map = await _lookup_source_ids_for_paragraphs(nav_paragraph_ids, engine)
            for r in results:
                pid = r.get("paragraph_id")
                if pid and str(r.get("chunk_type") or "") in _NAVIGATION_CHUNK_TYPES:
                    resolved_sid = para_source_map.get(str(pid))
                    if resolved_sid:
                        r["source_id"] = resolved_sid

    for r in results:
        r.pop("_parent_id", None)
        chunk_type = str(r.get("chunk_type") or "")
        if chunk_type not in _NAVIGATION_CHUNK_TYPES:
            continue
        if not r.get("paragraph_id"):
            _set_navigation_error(
                r,
                "Kein Absatz für diesen Treffer verfügbar — bitte Ingest neu ausführen.",
            )

    return results
