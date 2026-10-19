"""MCP list_typology / search_typology — reads rag_chunks, searches Qdrant typology only."""
from __future__ import annotations

import asyncio
import json
import re
import unicodedata
from typing import Any, Mapping

from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.config import settings
from app.core.providers import get_embedding_client, get_qdrant_client
from app.db.session import get_engine
from app.retrieval.models import RetrievedSnippet
from app.retrieval.services.action_prompt_service import (
    load_assistant_embedding_prefixes,
    load_assistant_rag_collection,
)
from app.retrieval.utils.retrievers import (
    filter_snippets_by_typology_source,
    hybrid_retrieve,
    typology_dense_retrieve,
)

_SNIPPET_LIMIT = 300
_VALID_HEALTH = frozenset({"gesund", "krank"})
_VALID_ROLES = frozenset({"overview", "group", "member"})


def _parse_meta(raw: object) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return {}


def _slugify(name: str) -> str:
    s = unicodedata.normalize("NFKD", name.strip().lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s


def _norm(s: str) -> str:
    return _slugify(s).replace("-", " ")


def _snippet(t: str, limit: int = _SNIPPET_LIMIT) -> str:
    s = (t or "").strip().replace("\n", " ")
    if len(s) <= limit:
        return s
    return s[: limit - 1].rstrip() + "…"


def _assistant_partition() -> str:
    slug = settings.app_default_assistant_slug
    return load_assistant_rag_collection(slug)


def _load_typology_rows(engine: Engine, partition: str) -> list[dict[str, Any]]:
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT chunk_id, source_id, text, metadata
                FROM rag_chunks
                WHERE rag_partition = :part
                  AND chunk_type = 'typology'
                  AND deprecated_at IS NULL
                ORDER BY source_id, chunk_id
                """
            ),
            {"part": partition},
        ).mappings().all()
    out: list[dict[str, Any]] = []
    for row in rows:
        meta = _parse_meta(row.get("metadata"))
        out.append(
            {
                "chunk_id": str(row["chunk_id"]),
                "source_id": str(row["source_id"]),
                "text": str(row.get("text") or ""),
                "meta": meta,
            }
        )
    return out


def _index_typologies(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_slug: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        slug = str(row["meta"].get("typology_slug") or "").strip()
        if not slug:
            continue
        by_slug.setdefault(slug, []).append(row)
    return by_slug


def _typology_has_health(chunks: list[dict[str, Any]]) -> bool:
    return any(str(c["meta"].get("health") or "") in _VALID_HEALTH for c in chunks)


def _resolve_typology_slug(
    *,
    name: str | None,
    chunk_id: str | None,
    by_slug: dict[str, list[dict[str, Any]]],
) -> str | None:
    cid = (chunk_id or "").strip()
    if cid:
        for slug, chunks in by_slug.items():
            if any(c["chunk_id"] == cid for c in chunks):
                return slug
        return None

    raw = (name or "").strip()
    if not raw:
        return None

    if raw in by_slug:
        return raw

    slug_guess = _slugify(raw)
    if slug_guess in by_slug:
        return slug_guess

    raw_norm = _norm(raw)
    for slug, chunks in by_slug.items():
        overview = next(
            (c for c in chunks if c["meta"].get("typology_role") == "overview"),
            None,
        )
        if overview is None:
            continue
        titles = [
            str(overview["meta"].get("segment_title") or ""),
            str(overview["meta"].get("source_title") or ""),
        ]
        aliases = overview["meta"].get("aliases")
        if isinstance(aliases, list):
            titles.extend(str(a) for a in aliases if isinstance(a, str))
        for t in titles:
            if not t.strip():
                continue
            if _norm(t) == raw_norm or raw_norm in _norm(t):
                return slug
        if _slugify(slug) == slug_guess:
            return slug
    return None


def _group_row(c: dict[str, Any]) -> dict[str, Any]:
    m = c["meta"]
    row: dict[str, Any] = {
        "group_id": m.get("group_id"),
        "group_title": m.get("group_title"),
        "segment_title": m.get("segment_title"),
        "health": m.get("health"),
        "chunk_id": c["chunk_id"],
        "counterpart_chunk_id": m.get("counterpart_chunk_id"),
    }
    return {k: v for k, v in row.items() if v is not None and v != ""}


def _member_row(c: dict[str, Any]) -> dict[str, Any]:
    m = c["meta"]
    row: dict[str, Any] = {
        "segment_title": m.get("segment_title"),
        "chunk_id": c["chunk_id"],
        "group_id": m.get("group_id"),
        "health": m.get("health"),
        "segment_index": m.get("segment_index"),
        "counterpart_chunk_id": m.get("counterpart_chunk_id"),
    }
    return {k: v for k, v in row.items() if v is not None and v != ""}


def _build_focus(
    *,
    focus_id: str,
    chunks: list[dict[str, Any]],
    by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    focus = by_id.get(focus_id)
    if focus is None:
        return {}

    m = focus["meta"]
    role = str(m.get("typology_role") or "")
    out: dict[str, Any] = {
        "chunk_id": focus_id,
        "typology_role": role,
        "segment_title": m.get("segment_title"),
    }
    cp = m.get("counterpart_chunk_id")
    if isinstance(cp, str) and cp.strip() and cp in by_id:
        counterpart = by_id[cp]
        cm = counterpart["meta"]
        out["counterpart"] = {
            "chunk_id": cp,
            "segment_title": cm.get("segment_title"),
            "health": cm.get("health"),
            "typology_role": cm.get("typology_role"),
        }

    gid = m.get("group_id")
    health = m.get("health")
    same_group: list[dict[str, Any]] = []
    if role == "member" and gid:
        for c in chunks:
            if c["chunk_id"] == focus_id:
                continue
            cm = c["meta"]
            if cm.get("typology_role") != "member":
                continue
            if cm.get("group_id") != gid:
                continue
            if health and cm.get("health") != health:
                continue
            same_group.append(_member_row(c))
    elif role == "member" and not gid:
        for c in chunks:
            if c["chunk_id"] == focus_id:
                continue
            if c["meta"].get("typology_role") != "member":
                continue
            same_group.append(_member_row(c))

    if same_group:
        out["same_group"] = same_group

    other_groups: list[dict[str, Any]] = []
    for c in chunks:
        if c["meta"].get("typology_role") != "group":
            continue
        if role == "group" and c["chunk_id"] == focus_id:
            continue
        if role == "member" and gid and c["meta"].get("group_id") == gid:
            if health and c["meta"].get("health") == health:
                continue
        other_groups.append(_group_row(c))
    if other_groups:
        out["other_groups"] = other_groups

    return out


def _typology_catalog(by_slug: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    catalog: list[dict[str, Any]] = []
    for slug, chunks in by_slug.items():
        overview = next(
            (c for c in chunks if c["meta"].get("typology_role") == "overview"),
            None,
        )
        if overview is None:
            continue
        om = overview["meta"]
        short = om.get("short_text")
        entry: dict[str, Any] = {
            "typology_slug": slug,
            "title": om.get("segment_title") or om.get("source_title") or slug,
            "overview_chunk_id": overview["chunk_id"],
            "has_health": _typology_has_health(chunks),
            "group_count": sum(
                1 for c in chunks if c["meta"].get("typology_role") == "group"
            ),
            "member_count": sum(
                1 for c in chunks if c["meta"].get("typology_role") == "member"
            ),
        }
        if short:
            entry["short_text"] = _snippet(str(short), limit=220)
        else:
            entry["short_text"] = _snippet(overview["text"], limit=220)
        catalog.append(entry)
    catalog.sort(key=lambda e: str(e.get("title") or e["typology_slug"]).casefold())
    return catalog


async def list_typologies(*, engine: Engine | None = None) -> dict[str, Any]:
    """All typology overviews from rag_chunks (no vector search)."""
    eng = engine or get_engine()
    partition = _assistant_partition()
    rows = await asyncio.to_thread(_load_typology_rows, eng, partition)
    by_slug = _index_typologies(rows)
    catalog = _typology_catalog(by_slug)
    return {"count": len(catalog), "typologies": catalog}


async def list_typology(
    *,
    name: str | None = None,
    chunk_id: str | None = None,
    engine: Engine | None = None,
) -> dict[str, Any]:
    if not (name or "").strip() and not (chunk_id or "").strip():
        return {"error": "name or chunk_id is required", "code": "missing_argument"}

    eng = engine or get_engine()
    partition = _assistant_partition()
    rows = await asyncio.to_thread(_load_typology_rows, eng, partition)
    by_slug = _index_typologies(rows)
    if not by_slug:
        return {"error": "no typology chunks in corpus", "code": "empty_corpus"}

    slug = _resolve_typology_slug(name=name, chunk_id=chunk_id, by_slug=by_slug)
    if not slug:
        return {"error": f"unknown typology: {name or chunk_id}", "code": "unknown_typology"}

    chunks = by_slug[slug]
    by_id = {c["chunk_id"]: c for c in chunks}
    overview = next((c for c in chunks if c["meta"].get("typology_role") == "overview"), None)
    if overview is None:
        return {"error": "typology has no overview chunk", "code": "invalid_typology"}

    om = overview["meta"]
    focus_id = (chunk_id or "").strip() or None
    if focus_id and focus_id not in by_id:
        return {"error": f"chunk_id not in typology {slug}", "code": "unknown_chunk"}

    resp: dict[str, Any] = {
        "typology_slug": slug,
        "title": om.get("segment_title") or om.get("source_title"),
        "has_health": _typology_has_health(chunks),
        "overview": {
            "chunk_id": overview["chunk_id"],
            "segment_title": om.get("segment_title"),
        },
        "groups": [_group_row(c) for c in chunks if c["meta"].get("typology_role") == "group"],
        "members": [_member_row(c) for c in chunks if c["meta"].get("typology_role") == "member"],
    }
    if focus_id:
        resp["focus"] = _build_focus(focus_id=focus_id, chunks=chunks, by_id=by_id)
    return resp


def _match_group_filter(meta: Mapping[str, Any], group: str) -> bool:
    g = group.strip().lower()
    if not g:
        return True
    gid = str(meta.get("group_id") or "").lower()
    gtitle = str(meta.get("group_title") or "").lower()
    if g == gid or g in gid:
        return True
    if g in gtitle or gtitle in g:
        return True
    if g.replace(" ", "") in gtitle.replace(" ", ""):
        return True
    return False


def _meta_from_snippet(snip: RetrievedSnippet) -> dict[str, Any]:
    payload = snip.payload if isinstance(snip.payload, Mapping) else {}
    inner = payload.get("payload") if isinstance(payload.get("payload"), dict) else payload
    return dict(inner) if isinstance(inner, dict) else {}


def _hit_from_snippet(snip: RetrievedSnippet) -> dict[str, Any]:
    meta = _meta_from_snippet(snip)
    short = meta.get("short_text")
    text_val = snip.text or ""
    item: dict[str, Any] = {
        "chunk_id": meta.get("chunk_id"),
        "score": round(float(snip.score), 4),
        "snippet": _snippet(str(short if short else text_val)),
        "segment_title": meta.get("segment_title"),
        "typology_slug": meta.get("typology_slug"),
        "typology_role": meta.get("typology_role"),
        "group_id": meta.get("group_id"),
        "group_title": meta.get("group_title"),
        "health": meta.get("health"),
        "counterpart_chunk_id": meta.get("counterpart_chunk_id"),
        "segment_index": meta.get("segment_index"),
    }
    return {k: v for k, v in item.items() if v is not None and v != ""}


def _post_filter_typology_hits(
    snippets: list[RetrievedSnippet],
    *,
    typology_slug: str | None,
    group: str | None,
    health: str | None,
    role: str | None,
) -> list[RetrievedSnippet]:
    out: list[RetrievedSnippet] = []
    for snip in snippets:
        meta = _meta_from_snippet(snip)
        if typology_slug and str(meta.get("typology_slug") or "") != typology_slug:
            continue
        if role and str(meta.get("typology_role") or "") != role:
            continue
        if health and str(meta.get("health") or "") != health:
            continue
        if group and not _match_group_filter(meta, group):
            continue
        out.append(snip)
    return out


async def search_typology(
    *,
    query: str,
    typology: str | None = None,
    group: str | None = None,
    health: str | None = None,
    role: str | None = None,
    limit: int = 8,
    engine: Engine | None = None,
) -> list[dict[str, Any]] | dict[str, Any]:
    q = (query or "").strip()
    if not q:
        return {"error": "query is required", "code": "empty_query"}

    if health and health not in _VALID_HEALTH:
        return {"error": f"invalid health: {health}", "code": "invalid_health"}
    if role and role not in _VALID_ROLES:
        return {"error": f"invalid role: {role}", "code": "invalid_role"}

    k = max(1, min(int(limit) if limit is not None else 8, 20))
    eng = engine or get_engine()
    partition = _assistant_partition()
    rows = await asyncio.to_thread(_load_typology_rows, eng, partition)
    by_slug = _index_typologies(rows)
    if not rows:
        return []

    typology_slug: str | None = None
    source_ids: list[str] = []
    if (typology or "").strip():
        typology_slug = _resolve_typology_slug(name=typology, chunk_id=None, by_slug=by_slug)
        if not typology_slug:
            return {"error": f"unknown typology: {typology}", "code": "unknown_typology"}
        chunks = by_slug[typology_slug]
        if health and not _typology_has_health(chunks):
            return {"error": "health filter not applicable for this typology", "code": "health_not_applicable"}
        source_ids = list({c["source_id"] for c in chunks})
    else:
        source_ids = list({c["source_id"] for c in rows})

    assistant_slug = settings.app_default_assistant_slug
    coll = load_assistant_rag_collection(assistant_slug)
    _, prefix_query = load_assistant_embedding_prefixes(assistant_slug)

    fetch_k = max(k * 4, 24)
    if settings.use_hybrid_retrieval:
        raw = await hybrid_retrieve(
            query=q,
            k_dense=fetch_k,
            k_sparse=fetch_k,
            k_fused=fetch_k,
            worldview=None,
            book_types=["typology"],
            collection=coll,
            embedding_client=get_embedding_client(),
            qdrant_client=get_qdrant_client(),
            query_prefix=prefix_query or None,
            force_sparse=True,
        )
        filtered = filter_snippets_by_typology_source(
            raw,
            chunk_types=["typology"],
            source_ids=source_ids,
        )
    else:
        filtered = await typology_dense_retrieve(
            query=q,
            k=fetch_k,
            worldview=None,
            book_types=["typology"],
            collection=coll,
            embedding_client=get_embedding_client(),
            qdrant_client=get_qdrant_client(),
            source_ids=source_ids,
            chunk_types=["typology"],
            query_prefix=prefix_query or None,
        )

    filtered = _post_filter_typology_hits(
        filtered,
        typology_slug=typology_slug,
        group=group,
        health=health,
        role=role,
    )
    return [_hit_from_snippet(s) for s in filtered[:k]]


def typology_fields_for_search_result(meta: Mapping[str, Any]) -> dict[str, Any]:
    """Extra MCP search_corpus fields when chunk_type is typology."""
    if str(meta.get("chunk_type") or "") != "typology":
        return {}
    out: dict[str, Any] = {}
    for key in (
        "typology_role",
        "typology_slug",
        "group_id",
        "group_title",
        "health",
        "counterpart_chunk_id",
        "short_text",
    ):
        val = meta.get(key)
        if val is not None and val != "":
            out[key] = val
    return out
