"""MCP tools for Filo x Claude integration (Steps 11 + 13)."""
from __future__ import annotations

import asyncio
import logging
import re
import uuid
from typing import Any, Mapping

import httpx
from sqlalchemy import text

from app.config import settings
from app.db.session import get_engine
from app.mcp_server.citation import (
    fetch_chapter_citation,
    fetch_citation,
    fetch_citations_batch,
    format_zitierform,
    note_display_title,
    resolve_segment_slug,
    text_return_url,
)
from app.services.app_catalog_repository import PostgresCatalogRepository
from app.services.app_search_service import (
    _lookup_paragraph_ids,
    app_search,
    app_search_catalog_quotes,
)
from app.services.corpus_search_policy import is_augment_hit, is_original_hit

logger = logging.getLogger(__name__)

_SNIPPET_LIMIT = 300
_CONTENT_MAX_BYTES = 1_048_576  # 1 MB
_ALLOWED_SEARCH_TYPES = frozenset(
    {"text", "quote", "curated_text", "concept", "chapter_summary"}
)
_REF_ANCHOR_CHUNK_TYPES = frozenset({"begriff", "typology", "curated_text"})
_MAX_WEGWEISER = 5
_MAX_MEMBERS_PER_GROUP = 2
_WEGWEISER_HINWEISE_REF = (
    "search_corpus(ref=…) mit wegweiser[].ref oder gegenpol.ref (UUID). "
    "Typologie-Slug (z. B. soziale-gesundheit) → Übersichtsknoten. "
    "Feld feld-N: kontext.group_chunk_id der Gruppe (nicht group_id)."
)
_WEGWEISER_HINWEISE_BELEGE = (
    "references_status fehlt = keine Referenzen in rag_chunks — Typologie mit "
    "ragprep neu chunken/embedden, oder Steiner-Belege per Textsuche."
)
_CHUNK_ID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def _mcp_text(t: str | None) -> str:
    """Normalize text for MCP responses (strip soft hyphens)."""
    from app.utils.text_utils import strip_soft_hyphens

    return strip_soft_hyphens(t or "")


def _snippet(t: str, limit: int = _SNIPPET_LIMIT) -> str:
    s = _mcp_text(t).strip().replace("\n", " ")
    if len(s) <= limit:
        return s
    return s[: limit - 1].rstrip() + "…"


def _bare_source_id(source_id: str | None) -> str:
    sid = (source_id or "").strip()
    if not sid:
        return ""
    for suffix in (":summary", ":quotes"):
        if sid.endswith(suffix):
            return sid[: -len(suffix)]
    return sid


def _normalize_chunk_references(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        cid = str(item.get("chunk_id") or "").strip()
        if not cid:
            continue
        entry: dict[str, Any] = {"chunk_id": cid}
        chunk_type = str(item.get("chunk_type") or "").strip()
        if chunk_type:
            entry["chunk_type"] = chunk_type
        if item.get("zitierbar") is False:
            entry["zitierbar"] = False
        desc = item.get("description")
        if not (isinstance(desc, str) and desc.strip()):
            for key in ("begruendung", "relation"):
                alt = item.get(key)
                if isinstance(alt, str) and alt.strip():
                    desc = alt.strip()
                    break
        if isinstance(desc, str) and desc.strip():
            entry["description"] = desc.strip()
        snippet = item.get("snippet") or item.get("text")
        if isinstance(snippet, str) and snippet.strip():
            entry["snippet"] = snippet.strip()
        for key in (
            "source_id",
            "segment_slug",
            "title",
            "segment_title",
            "vortragstitel",
            "lecture_id",
            "ga",
        ):
            val = item.get(key)
            if isinstance(val, str) and val.strip():
                entry[key] = val.strip()
        for key in ("page_start", "page_end", "paragraph_number"):
            val = item.get(key)
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                entry[key] = int(val)
        rel = item.get("relevance")
        if isinstance(rel, (int, float)) and not isinstance(rel, bool):
            entry["relevance"] = float(rel)
        out.append(entry)
    return out


def _is_ga_angabe_ref(ref: Mapping[str, Any]) -> bool:
    """Stored GA fallback refs: not citable; resolved from denormalized fields."""
    if str(ref.get("chunk_type") or "").strip() == "ga_text":
        return True
    return ref.get("zitierbar") is False


def _fetch_chunk_references(
    engine: Any,
    chunk_ids: list[str],
) -> dict[str, list[dict[str, Any]]]:
    ids = [c.strip() for c in chunk_ids if c and str(c).strip()]
    if not ids:
        return {}

    def _query() -> dict[str, list[dict[str, Any]]]:
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT DISTINCT ON (chunk_id)
                        chunk_id,
                        "references" AS refs,
                        metadata
                    FROM rag_chunks
                    WHERE chunk_id = ANY(:ids)
                      AND deprecated_at IS NULL
                    ORDER BY chunk_id
                    """
                ),
                {"ids": ids},
            ).mappings().all()
        out: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            cid = str(row["chunk_id"])
            refs = row.get("refs")
            if not refs:
                meta = row.get("metadata")
                if isinstance(meta, Mapping):
                    refs = meta.get("references")
            parsed = _normalize_chunk_references(refs)
            if parsed:
                out[cid] = parsed
        return out

    return _query()


async def _enrich_corpus_hits(
    items: list[dict[str, Any]],
    raw_hits: list[dict[str, Any]],
    engine: Any,
) -> None:
    by_chunk = {str(h.get("chunk_id") or ""): h for h in raw_hits if h.get("chunk_id")}
    for item in items:
        raw = by_chunk.get(str(item.get("chunk_id") or ""), {})
        if raw.get("parent_id"):
            item["parent_id"] = raw["parent_id"]
        if raw.get("source_index") is not None:
            item["source_index"] = raw["source_index"]
        if raw.get("references"):
            item["_raw_references"] = raw["references"]

    pids = [str(i["paragraph_id"]) for i in items if i.get("paragraph_id")]
    citations = await asyncio.to_thread(fetch_citations_batch, engine, pids)
    for item in items:
        pid = item.get("paragraph_id")
        if pid and str(pid) in citations:
            item["citation"] = citations[str(pid)]

    summary_jobs: list[tuple[dict[str, Any], str, str]] = []
    for item in items:
        if str(item.get("chunk_type") or "") != "chapter_summary":
            continue
        if item.get("citation") and item["citation"].get("return_url"):
            continue
        parent = _bare_source_id(str(item.get("parent_id") or item.get("source_id") or ""))
        idx = item.get("source_index")
        slug = str(idx) if idx is not None else ""
        if parent and slug:
            summary_jobs.append((item, parent, slug))
    for item, parent, slug in summary_jobs:
        chapter = await asyncio.to_thread(fetch_chapter_citation, engine, parent, slug)
        if chapter:
            item["citation"] = chapter

    for item in items:
        item.pop("_raw_references", None)
        item.pop("parent_id", None)
        item.pop("source_index", None)
        if str(item.get("chunk_type") or "") == "curated_text" and item.get("citation"):
            item.pop("citation", None)


def _get_user_id() -> str | None:
    """Extract user_id from the MCP access token (set by auth middleware)."""
    from mcp.server.auth.middleware.auth_context import get_access_token

    token = get_access_token()
    if token is None:
        return None
    return token.subject


def _get_raw_token() -> str | None:
    """Extract the raw Supabase JWT from the MCP access token."""
    from mcp.server.auth.middleware.auth_context import get_access_token

    token = get_access_token()
    if token is None:
        return None
    return token.token


def _content_too_long(content: str | None) -> dict[str, Any] | None:
    raw = content if content is not None else ""
    if len(raw.encode("utf-8")) > _CONTENT_MAX_BYTES:
        return {
            "error": f"content exceeds maximum length of {_CONTENT_MAX_BYTES} bytes (1 MB)",
            "code": "content_too_long",
        }
    return None


def _validate_search_types(types: list[str] | None) -> dict[str, Any] | None:
    if not types:
        return None
    unknown = sorted(
        {
            (t or "").strip().lower()
            for t in types
            if (t or "").strip() and (t or "").strip().lower() not in _ALLOWED_SEARCH_TYPES
        }
    )
    if unknown:
        return {
            "error": (
                "invalid types: "
                + ", ".join(unknown)
                + f"; allowed: {', '.join(sorted(_ALLOWED_SEARCH_TYPES))}"
            ),
            "code": "invalid_types",
        }
    return None


async def _supabase_rpc(
    function_name: str,
    params: dict[str, Any],
    user_jwt: str,
) -> dict[str, Any]:
    """Call a Supabase PostgREST RPC function with the user's JWT."""
    base = (settings.supabase_url or "").rstrip("/")
    anon_key = settings.supabase_anon_key or ""
    if not base:
        return {"error": "supabase not configured"}

    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
        resp = await client.post(
            f"{base}/rest/v1/rpc/{function_name}",
            json=params,
            headers={
                "Authorization": f"Bearer {user_jwt}",
                "apikey": anon_key,
                "Content-Type": "application/json",
            },
        )

    if resp.status_code >= 400:
        logger.warning("RPC %s failed: %s %s", function_name, resp.status_code, resp.text)
        return {"error": f"RPC failed: {resp.status_code}"}
    return resp.json()


# ---------------------------------------------------------------------------
# search_corpus
# ---------------------------------------------------------------------------


def _references_status(raw_hits: list[dict[str, Any]], meta: Mapping[str, Any]) -> str:
    refs = meta.get("references")
    if isinstance(refs, list) and refs:
        return "belegt"
    for h in raw_hits:
        if isinstance(h.get("references"), list) and h["references"]:
            return "belegt"
    return "fehlt"


def _kuratiert_from_meta(meta: Mapping[str, Any]) -> bool:
    val = meta.get("kuratiert")
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in ("true", "1", "yes", "ja")
    return False


def _normalize_glieder(raw: Any) -> list[str] | None:
    if not isinstance(raw, list):
        return None
    out = [str(x).strip() for x in raw if str(x).strip()]
    return out or None


def _wegweiser_titel(raw: Mapping[str, Any], meta: Mapping[str, Any]) -> str:
    for key in ("segment_title", "title", "schlagwort", "source_title"):
        val = raw.get(key) or meta.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return str(raw.get("chunk_id") or "")


def _wegweiser_kontext(meta: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    group_id = meta.get("group_id")
    if isinstance(group_id, str) and group_id.strip():
        out["group_id"] = group_id.strip()
    slug = meta.get("typology_slug")
    if isinstance(slug, str) and slug.strip():
        out["typology_slug"] = slug.strip()
    parent = meta.get("parent_id")
    if isinstance(parent, str) and parent.strip():
        out["group_chunk_id"] = parent.strip()
    return out


def _build_wegweiser(
    raw: dict[str, Any],
    meta: Mapping[str, Any],
) -> dict[str, Any]:
    chunk_type = str(raw.get("chunk_type") or meta.get("chunk_type") or "")
    kind = "chapter_summary" if chunk_type == "chapter_summary" else chunk_type
    ref = str(raw.get("chunk_id") or "")
    kuratiert = _kuratiert_from_meta(meta)
    entry: dict[str, Any] = {
        "kind": kind,
        "titel": _wegweiser_titel(raw, meta),
        "ref": ref,
        "references_status": _references_status([raw], meta),
        "herkunft": "kuratiert" if kuratiert else "augment",
        "zitierbar": False,
        "score": round(float(raw.get("score", 0)), 4),
    }
    kontext = _wegweiser_kontext(meta)
    if kontext:
        entry["kontext"] = kontext
    role = str(raw.get("typology_role") or meta.get("typology_role") or "")
    if role:
        entry["typology_role"] = role
    if role == "group":
        entry.setdefault("kontext", {})["group_chunk_id"] = ref
    for key in ("group_title", "health"):
        val = raw.get(key) if raw.get(key) is not None else meta.get(key)
        if val is not None and val != "":
            entry[key] = val
    if kuratiert:
        for key in ("schlagsatz", "kurztext"):
            val = meta.get(key)
            if isinstance(val, str) and val.strip():
                entry[key] = _snippet(val.strip(), limit=300)
    glieder = _normalize_glieder(raw.get("glieder") if raw.get("glieder") is not None else meta.get("glieder"))
    if glieder:
        entry["glieder"] = glieder
    cp = meta.get("counterpart_chunk_id")
    if isinstance(cp, str) and cp.strip():
        entry["_counterpart_chunk_id"] = cp.strip()
    return entry


def _typology_role_of(raw: Mapping[str, Any]) -> str:
    return str(raw.get("typology_role") or "").strip().lower()


def _raw_references_belegt(raw: Mapping[str, Any]) -> bool:
    refs = raw.get("references")
    return isinstance(refs, list) and len(refs) > 0


def _wegweiser_priority_tier(raw: Mapping[str, Any]) -> int:
    """Lower tier = higher priority in wegweiser list."""
    role = _typology_role_of(raw)
    belegt = _raw_references_belegt(raw)
    if role == "member":
        return 0 if belegt else 1
    if role == "group":
        return 2 if belegt else 4
    if str(raw.get("chunk_type") or "") == "chapter_summary":
        return 3 if belegt else 5
    if str(raw.get("chunk_type") or "") == "begriff":
        return 6
    return 7


def _sort_wegweiser_raw(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda r: (_wegweiser_priority_tier(r), -float(r.get("score", 0))),
    )


def _dedupe_wegweiser_raw(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for r in rows:
        cid = str(r.get("chunk_id") or "")
        if not cid:
            continue
        prev = best.get(cid)
        if prev is None or float(r.get("score", 0)) > float(prev.get("score", 0)):
            best[cid] = r
    return list(best.values())


def _rag_row_to_weg_raw(row: Mapping[str, Any], score: float) -> dict[str, Any]:
    from app.mcp_server.typology_tools import typology_fields_for_search_result

    meta = _meta_from_row(row)
    meta_with_type = {**meta, "chunk_type": row.get("chunk_type") or meta.get("chunk_type") or "typology"}
    text_val = _mcp_text(str(row.get("text") or ""))
    refs_norm = _normalize_stored_references(row)
    out: dict[str, Any] = {
        "chunk_id": str(row["chunk_id"]),
        "chunk_type": str(row.get("chunk_type") or "typology"),
        "score": score,
        "snippet": _snippet(text_val),
        "text": text_val,
        "segment_title": meta.get("segment_title"),
        "title": meta.get("source_title"),
    }
    out.update(typology_fields_for_search_result(meta_with_type))
    for key in ("schlagwort", "schlagsatz", "kurztext", "kuratiert"):
        val = meta.get(key)
        if val is not None and val != "":
            out[key] = _mcp_text(val) if isinstance(val, str) else val
    glieder = _normalize_glieder(meta.get("glieder"))
    if glieder:
        out["glieder"] = glieder
    if refs_norm:
        out["references"] = refs_norm
    return out


def _fetch_typology_members_for_group(
    engine: Any,
    *,
    group_id: str,
    health: str | None,
    typology_slug: str | None,
    partition: str,
    limit: int,
) -> list[dict[str, Any]]:
    gid = (group_id or "").strip()
    if not gid:
        return []

    def _query() -> list[dict[str, Any]]:
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT chunk_id, source_id, chunk_type, text, metadata, "references" AS refs
                    FROM rag_chunks
                    WHERE rag_partition = :part
                      AND chunk_type = 'typology'
                      AND deprecated_at IS NULL
                      AND metadata->>'typology_role' = 'member'
                      AND metadata->>'group_id' = :gid
                      AND (
                        CAST(:health AS text) IS NULL
                        OR metadata->>'health' = CAST(:health AS text)
                      )
                      AND (
                        CAST(:slug AS text) IS NULL
                        OR metadata->>'typology_slug' = CAST(:slug AS text)
                      )
                    ORDER BY
                      CASE
                        WHEN "references" IS NOT NULL
                          AND jsonb_typeof("references") = 'array'
                          AND jsonb_array_length("references") > 0
                          THEN 0
                        WHEN metadata->'references' IS NOT NULL
                          AND jsonb_typeof(metadata->'references') = 'array'
                          AND jsonb_array_length(metadata->'references') > 0
                          THEN 0
                        ELSE 1
                      END,
                      updated_at DESC
                    LIMIT :lim
                    """
                ),
                {
                    "part": partition,
                    "gid": gid,
                    "health": (health or "").strip() or None,
                    "slug": (typology_slug or "").strip() or None,
                    "lim": max(1, limit),
                },
            ).mappings().all()
        return [dict(r) for r in rows]

    return _query()


async def _expand_and_rank_wegweiser(
    weg_raw: list[dict[str, Any]],
    engine: Any,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    from app.config import settings
    from app.retrieval.services.action_prompt_service import load_assistant_rag_collection

    partition = load_assistant_rag_collection(settings.app_default_assistant_slug)
    pool = list(weg_raw)
    groups_seen: set[tuple[str, str, str]] = set()

    for r in weg_raw:
        if _typology_role_of(r) != "group":
            continue
        gid = str(r.get("group_id") or "").strip()
        if not gid:
            continue
        health = str(r.get("health") or "").strip()
        slug = str(r.get("typology_slug") or "").strip()
        key = (gid, health, slug)
        if key in groups_seen:
            continue
        groups_seen.add(key)
        base_score = float(r.get("score", 0))
        member_rows = await asyncio.to_thread(
            _fetch_typology_members_for_group,
            engine,
            group_id=gid,
            health=health or None,
            typology_slug=slug or None,
            partition=partition,
            limit=_MAX_MEMBERS_PER_GROUP,
        )
        for i, row in enumerate(member_rows):
            member_score = (
                base_score * (0.98 - i * 0.01)
                if base_score > 0
                else max(0.02, 0.05 - i * 0.005)
            )
            pool.append(_rag_row_to_weg_raw(row, member_score))

    pool = _dedupe_wegweiser_raw(pool)
    pool = _sort_wegweiser_raw(pool)
    return pool[:limit]


def _split_hits_and_wegweiser(
    raw_results: list[dict[str, Any]],
    *,
    limit_hits: int,
    limit_wegweiser: int | None = _MAX_WEGWEISER,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    hits_raw: list[dict[str, Any]] = []
    weg_raw: list[dict[str, Any]] = []
    for r in raw_results:
        ct = str(r.get("chunk_type") or "")
        if is_original_hit(ct):
            hits_raw.append(r)
        elif is_augment_hit(ct):
            weg_raw.append(r)

    hits_raw.sort(key=lambda x: float(x.get("score", 0)), reverse=True)
    weg_raw.sort(key=lambda x: float(x.get("score", 0)), reverse=True)
    if limit_wegweiser is not None:
        weg_raw = weg_raw[:limit_wegweiser]
    return hits_raw[:limit_hits], weg_raw


def _raw_hit_to_mcp_item(r: dict[str, Any]) -> dict[str, Any]:
    item: dict[str, Any] = {
        "chunk_id": r.get("chunk_id"),
        "chunk_type": r.get("chunk_type"),
        "score": round(float(r.get("score", 0)), 4),
        "snippet": _snippet(r.get("snippet") or r.get("text") or ""),
    }
    for key in (
        "title",
        "segment_title",
        "author",
        "paragraph_id",
        "source_id",
        "navigation_error",
        "quote_verified",
    ):
        val = r.get(key)
        if val is None:
            continue
        item[key] = _mcp_text(val) if isinstance(val, str) else val
    return item


def _assistant_rag_partition() -> str:
    from app.retrieval.services.action_prompt_service import load_assistant_rag_collection

    return load_assistant_rag_collection(settings.app_default_assistant_slug)


def _fetch_chunks_by_ids(engine: Any, chunk_ids: list[str]) -> dict[str, dict[str, Any]]:
    ids = [c.strip() for c in chunk_ids if c and str(c).strip()]
    if not ids:
        return {}

    def _query() -> dict[str, dict[str, Any]]:
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT chunk_id, source_id, chunk_type, text, metadata, "references" AS refs
                    FROM rag_chunks
                    WHERE chunk_id = ANY(:ids)
                      AND deprecated_at IS NULL
                    """
                ),
                {"ids": ids},
            ).mappings().all()
        return {str(r["chunk_id"]): dict(r) for r in rows}

    return _query()


def _references_status_from_row(row: Mapping[str, Any] | None) -> tuple[str, int]:
    if row is None:
        return "unbekannt", 0
    stored = _normalize_stored_references(row)
    count = len(stored)
    return ("belegt" if count else "fehlt"), count


def _build_gegenpol_block(
    cp_row: Mapping[str, Any],
    *,
    pole_health: str | None,
) -> dict[str, Any]:
    cp_meta = _meta_from_row(cp_row)
    cp_ref = str(cp_row["chunk_id"])
    status, count = _references_status_from_row(cp_row)
    block: dict[str, Any] = {
        "ref": cp_ref,
        "titel": _wegweiser_titel({"chunk_id": cp_ref}, cp_meta),
        "references_status": status,
        "references_count": count,
    }
    health = cp_meta.get("health")
    if isinstance(health, str) and health.strip():
        block["health"] = health.strip()
    elif pole_health in ("gesund", "krank"):
        block["health"] = "krank" if pole_health == "gesund" else "gesund"
    return block


def _fetch_group_chunk_ids(
    engine: Any,
    keys: list[tuple[str, str, str]],
    partition: str,
) -> dict[tuple[str, str, str], str]:
    """Map (group_id, health, typology_slug) → group chunk UUID."""
    if not keys:
        return {}
    unique = list(dict.fromkeys(keys))
    gids = list({k[0] for k in unique})

    def _query() -> dict[tuple[str, str, str], str]:
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT chunk_id, metadata
                    FROM rag_chunks
                    WHERE rag_partition = :part
                      AND chunk_type = 'typology'
                      AND deprecated_at IS NULL
                      AND metadata->>'typology_role' = 'group'
                      AND metadata->>'group_id' = ANY(:gids)
                    """
                ),
                {"part": partition, "gids": gids},
            ).mappings().all()
        out: dict[tuple[str, str, str], str] = {}
        for row in rows:
            meta = _meta_from_row(row)
            gid = str(meta.get("group_id") or "").strip()
            health = str(meta.get("health") or "").strip()
            slug = str(meta.get("typology_slug") or "").strip()
            key = (gid, health, slug)
            if key in unique:
                out[key] = str(row["chunk_id"])
        return out

    return _query()


def _compact_wegweiser_entry(entry: dict[str, Any]) -> None:
    entry.pop("_counterpart_chunk_id", None)
    entry.pop("via", None)
    entry.pop("counterpart_chunk_id", None)
    entry.pop("ref_hinweis", None)
    entry.pop("belege_hinweis", None)
    kontext = entry.get("kontext")
    if isinstance(kontext, dict):
        for key in ("group_id", "typology_slug"):
            entry.pop(key, None)


def _wegweiser_response_meta(entries: list[dict[str, Any]]) -> dict[str, str]:
    meta: dict[str, str] = {"ref": _WEGWEISER_HINWEISE_REF}
    if any(
        e.get("references_count") == 0
        and e.get("typology_role") in ("member", "group", "overview")
        for e in entries
    ):
        meta["belege"] = _WEGWEISER_HINWEISE_BELEGE
    return meta


def _fetch_glieder_by_typology_slug(
    engine: Any,
    slugs: list[str],
    partition: str,
) -> dict[str, list[str]]:
    cleaned = list(dict.fromkeys(s.strip() for s in slugs if s and str(s).strip()))
    if not cleaned:
        return {}

    def _query() -> dict[str, list[str]]:
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT lower(trim(metadata->>'typology_slug')) AS slug,
                           metadata->'glieder' AS glieder
                    FROM rag_chunks
                    WHERE rag_partition = :part
                      AND chunk_type = 'typology'
                      AND deprecated_at IS NULL
                      AND metadata->>'typology_role' = 'overview'
                      AND lower(trim(metadata->>'typology_slug')) = ANY(:slugs)
                    """
                ),
                {
                    "part": partition,
                    "slugs": [s.lower() for s in cleaned],
                },
            ).mappings().all()
        out: dict[str, list[str]] = {}
        for r in rows:
            slug = str(r.get("slug") or "").strip()
            glieder = _normalize_glieder(r.get("glieder"))
            if slug and glieder:
                out[slug] = glieder
        return out

    return _query()


async def _enrich_wegweiser_from_db(
    engine: Any,
    entries: list[dict[str, Any]],
) -> None:
    if not entries:
        return
    partition = _assistant_rag_partition()
    ids: list[str] = []
    group_keys: list[tuple[str, str, str]] = []
    slugs_for_glieder: list[str] = []
    for e in entries:
        ref = str(e.get("ref") or "")
        if ref:
            ids.append(ref)
        cp = e.get("_counterpart_chunk_id")
        if isinstance(cp, str) and cp.strip():
            ids.append(cp.strip())
        ctx = e.get("kontext")
        if isinstance(ctx, dict) and ctx.get("group_id") and not ctx.get("group_chunk_id"):
            group_keys.append(
                (
                    str(ctx["group_id"]),
                    str(e.get("health") or ""),
                    str(ctx.get("typology_slug") or ""),
                )
            )
        if not _normalize_glieder(e.get("glieder")):
            slug = ""
            if isinstance(ctx, dict):
                slug = str(ctx.get("typology_slug") or "").strip()
            if slug:
                slugs_for_glieder.append(slug)

    rows = await asyncio.to_thread(_fetch_chunks_by_ids, engine, list(dict.fromkeys(ids)))
    group_map = await asyncio.to_thread(
        _fetch_group_chunk_ids, engine, group_keys, partition
    )
    glieder_by_slug = await asyncio.to_thread(
        _fetch_glieder_by_typology_slug, engine, slugs_for_glieder, partition
    )

    for e in entries:
        ref = str(e.get("ref") or "")
        row = rows.get(ref)
        status, count = _references_status_from_row(row)
        e["references_status"] = status
        e["references_count"] = count

        ctx = e.get("kontext")
        if isinstance(ctx, dict):
            if not ctx.get("group_chunk_id") and row is not None:
                parent = _meta_from_row(row).get("parent_id")
                if isinstance(parent, str) and parent.strip():
                    ctx["group_chunk_id"] = parent.strip()
            gk = (
                str(ctx.get("group_id") or ""),
                str(e.get("health") or ""),
                str(ctx.get("typology_slug") or ""),
            )
            if gk in group_map and not ctx.get("group_chunk_id"):
                ctx["group_chunk_id"] = group_map[gk]

        if not _normalize_glieder(e.get("glieder")):
            from_row = _normalize_glieder(_meta_from_row(row).get("glieder")) if row else None
            if from_row:
                e["glieder"] = from_row
            else:
                slug_key = ""
                if isinstance(ctx, dict):
                    slug_key = str(ctx.get("typology_slug") or "").strip().lower()
                if slug_key and slug_key in glieder_by_slug:
                    e["glieder"] = glieder_by_slug[slug_key]

        cp_id = e.get("_counterpart_chunk_id")
        if isinstance(cp_id, str) and cp_id.strip() and cp_id in rows:
            e["gegenpol"] = _build_gegenpol_block(
                rows[cp_id.strip()],
                pole_health=str(e.get("health") or "") or None,
            )

    for e in entries:
        _compact_wegweiser_entry(e)


def _resolve_ref_identifier(engine: Any, ref: str) -> str | None:
    """Map typology slug to overview chunk_id; pass through UUID chunk_ids."""
    s = (ref or "").strip()
    if not s:
        return None
    if _CHUNK_ID_RE.match(s):
        return s
    partition = _assistant_rag_partition()

    def _overview_id_for_slug(slug: str) -> str | None:
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT chunk_id
                    FROM rag_chunks
                    WHERE rag_partition = :part
                      AND chunk_type = 'typology'
                      AND deprecated_at IS NULL
                      AND metadata->>'typology_role' = 'overview'
                      AND lower(trim(metadata->>'typology_slug')) = lower(trim(:slug))
                    LIMIT 1
                    """
                ),
                {"part": partition, "slug": slug},
            ).mappings().first()
        return str(row["chunk_id"]) if row else None

    return _overview_id_for_slug(s)


def _fetch_chunk_row(engine: Any, chunk_id: str) -> dict[str, Any] | None:
    cid = (chunk_id or "").strip()
    if not cid:
        return None

    def _query() -> dict[str, Any] | None:
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT chunk_id, source_id, chunk_type, text, metadata, "references" AS refs
                    FROM rag_chunks
                    WHERE chunk_id = :cid AND deprecated_at IS NULL
                    """
                ),
                {"cid": cid},
            ).mappings().first()
        return dict(row) if row else None

    return _query()


def _meta_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    meta = row.get("metadata")
    if isinstance(meta, Mapping):
        return dict(meta)
    return {}


def _normalize_stored_references(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    refs = row.get("refs")
    meta = _meta_from_row(row)
    if not refs:
        refs = meta.get("references")
    return _normalize_chunk_references(refs)


def _via_kind_for_anchor(anchor_type: str) -> str:
    if anchor_type == "typology":
        return "typology"
    if anchor_type == "curated_text":
        return "curated_text"
    return "begriff"


def _score_from_ref(
    ref_entry: Mapping[str, Any],
    parent_score: float | None,
) -> float:
    score = float(ref_entry.get("relevance") or 1.0)
    if parent_score is not None and parent_score > 0:
        score = min(1.0, score * float(parent_score))
    return score


def _hit_from_ga_angabe_ref(
    ref_entry: Mapping[str, Any],
    *,
    via_base: dict[str, Any],
    parent_score: float | None,
) -> dict[str, Any]:
    """Build a non-citable hit from denormalized GA reference fields (no DB body)."""
    text_val = str(ref_entry.get("snippet") or ref_entry.get("description") or "")
    title = (
        ref_entry.get("title")
        or ref_entry.get("vortragstitel")
        or ref_entry.get("segment_title")
    )
    item = _raw_hit_to_mcp_item(
        {
            "chunk_id": ref_entry["chunk_id"],
            "chunk_type": "ga_text",
            "source_id": ref_entry.get("source_id"),
            "score": _score_from_ref(ref_entry, parent_score),
            "snippet": text_val,
            "text": text_val,
            "title": title,
            "segment_title": ref_entry.get("segment_title")
            or ref_entry.get("vortragstitel"),
        }
    )
    item["via"] = dict(via_base)
    item["zitierbar"] = False
    item.pop("citation", None)
    for key in ("ga", "lecture_id", "page_start", "page_end", "vortragstitel"):
        if ref_entry.get(key) is not None and ref_entry.get(key) != "":
            item[key] = ref_entry[key]
    desc = ref_entry.get("description")
    if desc:
        item["reference_note"] = desc
    return item


async def _hits_from_stored_references(
    engine: Any,
    *,
    stored: list[dict[str, Any]],
    via_base: dict[str, Any],
    limit: int,
    parent_score: float | None = None,
) -> list[dict[str, Any]]:
    """Resolve stored references into MCP hits (corpus citable; ga_text Angaben not)."""
    if not stored:
        return []
    capped = stored[: max(1, min(limit, 20))]
    book_ids = [
        r["chunk_id"]
        for r in capped
        if not _is_ga_angabe_ref(r)
    ]

    def _load_bodies() -> list[dict[str, Any]]:
        if not book_ids:
            return []
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT chunk_id, source_id, chunk_type, text, metadata
                    FROM rag_chunks
                    WHERE chunk_id = ANY(:ids)
                      AND deprecated_at IS NULL
                      AND chunk_type IN ('book', 'secondary_book')
                    """
                ),
                {"ids": book_ids},
            ).mappings().all()
        return [dict(r) for r in rows]

    bodies = await asyncio.to_thread(_load_bodies)
    by_id = {str(r["chunk_id"]): r for r in bodies}
    body_rows: list[dict[str, Any]] = []
    corpus_rows: list[dict[str, Any]] = []
    for ref_entry in capped:
        if _is_ga_angabe_ref(ref_entry):
            body_rows.append(
                _hit_from_ga_angabe_ref(
                    ref_entry, via_base=via_base, parent_score=parent_score
                )
            )
            continue
        cid = ref_entry["chunk_id"]
        body = by_id.get(cid)
        if body is None:
            continue
        bmeta = _meta_from_row(body)
        text_val = str(body.get("text") or "")
        item = _raw_hit_to_mcp_item(
            {
                "chunk_id": body["chunk_id"],
                "chunk_type": body.get("chunk_type"),
                "source_id": body.get("source_id"),
                "score": _score_from_ref(ref_entry, parent_score),
                "snippet": text_val,
                "text": text_val,
                "title": bmeta.get("book_title") or bmeta.get("source_title"),
                "segment_title": bmeta.get("segment_title"),
                "author": bmeta.get("author"),
            }
        )
        item["via"] = dict(via_base)
        desc = ref_entry.get("description")
        if desc:
            item["reference_note"] = desc
        body_rows.append(item)
        corpus_rows.append(item)

    chunk_ids = [str(h["chunk_id"]) for h in corpus_rows if h.get("chunk_id")]
    pid_map = await _lookup_paragraph_ids(chunk_ids, engine) if chunk_ids else {}
    for h in corpus_rows:
        if not h.get("paragraph_id"):
            h["paragraph_id"] = pid_map.get(str(h.get("chunk_id") or ""))
    await _enrich_corpus_hits(corpus_rows, corpus_rows, engine)
    for h in body_rows:
        if h.get("zitierbar") is False:
            h.pop("citation", None)
    return body_rows


async def _expand_curated_matches_to_hits(
    engine: Any,
    curated_raw: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Turn curated_text search matches into corpus hits via stored references (no wegweiser)."""
    if not curated_raw:
        return []
    curated_raw = sorted(
        curated_raw, key=lambda x: float(x.get("score", 0)), reverse=True
    )
    seen_hit: set[str] = set()
    out: list[dict[str, Any]] = []
    for raw in curated_raw:
        cid = str(raw.get("chunk_id") or "").strip()
        if not cid:
            continue
        row = await asyncio.to_thread(_fetch_chunk_row, engine, cid)
        if row is None:
            continue
        stored = _normalize_stored_references(row)
        via_base = {
            "kind": "curated_text",
            "ref": cid,
            "weg": "references",
        }
        hits = await _hits_from_stored_references(
            engine,
            stored=stored,
            via_base=via_base,
            limit=limit,
            parent_score=float(raw.get("score") or 0) or None,
        )
        for h in hits:
            hid = str(h.get("chunk_id") or "")
            if not hid or hid in seen_hit:
                continue
            seen_hit.add(hid)
            out.append(h)
            if len(out) >= limit:
                return out
    return out


async def _search_corpus_by_ref(ref: str, *, limit: int) -> dict[str, Any]:
    engine = get_engine()
    raw_ref = (ref or "").strip()
    resolved = await asyncio.to_thread(_resolve_ref_identifier, engine, raw_ref)
    if resolved is None:
        hint = (
            "ref must be a chunk UUID (wegweiser.ref / via.counterpart_ref) "
            "or a typology slug for the overview chunk."
        )
        if raw_ref.startswith("feld-"):
            hint += (
                " group_id (feld-N) is not a ref — use wegweiser.kontext.group_chunk_id "
                "or a member ref (UUID)."
            )
        return {"error": hint, "code": "ref_not_found", "ref": raw_ref}

    row = await asyncio.to_thread(_fetch_chunk_row, engine, resolved)
    if row is None:
        return {"error": "ref not found", "code": "ref_not_found", "ref": raw_ref}

    anchor_type = str(row.get("chunk_type") or "")
    if anchor_type not in _REF_ANCHOR_CHUNK_TYPES:
        return {
            "error": "ref must be a concept, typology, or curated_text chunk_id",
            "code": "ref_invalid_kind",
        }

    stored = _normalize_stored_references(row)
    via_base = {
        "kind": _via_kind_for_anchor(anchor_type),
        "ref": ref,
        "weg": "references",
    }

    # curated_text: only expanded corpus hits — never a wegweiser anchor
    if anchor_type == "curated_text":
        body_rows = await _hits_from_stored_references(
            engine, stored=stored, via_base=via_base, limit=limit
        )
        return {"hits": body_rows, "wegweiser": []}

    meta = _meta_from_row(row)
    anchor_weg = _build_wegweiser(
        {
            "chunk_id": row["chunk_id"],
            "chunk_type": anchor_type,
            "score": 1.0,
            "segment_title": meta.get("segment_title"),
            "title": meta.get("source_title"),
        },
        meta,
    )

    await _enrich_wegweiser_from_db(engine, [anchor_weg])

    if not stored:
        hinweise = _wegweiser_response_meta([anchor_weg])
        return {
            "hits": [],
            "wegweiser": [anchor_weg],
            "wegweiser_hinweise": hinweise,
            "references_status": anchor_weg.get("references_status", "fehlt"),
            "references_count": anchor_weg.get("references_count", 0),
        }

    body_rows = await _hits_from_stored_references(
        engine, stored=stored, via_base=via_base, limit=limit
    )
    return {
        "hits": body_rows,
        "wegweiser": [anchor_weg],
        "wegweiser_hinweise": _wegweiser_response_meta([anchor_weg]),
        "references_status": "belegt",
    }


async def search_corpus(
    query: str,
    types: list[str] | None = None,
    limit: int = 10,
    ref: str | None = None,
    quotes: str | None = None,
    author: str | None = None,
) -> dict[str, Any]:
    """Search the corpus: original sources in ``hits``, augment in ``wegweiser``.

    Catalog quotes mode (``quotes=author|quoted``): slim hits only
    ``{author, text, return_url}``; empty hits include ``authors`` list.
    """
    ref_id = (ref or "").strip()
    if ref_id:
        k = max(1, min(int(limit) if limit is not None else 10, 20))
        return await _search_corpus_by_ref(ref_id, limit=k)

    quotes_kind = (quotes or "").strip().lower() or None
    if quotes_kind is not None:
        if quotes_kind not in ("author", "quoted"):
            return {
                "error": "quotes must be 'author' or 'quoted'",
                "code": "invalid_quotes",
            }
        k = max(1, min(int(limit) if limit is not None else 10, 20))
        return await app_search_catalog_quotes(
            query=(query or "").strip(),
            quotes_kind=quotes_kind,
            author=(author or "").strip() or None,
            limit=k,
            engine=get_engine(),
        )

    q = (query or "").strip()
    if not q:
        return {"error": "query is required", "code": "empty_query"}

    type_err = _validate_search_types(types)
    if type_err:
        return type_err

    k = max(1, min(int(limit) if limit is not None else 10, 20))
    # Retrieve enough candidates so augment can fill wegweiser after the split.
    fetch_k = min(50, max(k * 3, k + _MAX_WEGWEISER))
    raw_results = await app_search(
        query=q,
        types=types,
        limit=fetch_k,
        engine=get_engine(),
    )

    only_originals = types is not None and all(
        (t or "").strip().lower() in ("text", "quote", "curated_text") for t in types
    )
    only_augment = types is not None and all(
        (t or "").strip().lower() in ("concept", "chapter_summary") for t in types
    )

    hits_raw, weg_raw = _split_hits_and_wegweiser(
        raw_results,
        limit_hits=k,
        limit_wegweiser=None if not only_originals else 0,
    )
    if only_augment:
        hits_raw = []

    curated_raw = [
        r
        for r in raw_results
        if str(r.get("chunk_type") or "") == "curated_text"
    ]

    out_hits = [_raw_hit_to_mcp_item(r) for r in hits_raw]
    engine = get_engine()
    await _enrich_corpus_hits(out_hits, hits_raw, engine)

    # curated_text never in hits/wegweiser — expand stored references into corpus hits
    if curated_raw and not only_augment:
        expanded = await _expand_curated_matches_to_hits(
            engine, curated_raw, limit=k
        )
        seen = {str(h.get("chunk_id") or "") for h in out_hits}
        for h in expanded:
            hid = str(h.get("chunk_id") or "")
            if not hid or hid in seen:
                continue
            seen.add(hid)
            out_hits.append(h)
            if len(out_hits) >= k:
                break
        out_hits.sort(key=lambda x: float(x.get("score", 0)), reverse=True)
        out_hits = out_hits[:k]

    if not only_originals and weg_raw:
        weg_raw = await _expand_and_rank_wegweiser(
            weg_raw, engine, limit=_MAX_WEGWEISER
        )

    wegweiser: list[dict[str, Any]] = []
    for r in weg_raw:
        meta = dict(r)
        wegweiser.append(_build_wegweiser(r, meta))

    if wegweiser:
        await _enrich_wegweiser_from_db(engine, wegweiser)

    payload: dict[str, Any] = {"hits": out_hits, "wegweiser": wegweiser}
    if wegweiser:
        payload["wegweiser_hinweise"] = _wegweiser_response_meta(wegweiser)
    return payload


# ---------------------------------------------------------------------------
# get_passage
# ---------------------------------------------------------------------------


def _sanitize_passage_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Strip soft hyphens from passage text fields before MCP return."""
    paras = payload.get("paragraphs")
    if isinstance(paras, list):
        for p in paras:
            if isinstance(p, dict) and isinstance(p.get("text"), str):
                p["text"] = _mcp_text(p["text"])
    if isinstance(payload.get("text"), str):
        payload["text"] = _mcp_text(payload["text"])
    return payload


async def get_passage(
    paragraph_id: str | None = None,
    source_id: str | None = None,
    segment_index: int | None = None,
    from_paragraph: int | None = None,
    to_paragraph: int | None = None,
    before: int = 0,
    after: int = 0,
) -> dict[str, Any]:
    """Read one or more paragraphs with optional context and nav (plan §4).

    Address by ``paragraph_id`` (with optional ``before``/``after``) or by
    ``source_id``, ``segment_index``, and ``from_paragraph`` (optional ``to_paragraph``).
    """
    from app.services.passage_read_service import read_passages

    engine = get_engine()
    payload = await asyncio.to_thread(
        read_passages,
        engine,
        paragraph_id=paragraph_id,
        source_id=source_id,
        segment_index=segment_index,
        from_paragraph=from_paragraph,
        to_paragraph=to_paragraph,
        before=before,
        after=after,
    )
    if isinstance(payload, dict) and not payload.get("error"):
        return _sanitize_passage_payload(payload)
    return payload


# ---------------------------------------------------------------------------
# list_volumes
# ---------------------------------------------------------------------------


def lecture_display_title(
    stored_title: str | None,
    vortragstitel: str | None,
    anlass: str | None,
    ort: str | None,
    datum: str | None,
    zyklus_titel: str | None,
    lecture_id: str,
) -> str:
    """Display title for a catalog lecture. Never embeds a GA number."""
    stored = (stored_title or "").strip()
    if stored:
        return stored
    title = (vortragstitel or "").strip()
    if title:
        return title
    location = ", ".join(p for p in ((ort or "").strip(), (datum or "").strip()) if p)
    anlass_txt = (anlass or "").strip()
    if anlass_txt:
        return f"{anlass_txt} ({location})" if location else anlass_txt
    cycle = (zyklus_titel or "").strip()
    if cycle and location:
        return f"{cycle} — {location}"
    if location:
        return location
    return f"Vortrag {lecture_id}"


def _lecture_location(ort: str | None, datum: str | None) -> str:
    return ", ".join(p for p in ((ort or "").strip(), (datum or "").strip()) if p)


def _lecture_payload(row: dict[str, Any]) -> dict[str, Any]:
    lecture_id = str(row.get("lecture_id") or "")
    display = lecture_display_title(
        row.get("stored_title"),
        row.get("vortragstitel"),
        row.get("anlass"),
        row.get("ort"),
        row.get("datum"),
        row.get("zyklus_titel"),
        lecture_id,
    )
    ort = str(row.get("ort") or "").strip()
    datum = str(row.get("datum") or "").strip()
    location = _lecture_location(ort, datum)
    zyklus = row.get("zyklus")
    return {
        "lecture_id": lecture_id,
        "source_id": str(row["source_id"]) if row.get("source_id") else None,
        "vortragstitel": (row.get("vortragstitel") or None),
        "display_title": display,
        "datum": datum or None,
        "ort": ort or None,
        "reihe": (row.get("reihe") or None),
        "zyklus": int(zyklus) if zyklus is not None else None,
        "zyklus_titel": (row.get("zyklus_titel") or None),
        "has_chunks": bool(row.get("has_chunks")),
        "zitierform": format_zitierform(display, location, None),
    }


_LECTURE_SELECT = """
    SELECT
        l.id AS lecture_id,
        l.uuid::text AS source_id,
        l.ga,
        l.vortragstitel,
        l.datum,
        l.ort,
        l.reihe,
        l.zyklus,
        l.anlass,
        l.has_chunks,
        c.titel AS zyklus_titel,
        s.title AS stored_title
    FROM rag_lecture_catalog l
    LEFT JOIN rag_lecture_cycles c ON c.zyklus = l.zyklus
    LEFT JOIN rag_sources s ON s.id::text = l.uuid::text
"""


async def list_volumes() -> list[dict[str, Any]]:
    """List all available books/volumes in the corpus.

    Returns source_id, display_name, source_type, is_primary, ga, zyklus for each volume.
    Use list_lectures(ga) for every lecture in a GA volume (including without chunks).
    """
    catalog = PostgresCatalogRepository(get_engine())
    sources = await catalog.list_sources()
    return [
        {
            "source_id": s["source_id"],
            "display_name": s["display_name"],
            "source_type": s.get("source_type") or "book",
            "is_primary": bool(s.get("is_primary")),
            "ga": s.get("ga"),
            "zyklus": s.get("zyklus"),
        }
        for s in sources
    ]


# ---------------------------------------------------------------------------
# get_lecture_info / list_lectures
# ---------------------------------------------------------------------------


async def get_lecture_info(source_id: str) -> dict[str, Any]:
    """Look up a lecture by catalog UUID or lecture id (YYYYMMDD[a-z]).

    Returns location, date, cycle, whether chunks exist, and a title without GA.
    Use list_lectures to list every lecture in the same GA volume.

    Args:
        source_id: Lecture UUID (rag_sources / catalog) or catalog id.
    """
    sid = (source_id or "").strip()
    if not sid:
        return {"error": "source_id is required", "code": "empty_source_id"}

    def _query() -> dict[str, Any] | None:
        with get_engine().connect() as conn:
            row = conn.execute(
                text(
                    _LECTURE_SELECT
                    + """
                    WHERE l.uuid::text = :sid OR l.id = :sid
                    """
                ),
                {"sid": sid},
            ).mappings().first()
        return dict(row) if row else None

    row = await asyncio.to_thread(_query)
    if row is None:
        return {"error": "lecture not found", "code": "lecture_not_found"}

    payload = _lecture_payload(row)
    payload["ga"] = (str(row["ga"]).strip() if row.get("ga") else None)
    return payload


async def list_lectures(ga: str) -> dict[str, Any]:
    """List all catalog lectures in a GA volume.

    `ga` is an internal lookup field. Titles and zitierform never include the
    GA number. The catalog may list lectures that are not yet in rag_sources.

    Args:
        ga: GA volume number (e.g. "151", "337b").
    """
    ga_key = (ga or "").strip()
    if not ga_key:
        return {"error": "ga is required", "code": "empty_ga"}

    def _query() -> list[dict[str, Any]]:
        with get_engine().connect() as conn:
            rows = conn.execute(
                text(
                    _LECTURE_SELECT
                    + """
                    WHERE lower(trim(l.ga)) = lower(:ga)
                       OR (',' || lower(replace(l.ga, ' ', '')) || ',')
                          LIKE '%,' || lower(:ga) || ',%'
                    ORDER BY l.lecture_date ASC NULLS LAST, l.id ASC
                    """
                ),
                {"ga": ga_key},
            ).mappings().all()
        return [dict(r) for r in rows]

    rows = await asyncio.to_thread(_query)
    lectures = [_lecture_payload(r) for r in rows]
    return {
        "ga": ga_key,
        "lecture_count": len(lectures),
        "lectures": lectures,
    }


# ---------------------------------------------------------------------------
# get_protocol
# ---------------------------------------------------------------------------


async def get_protocol(
    source_id: str,
    segment_slug: str,
) -> dict[str, Any]:
    """Read the user's study protocol for a specific chapter/segment.

    Protocols collect notes, questions, and insights per chapter.

    Args:
        source_id: Book/source ID.
        segment_slug: Chapter identifier (e.g. segment index or slug).
    """
    user_id = _get_user_id()
    if not user_id:
        return {"error": "not authenticated"}

    sid = (source_id or "").strip()
    raw_slug = (segment_slug or "").strip()
    if not sid or not raw_slug:
        return {"error": "source_id and segment_slug are required"}

    engine = get_engine()
    slug = await asyncio.to_thread(resolve_segment_slug, engine, sid, raw_slug)
    if not slug:
        return {"error": f"segment not found: {raw_slug}", "code": "segment_not_found"}

    def _query() -> dict[str, Any] | None:
        with engine.connect() as conn:
            proto = conn.execute(
                text(
                    """
                    SELECT id::text AS protocol_id, created_at, updated_at
                    FROM protocols
                    WHERE user_id = CAST(:uid AS uuid)
                      AND source_id = :sid
                      AND segment_slug = :slug
                    """
                ),
                {"uid": user_id, "sid": sid, "slug": slug},
            ).mappings().first()
            if not proto:
                return None

            entries = conn.execute(
                text(
                    """
                    SELECT
                        id::text AS entry_id,
                        paragraph_id::text AS paragraph_id,
                        entry_type,
                        content,
                        created_at
                    FROM protocol_entries
                    WHERE protocol_id = CAST(:pid AS uuid)
                    ORDER BY created_at ASC
                    """
                ),
                {"pid": proto["protocol_id"]},
            ).mappings().all()

            return {
                "protocol_id": proto["protocol_id"],
                "source_id": sid,
                "segment_slug": slug,
                "created_at": str(proto["created_at"]),
                "updated_at": str(proto["updated_at"]),
                "entries": [
                    {
                        "entry_id": e["entry_id"],
                        "paragraph_id": e["paragraph_id"],
                        "entry_type": e["entry_type"],
                        "content": _snippet(e["content"], 500),
                        "created_at": str(e["created_at"]),
                    }
                    for e in entries
                ],
            }

    result = await asyncio.to_thread(_query)
    if result is None:
        return {"protocol": None, "message": "No protocol found for this chapter."}

    chapter = await asyncio.to_thread(
        fetch_chapter_citation,
        engine,
        str(result["source_id"]),
        str(result["segment_slug"]),
    )
    if chapter:
        result["chapter_citation"] = chapter

    entry_pids = [
        str(e["paragraph_id"])
        for e in result.get("entries", [])
        if e.get("paragraph_id")
    ]
    entry_citations = await asyncio.to_thread(fetch_citations_batch, engine, entry_pids)
    for entry in result.get("entries", []):
        pid = entry.get("paragraph_id")
        if pid and str(pid) in entry_citations:
            entry["citation"] = entry_citations[str(pid)]

    return result


# ---------------------------------------------------------------------------
# list_work_texts
# ---------------------------------------------------------------------------


async def list_work_texts(
    text_type: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """List the user's work texts (notes, drafts, essays).

    Args:
        text_type: Filter by type (e.g. "note", "essay"). Default: all.
        limit: Max results (1-50, default 20).
    """
    user_id = _get_user_id()
    if not user_id:
        return [{"error": "not authenticated"}]

    # limit=0 must clamp to 1 (``limit or 20`` would treat 0 as missing).
    k = max(1, min(int(limit) if limit is not None else 20, 50))

    def _query() -> list[dict[str, Any]]:
        params: dict[str, Any] = {"uid": user_id, "lim": k}
        type_clause = ""
        if text_type and text_type.strip():
            type_clause = "AND text_type = :ttype"
            params["ttype"] = text_type.strip()

        with get_engine().connect() as conn:
            rows = conn.execute(
                text(
                    f"""
                    SELECT
                        id,
                        title,
                        LEFT(content, 2048) AS content_preview,
                        text_type,
                        status,
                        version,
                        paragraph_id,
                        created_by,
                        created_at,
                        updated_at
                    FROM app_notes
                    WHERE user_id = CAST(:uid AS uuid)
                      AND deleted_at IS NULL
                      {type_clause}
                    ORDER BY updated_at DESC
                    LIMIT :lim
                    """
                ),
                params,
            ).mappings().all()
        return list(rows)

    rows = await asyncio.to_thread(_query)
    engine = get_engine()
    anchor_pids = [
        str(r["paragraph_id"])
        for r in rows
        if r.get("paragraph_id")
    ]
    anchor_citations = await asyncio.to_thread(fetch_citations_batch, engine, anchor_pids)

    out: list[dict[str, Any]] = []
    for row in rows:
        pid = str(row["paragraph_id"]) if row.get("paragraph_id") else None
        citation = anchor_citations.get(pid) if pid else None
        display, source = note_display_title(
            row.get("title"),
            row.get("content_preview"),
            citation,
        )
        out.append(
            {
                "note_id": row["id"],
                "title": row["title"],
                "display_title": display,
                "title_source": source,
                "text_type": row["text_type"],
                "status": row["status"],
                "version": row["version"],
                "paragraph_id": pid,
                "created_by": row["created_by"],
                "updated_at": str(row["updated_at"]),
            }
        )
    return out


# ---------------------------------------------------------------------------
# get_work_text
# ---------------------------------------------------------------------------


async def get_work_text(note_id: str) -> dict[str, Any]:
    """Read a specific work text by its ID.

    Returns title, content, version, and metadata.

    Args:
        note_id: The note ID.
    """
    user_id = _get_user_id()
    if not user_id:
        return {"error": "not authenticated"}

    nid = (note_id or "").strip()
    if not nid:
        return {"error": "note_id is required"}

    def _query() -> dict[str, Any] | None:
        with get_engine().connect() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT
                        id,
                        title,
                        content,
                        text_type,
                        status,
                        version,
                        paragraph_id,
                        conversation_url,
                        created_by,
                        created_at,
                        updated_at
                    FROM app_notes
                    WHERE id = :nid
                      AND user_id = CAST(:uid AS uuid)
                      AND deleted_at IS NULL
                    """
                ),
                {"nid": nid, "uid": user_id},
            ).mappings().first()
        if not row:
            return None
        return dict(row)

    result = await asyncio.to_thread(_query)
    if result is None:
        return {"error": "note not found"}

    engine = get_engine()
    pid = str(result["paragraph_id"]) if result.get("paragraph_id") else None
    citation = None
    if pid:
        citation = await asyncio.to_thread(fetch_citation, engine, pid)

    display, source = note_display_title(
        result.get("title"),
        result.get("content"),
        citation,
    )
    payload: dict[str, Any] = {
        "note_id": result["id"],
        "title": result["title"],
        "display_title": display,
        "title_source": source,
        "content": _mcp_text(result.get("content")),
        "text_type": result["text_type"],
        "status": result["status"],
        "version": result["version"],
        "paragraph_id": pid,
        "conversation_url": result["conversation_url"],
        "created_by": result["created_by"],
        "created_at": str(result["created_at"]),
        "updated_at": str(result["updated_at"]),
        "text_return_url": text_return_url(str(result["id"])),
    }
    if citation:
        payload["citation"] = citation
    return payload


# ---------------------------------------------------------------------------
# create_work_text (Step 13)
# ---------------------------------------------------------------------------


async def create_work_text(
    title: str,
    content: str,
    text_type: str = "note",
    paragraph_id: str | None = None,
    conversation_url: str | None = None,
) -> dict[str, Any]:
    """Create a new work text (note, draft, essay).

    Returns the new note_id and version on success.

    Args:
        title: Title of the work text.
        content: The text content.
        text_type: Type of text: "note", "essay", "draft". Default: "note".
        paragraph_id: Optional paragraph UUID to link this note to.
        conversation_url: Optional URL of the conversation that created this.
    """
    raw_token = _get_raw_token()
    if not raw_token:
        return {"error": "not authenticated"}

    too_long = _content_too_long(content)
    if too_long:
        return too_long

    note_id = str(uuid.uuid4())
    params: dict[str, Any] = {
        "p_id": note_id,
        "p_title": title.strip(),
        "p_content": content,
        "p_text_type": text_type.strip() or "note",
        "p_created_by": "claude",
    }
    if paragraph_id and paragraph_id.strip():
        params["p_paragraph_id"] = paragraph_id.strip()
    if conversation_url and conversation_url.strip():
        params["p_conversation_url"] = conversation_url.strip()

    return await _supabase_rpc("create_note", params, raw_token)


# ---------------------------------------------------------------------------
# update_work_text (Step 13)
# ---------------------------------------------------------------------------


async def update_work_text(
    note_id: str,
    content: str,
    expected_version: int,
    title: str | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    """Update an existing work text with optimistic version control.

    Returns {ok, new_version} on success, or {error: "conflict", current_version,
    current_content, current_title} if the version has changed since last read.
    Always read the note first with get_work_text to get the current version.

    Args:
        note_id: The note ID to update.
        content: The new content.
        expected_version: The version you read (from get_work_text). Must match.
        title: Optional new title (keeps current if omitted).
        status: Optional new status: "draft" or "final" (keeps current if omitted).
    """
    raw_token = _get_raw_token()
    if not raw_token:
        return {"error": "not authenticated"}

    nid = (note_id or "").strip()
    if not nid:
        return {"error": "note_id is required"}

    too_long = _content_too_long(content)
    if too_long:
        return too_long

    params: dict[str, Any] = {
        "p_id": nid,
        "p_content": content,
        "p_expected_version": expected_version,
        "p_changed_by": "claude",
    }
    if title is not None:
        params["p_title"] = title.strip()
    if status is not None:
        params["p_status"] = status.strip()

    return await _supabase_rpc("save_note", params, raw_token)


# ---------------------------------------------------------------------------
# append_to_protocol (Step 13)
# ---------------------------------------------------------------------------


async def append_to_protocol(
    source_id: str,
    segment_slug: str,
    entry_type: str,
    content: str,
    paragraph_id: str | None = None,
    conversation_url: str | None = None,
) -> dict[str, Any]:
    """Append a note/question/insight to the user's study protocol for a chapter.

    Creates the protocol if it does not exist yet.

    Args:
        source_id: Book/source ID.
        segment_slug: Chapter identifier (segment index or slug).
        entry_type: Type of entry: "note", "question", "insight", "summary".
        content: The entry text.
        paragraph_id: Optional paragraph UUID this entry refers to.
        conversation_url: Optional URL of the conversation.
    """
    user_id = _get_user_id()
    if not user_id:
        return {"error": "not authenticated"}

    sid = (source_id or "").strip()
    raw_slug = (segment_slug or "").strip()
    etype = (entry_type or "").strip()
    if not sid or not raw_slug or not etype:
        return {"error": "source_id, segment_slug, and entry_type are required"}

    engine = get_engine()
    slug = await asyncio.to_thread(resolve_segment_slug, engine, sid, raw_slug)
    if not slug:
        return {"error": f"segment not found: {raw_slug}", "code": "segment_not_found"}

    def _write() -> dict[str, Any]:
        with engine.connect() as conn:
            # Upsert protocol
            row = conn.execute(
                text(
                    """
                    INSERT INTO protocols (user_id, source_id, segment_slug)
                    VALUES (CAST(:uid AS uuid), :sid, :slug)
                    ON CONFLICT (user_id, source_id, segment_slug)
                    DO UPDATE SET updated_at = now()
                    RETURNING id::text AS protocol_id
                    """
                ),
                {"uid": user_id, "sid": sid, "slug": slug},
            ).mappings().first()

            if not row:
                return {"error": "failed to create protocol"}

            protocol_id = row["protocol_id"]

            # Insert entry
            params: dict[str, Any] = {
                "pid": protocol_id,
                "etype": etype,
                "content": content,
            }
            pid_col = "NULL"
            conv_col = "NULL"
            if paragraph_id and paragraph_id.strip():
                params["para_id"] = paragraph_id.strip()
                pid_col = "CAST(:para_id AS uuid)"
            if conversation_url and conversation_url.strip():
                params["conv_url"] = conversation_url.strip()
                conv_col = ":conv_url"

            entry = conn.execute(
                text(
                    f"""
                    INSERT INTO protocol_entries
                        (protocol_id, paragraph_id, entry_type, content, conversation_url)
                    VALUES
                        (CAST(:pid AS uuid), {pid_col}, :etype, :content, {conv_col})
                    RETURNING id::text AS entry_id
                    """
                ),
                params,
            ).mappings().first()

            conn.commit()

            return {
                "ok": True,
                "protocol_id": protocol_id,
                "entry_id": entry["entry_id"] if entry else None,
            }

    return await asyncio.to_thread(_write)


# ---------------------------------------------------------------------------
# get_handoff (Step 13)
# ---------------------------------------------------------------------------


async def get_handoff(handoff_id: str) -> dict[str, Any]:
    """Retrieve a handoff by its short ID.

    A handoff transfers context from the app to Claude: marked text, a user
    question, and the paragraph reference. If the handoff has expired, returns
    what is still available (paragraph_id) with an explanation.

    Args:
        handoff_id: The 5-character handoff ID (from the app deep link).
    """
    user_id = _get_user_id()
    if not user_id:
        return {"error": "not authenticated"}

    hid = (handoff_id or "").strip()
    if not hid:
        return {"error": "handoff_id is required"}

    def _query() -> dict[str, Any] | None:
        with get_engine().connect() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT
                        id,
                        paragraph_id,
                        source_id,
                        segment_slug,
                        marked_text,
                        user_question,
                        return_url,
                        created_at,
                        expires_at,
                        expires_at < now() AS is_expired
                    FROM handoffs
                    WHERE id = :hid
                      AND user_id = CAST(:uid AS uuid)
                    """
                ),
                {"hid": hid, "uid": user_id},
            ).mappings().first()
        if not row:
            return None
        return dict(row)

    result = await asyncio.to_thread(_query)
    if result is None:
        return {"error": "handoff not found"}

    engine = get_engine()
    paragraph_id = (
        str(result["paragraph_id"]) if result.get("paragraph_id") else None
    )
    citation = None
    if paragraph_id:
        citation = await asyncio.to_thread(fetch_citation, engine, paragraph_id)

    if result.get("is_expired"):
        out: dict[str, Any] = {
            "expired": True,
            "message": "Diese Uebergabe ist abgelaufen. Du kannst den Absatz aber noch direkt lesen.",
        }
        if paragraph_id:
            out["paragraph_id"] = paragraph_id
        if citation:
            out["citation"] = citation
        if result.get("user_question"):
            out["user_question"] = result["user_question"]
        return out

    payload: dict[str, Any] = {
        "handoff_id": result["id"],
        "paragraph_id": paragraph_id,
        "source_id": result["source_id"],
        "segment_slug": result["segment_slug"],
        "marked_text": result["marked_text"],
        "user_question": result["user_question"],
        "return_url": result["return_url"],
        "created_at": str(result["created_at"]),
        "expires_at": str(result["expires_at"]),
    }
    if citation:
        payload["citation"] = citation
    return payload


# ---------------------------------------------------------------------------
# search_ga / search_ga_summaries / get_ga_chunk (rudolf-steiner-ga)
# ---------------------------------------------------------------------------

GA_QDRANT_COLLECTION = "rudolf-steiner-ga"


async def get_ga_chunk(chunk_id: str) -> dict[str, Any]:
    """Fetch one GA chunk by ``chunk_id`` from collection rudolf-steiner-ga.

    Always allowed when you already have a ``chunk_id`` (e.g. from ``search_corpus``
    hits with ``chunk_type: ga_text`` / ``zitierbar: false``). Returns full text for
    your understanding — do not pass long continuous passages to the end user; keep
    ``zitierbar: false`` (short cite only as ``(GA <n>, S. <seite>)``
    if needed). For open-ended GA exploration use ``search_ga`` only after an
    explicit user request.

    Args:
        chunk_id: Payload ``chunk_id`` from a GA hit / curated Angabe.
    """
    cid = (chunk_id or "").strip()
    if not cid:
        return {"error": "chunk_id is required", "code": "empty_chunk_id"}

    from app.core.providers import get_qdrant_client

    client = get_qdrant_client()
    try:
        points = await client.scroll_points(
            GA_QDRANT_COLLECTION,
            filter_={"must": [{"key": "chunk_id", "match": {"value": cid}}]},
            limit=3,
            with_payload=True,
            with_vectors=False,
        )
    except Exception as exc:
        logger.exception("get_ga_chunk Qdrant scroll failed")
        return {"error": f"lookup failed: {exc}", "code": "lookup_failed"}

    if not points:
        return {"error": "chunk not found", "code": "not_found", "chunk_id": cid}

    raw_point = points[0]
    payload = raw_point.get("payload") if isinstance(raw_point, Mapping) else {}
    if not isinstance(payload, Mapping):
        payload = {}
    inner = payload.get("payload") if isinstance(payload.get("payload"), Mapping) else payload
    if not isinstance(inner, Mapping):
        inner = {}
    meta = dict(inner)
    if "text" not in meta and isinstance(payload.get("text"), str):
        meta["text"] = payload["text"]
    if "chunk_id" not in meta:
        meta["chunk_id"] = cid

    item = _ga_hit_from_payload(meta, full=True)
    item["zitierbar"] = False
    item.pop("citation", None)
    return item
_GA_TOKEN_RE = re.compile(r"^0*(\d+)([a-z]{0,2})$", re.IGNORECASE)


def _normalize_ga_token(raw: str) -> str | None:
    m = _GA_TOKEN_RE.match((raw or "").strip())
    if not m:
        return None
    return f"{m.group(1)}{(m.group(2) or '').lower()}"


def _parse_ga_list(ga: str | list[str] | None) -> tuple[list[str] | None, dict[str, Any] | None]:
    """Return (normalized_ga_list or None for all, error_dict or None)."""
    if ga is None or ga == "" or ga == []:
        return None, None
    raw_items: list[str]
    if isinstance(ga, str):
        raw_items = [p.strip() for p in ga.split(",") if p.strip()]
    else:
        raw_items = [str(p).strip() for p in ga if str(p).strip()]
    if not raw_items:
        return None, None
    out: list[str] = []
    for item in raw_items:
        norm = _normalize_ga_token(item)
        if norm is None:
            return None, {
                "error": f"invalid ga: {item!r}; expected e.g. '329' or '332a'",
                "code": "invalid_ga",
            }
        out.append(norm)
    return out, None


async def _known_catalog_gas() -> set[str]:
    """GA tokens that appear in the lecture catalog (may be empty if table missing)."""

    def _query() -> set[str]:
        with get_engine().connect() as conn:
            rows = conn.execute(
                text("SELECT DISTINCT ga FROM rag_lecture_catalog WHERE ga IS NOT NULL")
            ).fetchall()
        known: set[str] = set()
        for (raw,) in rows:
            if raw is None:
                continue
            for part in str(raw).split(","):
                norm = _normalize_ga_token(part.strip())
                if norm:
                    known.add(norm)
        return known

    return await asyncio.to_thread(_query)


async def _ga_present_in_qdrant(ga: str) -> bool:
    """True if rudolf-steiner-ga has at least one point with this payload.ga.

    Covers Schriften (GA 1–38) and Vorträge that are embedded but not yet in
    rag_lecture_catalog, so search_ga(..., ga=\"4\") is not rejected as unknown.
    """
    from app.core.providers import get_qdrant_client

    client = get_qdrant_client()
    points = await client.scroll_points(
        GA_QDRANT_COLLECTION,
        filter_={"must": [{"key": "ga", "match": {"value": ga}}]},
        limit=1,
        with_payload=False,
        with_vectors=False,
    )
    return bool(points)


async def _unknown_gas_after_lookup(gas: list[str]) -> list[str]:
    """GAs neither in lecture catalog nor present in Qdrant rudolf-steiner-ga."""
    known = await _known_catalog_gas()
    unknown = [g for g in gas if g not in known]
    if not unknown:
        return []
    still: list[str] = []
    for g in unknown:
        try:
            if await _ga_present_in_qdrant(g):
                continue
        except Exception:
            logger.exception("Qdrant GA existence check failed for %s", g)
            # Fail open for this token: let retrieval run (empty hits if truly absent).
            continue
        still.append(g)
    return still


def _ga_extra_must(
    gas: list[str] | None,
    lecture_id: str | None,
) -> list[dict[str, Any]]:
    must: list[dict[str, Any]] = []
    if gas:
        if len(gas) == 1:
            must.append({"key": "ga", "match": {"value": gas[0]}})
        else:
            must.append({"key": "ga", "match": {"any": gas}})
    lid = (lecture_id or "").strip()
    if lid:
        must.append({"key": "lecture_id", "match": {"value": lid}})
    return must


def _ga_hit_from_payload(payload: Mapping[str, Any], *, full: bool) -> dict[str, Any]:
    text_val = _mcp_text(str(payload.get("text") or ""))
    item: dict[str, Any] = {
        "chunk_id": payload.get("chunk_id"),
        "chunk_type": payload.get("chunk_type"),
        "text": text_val if full else _snippet(text_val, 500),
        "ga": payload.get("ga"),
        "lecture_id": payload.get("lecture_id"),
        "vortragstitel": payload.get("vortragstitel") or payload.get("segment_title"),
        "lecture_date": payload.get("lecture_date"),
        "venue": payload.get("venue"),
        "page_start": payload.get("page_start"),
        "page_end": payload.get("page_end"),
        "source_type": payload.get("source_type"),
        "segment_id": payload.get("segment_id"),
    }
    return {k: v for k, v in item.items() if v is not None and v != ""}


def _ga_rank_flags(
    variant: str,
    group_by_lecture: bool | None,
) -> dict[str, Any]:
    """Spec defaults for eval variants; ``configured`` follows RAGRUN_GA_SEARCH_*."""
    group = (
        settings.ga_search_group_by_lecture
        if group_by_lecture is None
        else group_by_lecture
    )
    if variant == "candidates":
        return {
            "candidates": 100,
            "pool": 60,
            "title_weight": 0.0,
            "use_title_path": False,
            "group_by_lecture": True if group_by_lecture is None else group_by_lecture,
            "max_per_ga": 3,
            "rerank": False,
        }
    if variant == "title":
        return {
            "candidates": 100,
            "pool": 60,
            "title_weight": 0.35,
            "use_title_path": True,
            "group_by_lecture": True if group_by_lecture is None else group_by_lecture,
            "max_per_ga": 3,
            "rerank": False,
        }
    if variant == "rerank":
        return {
            "candidates": 100,
            "pool": 60,
            "title_weight": 0.35,
            "use_title_path": True,
            "group_by_lecture": True if group_by_lecture is None else group_by_lecture,
            "max_per_ga": 3,
            "rerank": True,
        }
    return {
        "candidates": settings.ga_search_candidates,
        "pool": settings.ga_search_pool,
        "title_weight": settings.ga_search_title_weight,
        "use_title_path": True,
        "group_by_lecture": group,
        "max_per_ga": settings.ga_search_max_per_ga,
        "rerank": settings.ga_search_rerank,
    }


def _ga_results_from_ranked(ranked: list[Any], *, full: bool, explain: bool) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for hit in ranked:
        item = _ga_hit_from_payload(hit.meta, full=full)
        item["score"] = round(float(hit.score), 4)
        item["lecture_hits"] = int(hit.lecture_hits)
        if explain:
            item["score_parts"] = {
                "rrf": round(float(hit.rrf_norm), 4),
                "title": round(float(hit.title_score), 4),
                "rerank": None if hit.rerank is None else round(float(hit.rerank), 4),
            }
        out.append(item)
    return out


async def _search_ga_baseline(
    *,
    query: str,
    chunk_types: list[str],
    extra: list[dict[str, Any]],
    limit: int,
    full_text: bool,
) -> list[dict[str, Any]] | dict[str, Any]:
    """Previous hybrid path: k_dense = k_sparse = k_fused = limit, no title signal."""
    from app.core.providers import get_embedding_client, get_qdrant_client, get_sparse_embedder
    from app.retrieval.utils.retrievers import hybrid_retrieve

    prefix_query = settings.embedding_prefix_query or "query: "
    hits = await hybrid_retrieve(
        query=query,
        k_dense=limit,
        k_sparse=limit,
        k_fused=limit,
        worldview=None,
        book_types=chunk_types,
        collection=GA_QDRANT_COLLECTION,
        embedding_client=get_embedding_client(),
        qdrant_client=get_qdrant_client(),
        sparse_embedder=get_sparse_embedder() if settings.use_hybrid_retrieval else None,
        force_sparse=settings.use_hybrid_retrieval,
        query_prefix=prefix_query,
        extra_must=extra or None,
    )
    out: list[dict[str, Any]] = []
    for snip in hits:
        payload = snip.payload if isinstance(snip.payload, Mapping) else {}
        inner = payload.get("payload") if isinstance(payload.get("payload"), Mapping) else payload
        if not isinstance(inner, Mapping):
            inner = {}
        meta = dict(inner)
        if "text" not in meta and snip.text:
            meta["text"] = snip.text
        item = _ga_hit_from_payload(meta, full=full_text)
        item["score"] = round(float(snip.score), 4)
        out.append(item)
    return out


async def _search_ga_collection(
    *,
    query: str,
    chunk_types: list[str],
    ga: str | list[str] | None,
    lecture_id: str | None,
    limit: int,
    full_text: bool,
    group_by_lecture: bool | None = None,
    explain: bool = False,
    variant: str = "configured",
) -> list[dict[str, Any]] | dict[str, Any]:
    q = (query or "").strip()
    if not q:
        return {"error": "query is required", "code": "empty_query"}

    gas, err = _parse_ga_list(ga)
    if err:
        return err

    if gas:
        unknown = await _unknown_gas_after_lookup(gas)
        if unknown:
            return {
                "error": f"unknown ga: {', '.join(unknown)}",
                "code": "unknown_ga",
            }

    k = max(1, min(int(limit) if limit is not None else 8, 20))
    extra = _ga_extra_must(gas, lecture_id)

    from app.core.providers import get_embedding_client, get_qdrant_client, get_sparse_embedder

    prefix_query = settings.embedding_prefix_query or "query: "
    try:
        if variant == "baseline":
            return await _search_ga_baseline(
                query=q,
                chunk_types=chunk_types,
                extra=extra,
                limit=k,
                full_text=full_text,
            )
        flags = _ga_rank_flags(variant, group_by_lecture)
        from app.retrieval.utils.ga_ranking import ga_search_pipeline

        ranked = await ga_search_pipeline(
            query=q,
            chunk_types=chunk_types,
            extra_must=extra or None,
            collection=GA_QDRANT_COLLECTION,
            embedding_client=get_embedding_client(),
            qdrant_client=get_qdrant_client(),
            sparse_embedder=get_sparse_embedder() if settings.use_hybrid_retrieval else None,
            force_sparse=settings.use_hybrid_retrieval,
            query_prefix=prefix_query,
            limit=k,
            candidates=int(flags["candidates"]),
            pool=int(flags["pool"]),
            title_weight=float(flags["title_weight"]),
            use_title_path=bool(flags["use_title_path"]),
            group_by_lecture=bool(flags["group_by_lecture"]),
            max_per_ga=int(flags["max_per_ga"]),
            apply_ga_cap=not gas and not (lecture_id or "").strip(),
            gas=gas,
            rerank=bool(flags["rerank"]),
            rerank_top=int(settings.ga_search_rerank_top),
            rerank_timeout_s=float(settings.ga_search_rerank_timeout_s),
            rerank_url=settings.ga_search_rerank_url or "",
        )
    except Exception as exc:
        logger.exception("GA collection search failed")
        return {"error": f"search failed: {exc}", "code": "search_failed"}

    return _ga_results_from_ranked(ranked, full=full_text, explain=explain)


async def search_ga(
    query: str,
    ga: str | list[str] | None = None,
    lecture_id: str | None = None,
    limit: int = 8,
    group_by_lecture: bool | None = None,
    explain: bool = False,
) -> list[dict[str, Any]] | dict[str, Any]:
    """Search Rudolf Steiner GA wording in collection rudolf-steiner-ga.

    Only call after an explicit user request to search the GA / Gesamtausgabe.
    Prefer ``get_ga_chunk`` when a ``chunk_id`` is already known. Secondary to
    search_corpus. Hits are for understanding — do not pass long continuous
    passages to the end user. Short citations: (GA <nummer>, S. <seite>);
    omit the page part if it is missing.

    Broad topics are grouped by lecture and spread across GA volumes, so one
    volume does not fill the list. Pass ga or lecture_id to search inside one
    volume or one lecture; that turns the volume cap off. Each hit includes
    lecture_hits, the number of retrieved chunks from that lecture.

    Args:
        query: Search query (German or English).
        ga: One volume or list (e.g. "4", "329", "332a"). Empty = all volumes.
        lecture_id: Optional catalog id (YYYYMMDD[a-z]) to restrict to one lecture.
        limit: Max results (1-20, default 8).
        group_by_lecture: Group hits by lecture. Omit to use
            RAGRUN_GA_SEARCH_GROUP_BY_LECTURE (default true).
        explain: Include score_parts (rrf, title, rerank) on each hit.
    """

    return await _search_ga_collection(
        query=query,
        chunk_types=["ga_text"],
        ga=ga,
        lecture_id=lecture_id,
        limit=limit,
        full_text=True,
        group_by_lecture=group_by_lecture,
        explain=explain,
    )


async def search_ga_summaries(
    query: str,
    ga: str | list[str] | None = None,
    lecture_id: str | None = None,
    limit: int = 5,
    group_by_lecture: bool | None = None,
    explain: bool = False,
) -> list[dict[str, Any]] | dict[str, Any]:
    """Search chapter/lecture research summaries in collection rudolf-steiner-ga.

    Only call after an explicit user request to search the GA / Gesamtausgabe.
    Secondary to search_corpus. Prefer get_ga_chunk when a chunk_id is known.
    Same citation rules as search_ga; do not forward long continuous passages.

    Broad topics are grouped by lecture and spread across GA volumes. Pass ga or
    lecture_id to stay inside one volume or one lecture; that turns the volume
    cap off. Each hit includes lecture_hits, the number of retrieved chunks from
    that lecture.

    Args:
        query: Search query (German or English).
        ga: One volume or list (e.g. "4", "329", "332a"). Empty = all volumes.
        lecture_id: Optional catalog id (YYYYMMDD[a-z]).
        limit: Max results (1-20, default 5).
        group_by_lecture: Group hits by lecture. Omit to use
            RAGRUN_GA_SEARCH_GROUP_BY_LECTURE (default true).
        explain: Include score_parts (rrf, title, rerank) on each hit.
    """
    return await _search_ga_collection(
        query=query,
        chunk_types=["chapter_summary"],
        ga=ga,
        lecture_id=lecture_id,
        limit=limit,
        full_text=True,
        group_by_lecture=group_by_lecture,
        explain=explain,
    )
