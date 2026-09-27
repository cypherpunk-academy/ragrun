"""MCP tools for Filo x Claude integration (Steps 11 + 13)."""
from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

import httpx
from sqlalchemy import text

from app.config import settings
from app.db.session import get_engine
from app.services.app_catalog_repository import PostgresCatalogRepository
from app.services.app_search_service import app_search

logger = logging.getLogger(__name__)

_SNIPPET_LIMIT = 300


def _snippet(t: str, limit: int = _SNIPPET_LIMIT) -> str:
    s = (t or "").strip().replace("\n", " ")
    if len(s) <= limit:
        return s
    return s[: limit - 1].rstrip() + "…"


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


async def search_corpus(
    query: str,
    types: list[str] | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Search the philosophical corpus (books, talks, concepts, quotes).

    Returns ranked results with chunk_id, source metadata, and a short snippet.
    Use get_passage to read the full paragraph text for a result.

    Args:
        query: Search query (German or English).
        types: Filter by type: "text", "concept", "quote", "chapter_summary". Default: all.
        limit: Max results (1-20, default 10).
    """
    k = max(1, min(limit or 10, 20))
    results = await app_search(
        query=query,
        types=types,
        limit=k,
        engine=get_engine(),
    )
    out: list[dict[str, Any]] = []
    for r in results:
        item: dict[str, Any] = {
            "chunk_id": r.get("chunk_id"),
            "chunk_type": r.get("chunk_type"),
            "score": round(float(r.get("score", 0)), 4),
            "snippet": _snippet(r.get("snippet") or r.get("text") or ""),
        }
        if r.get("title"):
            item["title"] = r["title"]
        if r.get("segment_title"):
            item["segment_title"] = r["segment_title"]
        if r.get("author"):
            item["author"] = r["author"]
        if r.get("paragraph_id"):
            item["paragraph_id"] = r["paragraph_id"]
        if r.get("source_id"):
            item["source_id"] = r["source_id"]
        out.append(item)
    return out


# ---------------------------------------------------------------------------
# get_passage
# ---------------------------------------------------------------------------


async def get_passage(paragraph_id: str) -> dict[str, Any]:
    """Read a single paragraph by its UUID.

    Returns the paragraph text plus source/segment metadata for navigation.

    Args:
        paragraph_id: UUID of the paragraph (from search results or protocols).
    """
    pid = (paragraph_id or "").strip()
    if not pid:
        return {"error": "paragraph_id is required"}

    def _query() -> dict[str, Any] | None:
        with get_engine().connect() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT
                        p.id AS paragraph_id,
                        p.source_id,
                        p.segment_index,
                        p.segment_title,
                        p.paragraph_number,
                        p.text_raw AS text,
                        s.title AS source_title,
                        s.author AS source_author
                    FROM rag_paragraphs p
                    LEFT JOIN rag_sources s ON s.id::text = p.source_id
                    WHERE p.id = :pid
                      AND p.deprecated_at IS NULL
                    """
                ),
                {"pid": pid},
            ).mappings().first()
        if not row:
            return None
        return {
            "paragraph_id": row["paragraph_id"],
            "source_id": row["source_id"],
            "source_title": row["source_title"],
            "source_author": row["source_author"],
            "segment_index": row["segment_index"],
            "segment_title": row["segment_title"],
            "paragraph_number": row["paragraph_number"],
            "text": str(row["text"] or ""),
        }

    result = await asyncio.to_thread(_query)
    if result is None:
        return {"error": "paragraph not found"}
    return result


# ---------------------------------------------------------------------------
# list_volumes
# ---------------------------------------------------------------------------


async def list_volumes() -> list[dict[str, Any]]:
    """List all available books/volumes in the corpus.

    Returns source_id, title, author for each volume.
    Use source_id with get_passage or search_corpus for navigation.
    """
    catalog = PostgresCatalogRepository(get_engine())
    sources = await catalog.list_sources()
    return [
        {
            "source_id": s["source_id"],
            "display_name": s["display_name"],
        }
        for s in sources
    ]


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
    slug = (segment_slug or "").strip()
    if not sid or not slug:
        return {"error": "source_id and segment_slug are required"}

    def _query() -> dict[str, Any] | None:
        with get_engine().connect() as conn:
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

    k = max(1, min(limit or 20, 50))

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
        return [
            {
                "note_id": row["id"],
                "title": row["title"],
                "text_type": row["text_type"],
                "status": row["status"],
                "version": row["version"],
                "paragraph_id": str(row["paragraph_id"]) if row["paragraph_id"] else None,
                "created_by": row["created_by"],
                "updated_at": str(row["updated_at"]),
            }
            for row in rows
        ]

    return await asyncio.to_thread(_query)


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
        return {
            "note_id": row["id"],
            "title": row["title"],
            "content": row["content"],
            "text_type": row["text_type"],
            "status": row["status"],
            "version": row["version"],
            "paragraph_id": str(row["paragraph_id"]) if row["paragraph_id"] else None,
            "conversation_url": row["conversation_url"],
            "created_by": row["created_by"],
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
        }

    result = await asyncio.to_thread(_query)
    if result is None:
        return {"error": "note not found"}
    return result


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
    slug = (segment_slug or "").strip()
    etype = (entry_type or "").strip()
    if not sid or not slug or not etype:
        return {"error": "source_id, segment_slug, and entry_type are required"}

    def _write() -> dict[str, Any]:
        with get_engine().connect() as conn:
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

    if result.get("is_expired"):
        out: dict[str, Any] = {
            "expired": True,
            "message": "Diese Uebergabe ist abgelaufen. Du kannst den Absatz aber noch direkt lesen.",
        }
        if result.get("paragraph_id"):
            out["paragraph_id"] = str(result["paragraph_id"])
        if result.get("user_question"):
            out["user_question"] = result["user_question"]
        return out

    return {
        "handoff_id": result["id"],
        "paragraph_id": str(result["paragraph_id"]) if result["paragraph_id"] else None,
        "source_id": result["source_id"],
        "segment_slug": result["segment_slug"],
        "marked_text": result["marked_text"],
        "user_question": result["user_question"],
        "return_url": result["return_url"],
        "created_at": str(result["created_at"]),
        "expires_at": str(result["expires_at"]),
    }
