"""Build uniform citation (Beleg) objects for MCP tool responses."""
from __future__ import annotations

import re
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.config import settings

_CITationRow = dict[str, Any]

_HEADING_RE = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)
_FIRST_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

_CONTENT_PREVIEW_CHARS = 2048


def passage_return_url(paragraph_id: str) -> str:
    pid = (paragraph_id or "").strip()
    if not pid:
        return ""
    base = (settings.mcp_base_url or "").strip().rstrip("/")
    if not base:
        return ""
    return f"{base}/passage/{pid}"


def text_return_url(note_id: str) -> str:
    nid = (note_id or "").strip()
    if not nid:
        return ""
    base = (settings.mcp_base_url or "").strip().rstrip("/")
    if not base:
        return ""
    return f"{base}/text/{nid}"


def format_zitierform(
    band: str,
    segment_title: str,
    paragraph_number: int | None,
) -> str:
    """Human-readable citation string for Philo corpus bands (no GA numbers)."""
    b = (band or "").strip() or "Unbekannte Quelle"
    seg = (segment_title or "").strip()
    num = int(paragraph_number) if paragraph_number is not None else 0
    if num > 0:
        if seg:
            return f"{b}, {seg}, Absatz {num}"
        return f"{b}, Absatz {num}"
    if seg:
        return f"{b}, {seg}"
    return b


def row_to_citation(row: _CITationRow, *, include_paragraph: bool = True) -> dict[str, Any]:
    source_type = str(row.get("source_type") or "book").strip().lower()
    segment_kind = "vortrag" if source_type == "lecture" else "kapitel"
    band = str(row.get("source_title") or "").strip() or "Unbekannte Quelle"
    segment_title = str(row.get("segment_title") or "").strip()
    paragraph_number = int(row["paragraph_number"]) if row.get("paragraph_number") is not None else 0
    pid = str(row.get("paragraph_id") or "").strip() if include_paragraph else ""

    ort = ""
    datum = ""
    if segment_kind == "vortrag":
        ort = str(row.get("lecture_ort") or "").strip()
        datum = str(row.get("lecture_datum") or "").strip()

    zitier = format_zitierform(
        band,
        segment_title,
        paragraph_number if include_paragraph else None,
    )
    return {
        "paragraph_id": pid,
        "band": band,
        "segment_kind": segment_kind,
        "segment_title": segment_title,
        "paragraph_number": paragraph_number if include_paragraph else 0,
        "ort": ort,
        "datum": datum,
        "zitierform": zitier,
        "return_url": passage_return_url(pid) if pid else "",
    }


def _first_markdown_heading(content: str | None) -> str | None:
    if not content:
        return None
    m = _HEADING_RE.search(content)
    if not m:
        return None
    h = m.group(1).strip()
    return h or None


def _first_sentence(content: str | None, max_len: int = 60) -> str | None:
    if not content:
        return None
    flat = " ".join(content.strip().split())
    if not flat:
        return None
    parts = _FIRST_SENTENCE_RE.split(flat, maxsplit=1)
    sentence = parts[0].strip()
    if len(sentence) <= max_len:
        return sentence
    return sentence[: max_len - 1].rstrip() + "…"


def note_display_title(
    title: str | None,
    content: str | None,
    citation: dict[str, Any] | None,
) -> tuple[str, str]:
    """Return (display_title, title_source) for work-text lists."""
    if title and str(title).strip():
        return str(title).strip(), "stored"
    heading = _first_markdown_heading(content)
    if heading:
        return heading, "heading"
    sentence = _first_sentence(content)
    if sentence:
        return sentence, "first_sentence"
    if citation and citation.get("zitierform"):
        return str(citation["zitierform"]), "citation"
    return "Arbeitstext ohne Titel", "fallback"


_CITation_SELECT = """
    SELECT
        p.id::text AS paragraph_id,
        p.source_id,
        p.segment_slug,
        p.segment_title,
        p.paragraph_number,
        COALESCE(s.title, '') AS source_title,
        COALESCE(s.source_type, 'book') AS source_type,
        lc.ort AS lecture_ort,
        lc.datum AS lecture_datum
    FROM rag_paragraphs p
    LEFT JOIN rag_sources s ON s.id::text = p.source_id
    LEFT JOIN rag_lecture_catalog lc ON lc.uuid::text = p.source_id
    WHERE p.deprecated_at IS NULL
"""


def fetch_citation(engine: Engine, paragraph_id: str) -> dict[str, Any] | None:
    pid = (paragraph_id or "").strip()
    if not pid:
        return None
    with engine.connect() as conn:
        row = conn.execute(
            text(_CITation_SELECT + " AND p.id = CAST(:pid AS uuid)"),
            {"pid": pid},
        ).mappings().first()
    if not row:
        return None
    return row_to_citation(dict(row))


def fetch_citations_batch(
    engine: Engine,
    paragraph_ids: list[str],
) -> dict[str, dict[str, Any]]:
    ids = [p.strip() for p in paragraph_ids if p and str(p).strip()]
    if not ids:
        return {}
    with engine.connect() as conn:
        rows = conn.execute(
            text(_CITation_SELECT + " AND p.id::text = ANY(:ids)"),
            {"ids": ids},
        ).mappings().all()
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        pid = str(row["paragraph_id"])
        out[pid] = row_to_citation(dict(row))
    return out


def resolve_segment_slug(
    engine: Engine,
    source_id: str,
    segment_slug: str,
) -> str | None:
    """Map a segment index or alias to the canonical ``rag_paragraphs.segment_slug``.

    Accepts the stored slug, or a numeric segment index (e.g. ``\"15\"``).
    Returns None if neither resolves for the source.
    """
    sid = (source_id or "").strip()
    raw = (segment_slug or "").strip()
    if not sid or not raw:
        return None

    with engine.connect() as conn:
        by_slug = conn.execute(
            text(
                """
                SELECT segment_slug
                FROM rag_paragraphs
                WHERE source_id = :sid
                  AND segment_slug = :slug
                  AND deprecated_at IS NULL
                LIMIT 1
                """
            ),
            {"sid": sid, "slug": raw},
        ).mappings().first()
        if by_slug and by_slug["segment_slug"]:
            return str(by_slug["segment_slug"])

        if raw.isdigit():
            by_idx = conn.execute(
                text(
                    """
                    SELECT segment_slug
                    FROM rag_paragraphs
                    WHERE source_id = :sid
                      AND segment_index = :idx
                      AND deprecated_at IS NULL
                    ORDER BY paragraph_number ASC NULLS LAST
                    LIMIT 1
                    """
                ),
                {"sid": sid, "idx": int(raw)},
            ).mappings().first()
            if by_idx and by_idx["segment_slug"]:
                return str(by_idx["segment_slug"])

    return None


def fetch_chapter_citation(
    engine: Engine,
    source_id: str,
    segment_slug: str,
) -> dict[str, Any] | None:
    sid = (source_id or "").strip()
    raw_slug = (segment_slug or "").strip()
    if not sid or not raw_slug:
        return None
    slug = resolve_segment_slug(engine, sid, raw_slug) or raw_slug
    with engine.connect() as conn:
        row = conn.execute(
            text(
                _CITation_SELECT
                + """
                  AND p.source_id = :sid
                  AND p.segment_slug = :slug
                ORDER BY p.paragraph_number ASC NULLS LAST
                LIMIT 1
                """
            ),
            {"sid": sid, "slug": slug},
        ).mappings().first()
        if row:
            return row_to_citation(dict(row), include_paragraph=False)

        band_row = conn.execute(
            text(
                """
                SELECT COALESCE(title, '') AS source_title,
                       COALESCE(source_type, 'book') AS source_type
                FROM rag_sources
                WHERE id::text = :sid
                """
            ),
            {"sid": sid},
        ).mappings().first()
        if not band_row:
            return None
        pseudo = {
            "paragraph_id": "",
            "segment_title": "",
            "paragraph_number": 0,
            "source_title": band_row["source_title"],
            "source_type": band_row["source_type"],
            "lecture_ort": "",
            "lecture_datum": "",
        }
        return row_to_citation(pseudo, include_paragraph=False)
