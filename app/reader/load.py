"""Load a full chapter/lecture around a paragraph for the corpus reader."""
from __future__ import annotations

from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine


_PARA_SELECT = """
    SELECT
        p.id::text AS id,
        p.source_id,
        p.segment_slug,
        p.segment_title,
        p.segment_index,
        p.paragraph_number,
        p.text_raw,
        p.annotations,
        s.title AS source_title,
        s.author AS source_author
    FROM rag_paragraphs p
    LEFT JOIN rag_sources s ON s.id::text = p.source_id
    WHERE p.deprecated_at IS NULL
"""


def load_chapter_for_paragraph(
    engine: Engine,
    paragraph_id: str,
) -> dict[str, Any] | None:
    pid = (paragraph_id or "").strip()
    if not pid:
        return None
    with engine.connect() as conn:
        focus = conn.execute(
            text(_PARA_SELECT + " AND p.id::text = :pid"),
            {"pid": pid},
        ).mappings().first()
        if not focus:
            return None
        sid = str(focus["source_id"])
        slug = str(focus["segment_slug"] or "")
        rows = conn.execute(
            text(
                _PARA_SELECT
                + """
                  AND p.source_id = :sid
                  AND p.segment_slug = :slug
                ORDER BY p.paragraph_number ASC NULLS LAST, p.id
                """
            ),
            {"sid": sid, "slug": slug},
        ).mappings().all()
        siblings = conn.execute(
            text(
                """
                SELECT DISTINCT ON (segment_slug)
                    segment_slug,
                    segment_title,
                    segment_index,
                    id::text AS first_paragraph_id
                FROM rag_paragraphs
                WHERE source_id = :sid
                  AND deprecated_at IS NULL
                ORDER BY segment_slug, paragraph_number ASC NULLS LAST, id
                """
            ),
            {"sid": sid},
        ).mappings().all()

    paragraphs = [dict(r) for r in rows]
    sibs = sorted(
        [dict(r) for r in siblings],
        key=lambda s: (s.get("segment_index") is None, s.get("segment_index") or 0),
    )
    idx = next(
        (i for i, s in enumerate(sibs) if str(s.get("segment_slug")) == slug),
        None,
    )
    prev_nav = sibs[idx - 1] if idx is not None and idx > 0 else None
    next_nav = sibs[idx + 1] if idx is not None and idx < len(sibs) - 1 else None
    return {
        "focus": dict(focus),
        "paragraphs": paragraphs,
        "source_id": sid,
        "segment_slug": slug,
        "source_title": focus.get("source_title") or "",
        "source_author": focus.get("source_author") or "",
        "segment_title": focus.get("segment_title") or "",
        "prev": prev_nav,
        "next": next_nav,
    }


def list_segments(engine: Engine) -> list[dict[str, Any]]:
    """All active (source_id, segment_slug) pairs for export."""
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT DISTINCT ON (p.source_id, p.segment_slug)
                    p.source_id,
                    p.segment_slug,
                    p.segment_title,
                    p.segment_index,
                    s.title AS source_title,
                    s.author AS source_author,
                    p.id::text AS first_paragraph_id
                FROM rag_paragraphs p
                LEFT JOIN rag_sources s ON s.id::text = p.source_id
                WHERE p.deprecated_at IS NULL
                  AND p.segment_slug IS NOT NULL
                ORDER BY p.source_id, p.segment_slug, p.paragraph_number ASC NULLS LAST, p.id
                """
            )
        ).mappings().all()
    return [dict(r) for r in rows]
