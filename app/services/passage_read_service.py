"""Read paragraphs for MCP get_passage (plan §4, E9)."""
from __future__ import annotations

from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.mcp_server.citation import fetch_citation

MAX_PARAGRAPHS = 12
MAX_CONTEXT = 5

_PARAGRAPH_SELECT = """
    SELECT
        p.id AS paragraph_id,
        p.source_id,
        p.segment_index,
        p.segment_slug,
        p.segment_title,
        p.paragraph_number,
        p.text_raw AS text,
        s.title AS source_title,
        s.author AS source_author,
        s.source_type AS source_type,
        s.ga AS ga
    FROM rag_paragraphs p
    LEFT JOIN rag_sources s ON s.id::text = p.source_id
    WHERE p.deprecated_at IS NULL
"""


def _row_to_item(row: Any) -> dict[str, Any]:
    return {
        "paragraph_id": str(row["paragraph_id"]),
        "source_id": row["source_id"],
        "source_title": row["source_title"],
        "source_author": row["source_author"],
        "segment_index": row["segment_index"],
        "segment_slug": row["segment_slug"],
        "segment_title": row["segment_title"],
        "paragraph_number": row["paragraph_number"],
        "text": str(row["text"] or ""),
    }


def _attach_citations(engine: Engine, items: list[dict[str, Any]]) -> None:
    for item in items:
        citation = fetch_citation(engine, item["paragraph_id"])
        if citation:
            item["citation"] = citation


def _fetch_one(engine: Engine, paragraph_id: str) -> dict[str, Any] | None:
    with engine.connect() as conn:
        row = conn.execute(
            text(_PARAGRAPH_SELECT + " AND p.id = :pid"),
            {"pid": paragraph_id},
        ).mappings().first()
    return _row_to_item(row) if row else None


def _fetch_range(
    engine: Engine,
    source_id: str,
    segment_index: int,
    from_num: int,
    to_num: int,
) -> list[dict[str, Any]]:
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                _PARAGRAPH_SELECT
                + """
                AND p.source_id = :sid
                AND p.segment_index = :seg
                AND p.paragraph_number BETWEEN :lo AND :hi
                ORDER BY p.paragraph_number ASC
                """
            ),
            {"sid": source_id, "seg": segment_index, "lo": from_num, "hi": to_num},
        ).mappings().all()
    return [_row_to_item(r) for r in rows]


def _segment_bounds(engine: Engine, source_id: str, segment_index: int) -> tuple[int, int]:
    with engine.connect() as conn:
        row = conn.execute(
            text(
                """
                SELECT MIN(paragraph_number) AS lo, MAX(paragraph_number) AS hi
                FROM rag_paragraphs
                WHERE deprecated_at IS NULL
                  AND source_id = :sid
                  AND segment_index = :seg
                """
            ),
            {"sid": source_id, "seg": segment_index},
        ).mappings().first()
    if not row or row["lo"] is None:
        return 0, 0
    return int(row["lo"]), int(row["hi"])


def _source_meta(engine: Engine, source_id: str) -> dict[str, Any] | None:
    with engine.connect() as conn:
        row = conn.execute(
            text(
                """
                SELECT ga, source_type
                FROM rag_sources
                WHERE id::text = :sid
                """
            ),
            {"sid": source_id},
        ).mappings().first()
    return dict(row) if row else None


def _neighbor_in_segment(
    engine: Engine,
    source_id: str,
    segment_index: int,
    paragraph_number: int,
    delta: int,
) -> dict[str, Any] | None:
    target = paragraph_number + delta
    lo, hi = _segment_bounds(engine, source_id, segment_index)
    if lo == 0 or target < lo or target > hi:
        return None
    with engine.connect() as conn:
        row = conn.execute(
            text(
                _PARAGRAPH_SELECT
                + """
                AND p.source_id = :sid
                AND p.segment_index = :seg
                AND p.paragraph_number = :num
                """
            ),
            {"sid": source_id, "seg": segment_index, "num": target},
        ).mappings().first()
    return _row_to_item(row) if row else None


def _lecture_uuid_order(engine: Engine, ga: str) -> list[str]:
    ga_key = ga.strip()
    if not ga_key:
        return []
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT l.uuid::text AS source_id
                FROM rag_lecture_catalog l
                WHERE lower(trim(l.ga)) = lower(:ga)
                   OR (',' || lower(replace(l.ga, ' ', '')) || ',')
                      LIKE '%,' || lower(:ga) || ',%'
                ORDER BY l.lecture_date ASC NULLS LAST, l.id ASC
                """
            ),
            {"ga": ga_key},
        ).mappings().all()
    return [str(r["source_id"]) for r in rows]


def _first_paragraph(engine: Engine, source_id: str) -> dict[str, Any] | None:
    with engine.connect() as conn:
        row = conn.execute(
            text(
                _PARAGRAPH_SELECT
                + """
                AND p.source_id = :sid
                ORDER BY p.segment_index ASC, p.paragraph_number ASC
                LIMIT 1
                """
            ),
            {"sid": source_id},
        ).mappings().first()
    return _row_to_item(row) if row else None


def _last_paragraph(engine: Engine, source_id: str) -> dict[str, Any] | None:
    with engine.connect() as conn:
        row = conn.execute(
            text(
                _PARAGRAPH_SELECT
                + """
                AND p.source_id = :sid
                ORDER BY p.segment_index DESC, p.paragraph_number DESC
                LIMIT 1
                """
            ),
            {"sid": source_id},
        ).mappings().first()
    return _row_to_item(row) if row else None


def _cross_lecture_neighbor(
    engine: Engine, source_id: str, ga: str, direction: int
) -> dict[str, Any] | None:
    order = _lecture_uuid_order(engine, ga)
    if source_id not in order:
        return None
    idx = order.index(source_id)
    next_idx = idx + direction
    if next_idx < 0 or next_idx >= len(order):
        return None
    neighbor_sid = order[next_idx]
    if direction > 0:
        return _first_paragraph(engine, neighbor_sid)
    return _last_paragraph(engine, neighbor_sid)


def _nav_pointer(item: dict[str, Any] | None) -> dict[str, Any] | None:
    if not item:
        return None
    return {
        "paragraph_id": item["paragraph_id"],
        "source_id": item["source_id"],
        "segment_index": item["segment_index"],
        "paragraph_number": item["paragraph_number"],
    }


def _compute_nav(engine: Engine, window: list[dict[str, Any]]) -> dict[str, Any]:
    if not window:
        return {"prev": None, "next": None}
    first = window[0]
    last = window[-1]
    sid = str(first["source_id"])
    seg = int(first["segment_index"])
    lo_first, _ = _segment_bounds(engine, sid, seg)

    prev_item = _neighbor_in_segment(
        engine, sid, seg, int(first["paragraph_number"]), -1
    )
    if prev_item is None and int(first["paragraph_number"]) == lo_first:
        meta = _source_meta(engine, sid)
        if meta and meta.get("ga"):
            prev_item = _cross_lecture_neighbor(engine, sid, str(meta["ga"]), -1)

    sid_last = str(last["source_id"])
    seg_last = int(last["segment_index"])
    _, hi_last = _segment_bounds(engine, sid_last, seg_last)

    next_item = _neighbor_in_segment(
        engine, sid_last, seg_last, int(last["paragraph_number"]), 1
    )
    if next_item is None and int(last["paragraph_number"]) == hi_last:
        meta = _source_meta(engine, sid_last)
        if meta and meta.get("ga"):
            next_item = _cross_lecture_neighbor(engine, sid_last, str(meta["ga"]), 1)

    return {"prev": _nav_pointer(prev_item), "next": _nav_pointer(next_item)}


def read_passages(
    engine: Engine,
    *,
    paragraph_id: str | None = None,
    source_id: str | None = None,
    segment_index: int | None = None,
    from_paragraph: int | None = None,
    to_paragraph: int | None = None,
    before: int = 0,
    after: int = 0,
) -> dict[str, Any]:
    before = max(0, min(before, MAX_CONTEXT))
    after = max(0, min(after, MAX_CONTEXT))

    if paragraph_id:
        anchor = _fetch_one(engine, paragraph_id.strip())
        if anchor is None:
            return {"error": "paragraph not found"}
        sid = str(anchor["source_id"])
        seg = int(anchor["segment_index"])
        num = int(anchor["paragraph_number"])
        lo, hi = _segment_bounds(engine, sid, seg)
        from_num = max(lo, num - before) if lo else num - before
        to_num = min(hi, num + after) if hi else num + after
        paragraphs = _fetch_range(engine, sid, seg, from_num, to_num)
    elif source_id and segment_index is not None and from_paragraph is not None:
        to_p = to_paragraph if to_paragraph is not None else from_paragraph
        if to_p < from_paragraph:
            to_p = from_paragraph
        if to_p - from_paragraph + 1 > MAX_PARAGRAPHS:
            to_p = from_paragraph + MAX_PARAGRAPHS - 1
        paragraphs = _fetch_range(engine, source_id, segment_index, from_paragraph, to_p)
        if not paragraphs:
            return {"error": "paragraph not found"}
        sid = source_id
        seg = segment_index
    else:
        return {"error": "paragraph_id or (source_id, segment_index, from_paragraph) required"}

    if len(paragraphs) > MAX_PARAGRAPHS:
        paragraphs = paragraphs[:MAX_PARAGRAPHS]

    _attach_citations(engine, paragraphs)
    seg_first, seg_last = _segment_bounds(engine, sid, seg)
    nav = _compute_nav(engine, paragraphs)

    payload: dict[str, Any] = {
        "paragraphs": paragraphs,
        "nav": nav,
        "segment_index": seg,
        "segment_first": seg_first or None,
        "segment_last": seg_last or None,
    }

    if len(paragraphs) == 1:
        payload.update(paragraphs[0])

    return payload
