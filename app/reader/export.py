"""Export rag_paragraphs chapters to HTML files and optionally R2."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from sqlalchemy.engine import Engine

from app.reader.html import load_book_css, load_reader_js, render_chapter_html
from app.reader.load import list_segments, load_chapter_for_paragraph
from app.reader.r2 import r2_client, r2_configured, upload_bytes
from app.reader.urls import chapter_object_key

logger = logging.getLogger(__name__)


def _nav_for_export(
    row: dict[str, Any] | None,
    source_id: str,
) -> dict[str, str] | None:
    if not row:
        return None
    slug = str(row.get("segment_slug") or "")
    if not slug:
        return None
    title = str(row.get("segment_title") or "").strip() or "Kapitel"
    return {"href": f"/books/{source_id}/{slug}.html", "title": title}


def render_segment_html(engine: Engine, first_paragraph_id: str) -> str | None:
    chapter = load_chapter_for_paragraph(engine, first_paragraph_id)
    if chapter is None or not chapter.get("paragraphs"):
        return None
    sid = str(chapter["source_id"])
    return render_chapter_html(
        source_title=str(chapter.get("source_title") or ""),
        source_author=chapter.get("source_author"),
        source_type=str(chapter.get("source_type") or "book"),
        lecture_ort=str(chapter.get("lecture_ort") or ""),
        lecture_datum=str(chapter.get("lecture_datum") or ""),
        lecture_title=str(chapter.get("lecture_title") or ""),
        lecture_reihe=str(chapter.get("lecture_reihe") or ""),
        segment_title=str(chapter.get("segment_title") or ""),
        paragraphs=chapter["paragraphs"],
        target_paragraph_id=None,
        prev_nav=_nav_for_export(chapter.get("prev"), sid),
        next_nav=_nav_for_export(chapter.get("next"), sid),
        inline_assets=False,
        app_passage_id=None,
    )


def export_reader(
    engine: Engine,
    *,
    out_dir: Path | None = None,
    upload: bool = False,
    limit: int | None = None,
) -> dict[str, int]:
    segments = list_segments(engine)
    if limit is not None:
        segments = segments[: max(0, limit)]
    client = r2_client() if upload else None
    if upload and not r2_configured():
        raise RuntimeError(
            "R2 is not configured (RAGRUN_R2_ACCOUNT_ID, RAGRUN_R2_ACCESS_KEY_ID, "
            "RAGRUN_R2_SECRET_ACCESS_KEY, RAGRUN_R2_BUCKET)"
        )

    written = 0
    failed = 0
    if out_dir is not None:
        out_dir = Path(out_dir)
        (out_dir / "assets").mkdir(parents=True, exist_ok=True)
        (out_dir / "assets" / "book.css").write_text(load_book_css(), encoding="utf-8")
        (out_dir / "assets" / "reader.js").write_text(load_reader_js(), encoding="utf-8")

    if upload and client is not None:
        upload_bytes("assets/book.css", load_book_css().encode("utf-8"), "text/css", client=client)
        upload_bytes(
            "assets/reader.js",
            load_reader_js().encode("utf-8"),
            "text/javascript",
            client=client,
        )

    for seg in segments:
        html = render_segment_html(engine, str(seg["first_paragraph_id"]))
        if not html:
            failed += 1
            continue
        sid = str(seg["source_id"])
        slug = str(seg["segment_slug"])
        rel = f"books/{sid}/{slug}.html"
        if out_dir is not None:
            dest = out_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(html, encoding="utf-8")
        if upload and client is not None:
            key = chapter_object_key(sid, slug)
            upload_bytes(key, html.encode("utf-8"), "text/html; charset=utf-8", client=client)
        written += 1
        if written % 50 == 0:
            logger.info("export_reader: %d chapters", written)

    return {"segments": len(segments), "written": written, "failed": failed}
