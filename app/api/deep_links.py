"""Deep-link routes with HTML fallback (and 302 to the static corpus host).

/passage/{id} — public; Universal Link on api/staging. Browser: 302 to corpus.*
               or annotated chapter HTML from the DB when no corpus host is set.
/text/{id}    — private, shows "open in app" link without content
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, RedirectResponse

from app.config import settings
from app.db.session import get_engine
from app.reader.html import render_chapter_html
from app.reader.load import load_chapter_for_paragraph
from app.reader.urls import chapter_public_url, passage_link

logger = logging.getLogger(__name__)

router = APIRouter(tags=["deep-links"])

_CSS_FALLBACK = """\
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       max-width: 640px; margin: 40px auto; padding: 0 20px; color: #1a1a1a;
       line-height: 1.6; }
.private { text-align: center; margin-top: 80px; }
.cta { display: inline-block; margin-top: 24px; padding: 12px 24px;
       background: #5b4a8a; color: white; text-decoration: none;
       border-radius: 8px; font-weight: 500; }
"""


def _text_fallback_html(note_id: str) -> str:
    import html as html_mod

    nid = html_mod.escape(note_id)
    return f"""\
<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Arbeitstext</title>
<style>{_CSS_FALLBACK}</style>
</head>
<body>
<div class="private">
<h1>Arbeitstext</h1>
<p>Dieser Text ist privat und nur in der App sichtbar.</p>
<a class="cta" href="ragapp://text/{nid}">In der App oeffnen</a>
</div>
</body>
</html>"""


def _not_found_html(kind: str) -> str:
    return f"""\
<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Nicht gefunden</title>
<style>{_CSS_FALLBACK}</style>
</head>
<body>
<div class="private">
<h1>Nicht gefunden</h1>
<p>Dieser {kind} existiert nicht oder wurde entfernt.</p>
</div>
</body>
</html>"""


def _nav_href(row: dict[str, Any] | None, *, chapter_urls: bool, source_id: str) -> dict[str, str] | None:
    if not row:
        return None
    title = str(row.get("segment_title") or "").strip() or "Kapitel"
    if chapter_urls:
        slug = str(row.get("segment_slug") or "")
        href = f"/books/{source_id}/{slug}.html"
    else:
        first = str(row.get("first_paragraph_id") or "").strip()
        href = passage_link(first) if first else ""
    if not href:
        return None
    return {"href": href, "title": title}


def _render_fallback(chapter: dict[str, Any], paragraph_id: str) -> str:
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
        target_paragraph_id=paragraph_id,
        prev_nav=_nav_href(chapter.get("prev"), chapter_urls=False, source_id=sid),
        next_nav=_nav_href(chapter.get("next"), chapter_urls=False, source_id=sid),
        inline_assets=True,
        app_passage_id=paragraph_id,
    )


@router.get("/passage/{paragraph_id}", response_model=None)
async def passage_fallback(paragraph_id: str) -> HTMLResponse | RedirectResponse:
    """Public fallback (or 302 to corpus.*) for a passage deep link."""
    pid = paragraph_id.strip()
    if not pid:
        return HTMLResponse(_not_found_html("Absatz"), status_code=404)

    chapter = await asyncio.to_thread(
        load_chapter_for_paragraph, get_engine(), pid
    )
    if chapter is None:
        return HTMLResponse(_not_found_html("Absatz"), status_code=404)

    corpus_url = chapter_public_url(
        str(chapter["source_id"]),
        str(chapter["segment_slug"]),
        pid,
    )
    if corpus_url and (settings.app_env or "").lower() != "development":
        return RedirectResponse(url=corpus_url, status_code=302)

    return HTMLResponse(_render_fallback(chapter, pid))


@router.get("/text/{note_id}", response_class=HTMLResponse)
async def text_fallback(note_id: str) -> HTMLResponse:
    """Private fallback page for a work text deep link (no content shown)."""
    nid = note_id.strip()
    if not nid:
        return HTMLResponse(_not_found_html("Text"), status_code=404)

    return HTMLResponse(_text_fallback_html(nid))
