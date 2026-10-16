"""Step 14b: Deep-link routes with HTML fallback pages.

/passage/{id} — public, shows passage text + "open in app" link
/text/{id}    — private, shows "open in app" link without content
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from sqlalchemy import text

from app.db.session import get_engine

logger = logging.getLogger(__name__)

router = APIRouter(tags=["deep-links"])

_CSS = """\
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       max-width: 640px; margin: 40px auto; padding: 0 20px; color: #1a1a1a;
       line-height: 1.6; }
.meta { color: #666; font-size: 0.9em; margin-bottom: 24px; }
.text { white-space: pre-wrap; margin: 24px 0; padding: 16px;
        background: #f8f7f4; border-left: 3px solid #8b7355; border-radius: 4px; }
.cta { display: inline-block; margin-top: 24px; padding: 12px 24px;
       background: #5b4a8a; color: white; text-decoration: none;
       border-radius: 8px; font-weight: 500; }
.cta:hover { background: #4a3a6e; }
.private { text-align: center; margin-top: 80px; }
"""


def _passage_html(
    paragraph_id: str,
    text_content: str,
    source_title: str | None,
    source_author: str | None,
    segment_title: str | None,
) -> str:
    import html

    title = html.escape(source_title or "Passage")
    author = html.escape(source_author or "")
    segment = html.escape(segment_title or "")
    body = html.escape(text_content)
    meta_parts = [x for x in [author, segment] if x]
    meta = " — ".join(meta_parts)

    return f"""\
<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{_CSS}</style>
</head>
<body>
<h1>{title}</h1>
<p class="meta">{meta}</p>
<div class="text">{body}</div>
<a class="cta" href="ragapp://passage/{html.escape(paragraph_id)}">In der App oeffnen</a>
</body>
</html>"""


def _text_fallback_html(note_id: str) -> str:
    import html

    nid = html.escape(note_id)
    return f"""\
<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Arbeitstext</title>
<style>{_CSS}</style>
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
<style>{_CSS}</style>
</head>
<body>
<div class="private">
<h1>Nicht gefunden</h1>
<p>Dieser {kind} existiert nicht oder wurde entfernt.</p>
</div>
</body>
</html>"""


@router.get("/passage/{paragraph_id}", response_class=HTMLResponse)
async def passage_fallback(paragraph_id: str) -> HTMLResponse:
    """Public fallback page for a passage deep link."""
    pid = paragraph_id.strip()
    if not pid:
        return HTMLResponse(_not_found_html("Absatz"), status_code=404)

    def _query() -> dict[str, Any] | None:
        with get_engine().connect() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT
                        p.id,
                        p.text_raw,
                        p.segment_title,
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
        return dict(row)

    result = await asyncio.to_thread(_query)
    if result is None:
        return HTMLResponse(_not_found_html("Absatz"), status_code=404)

    return HTMLResponse(
        _passage_html(
            paragraph_id=pid,
            text_content=result["text_raw"] or "",
            source_title=result["source_title"],
            source_author=result["source_author"],
            segment_title=result["segment_title"],
        )
    )


@router.get("/text/{note_id}", response_class=HTMLResponse)
async def text_fallback(note_id: str) -> HTMLResponse:
    """Private fallback page for a work text deep link (no content shown)."""
    nid = note_id.strip()
    if not nid:
        return HTMLResponse(_not_found_html("Text"), status_code=404)

    return HTMLResponse(_text_fallback_html(nid))
