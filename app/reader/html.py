"""Render rag_paragraphs chapters to HTML (annotations, target highlight)."""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from app.reader.urls import passage_link

_ASSETS_DIR = Path(__file__).resolve().parent / "assets"
_FONT_HREF = (
    "https://fonts.googleapis.com/css2?family=Cormorant+Garamond:"
    "ital,wght@0,400;0,500;1,400&display=swap"
)


def load_book_css() -> str:
    return (_ASSETS_DIR / "book.css").read_text(encoding="utf-8")


def load_reader_js() -> str:
    return (_ASSETS_DIR / "reader.js").read_text(encoding="utf-8")


def _as_annotations(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return {}
    if isinstance(raw, Mapping):
        return dict(raw)
    return {}


def _range_list(annotations: Mapping[str, Any], key: str) -> list[dict[str, Any]]:
    items = annotations.get(key) or []
    if not isinstance(items, list):
        return []
    out: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        try:
            start = int(item["start"])
            end = int(item["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end <= start:
            continue
        row = dict(item)
        row["start"] = start
        row["end"] = end
        out.append(row)
    return out


def annotate_text_html(
    text: str,
    annotations: Any,
    *,
    passage_base: str | None = None,
) -> str:
    """Apply italics, foreign quotes, and page_refs onto escaped HTML."""
    raw = text or ""
    ann = _as_annotations(annotations)
    n = len(raw)
    if n == 0:
        return ""

    marks: list[tuple[int, str, str]] = []  # (pos, kind open/close, html)
    for item in _range_list(ann, "italics"):
        marks.append((max(0, min(item["start"], n)), "open", "<i>"))
        marks.append((max(0, min(item["end"], n)), "close", "</i>"))
    for item in _range_list(ann, "foreign_quotes"):
        marks.append((max(0, min(item["start"], n)), "open", "<q>«"))
        marks.append((max(0, min(item["end"], n)), "close", "»</q>"))
    for item in _range_list(ann, "page_refs"):
        target = str(item.get("target_paragraph_id") or "").strip()
        href = passage_link(target) if target else ""
        if passage_base and target and not href.startswith("http"):
            href = f"{passage_base.rstrip('/')}/passage/{target}"
        if not href:
            continue
        safe = html.escape(href, quote=True)
        marks.append(
            (max(0, min(item["start"], n)), "open", f'<a class="page-ref" href="{safe}">')
        )
        marks.append((max(0, min(item["end"], n)), "close", "</a>"))

    # Open tags before text at the same offset; close tags after.
    def _sort_key(m: tuple[int, str, str]) -> tuple[int, int]:
        pos, kind, _ = m
        return (pos, 0 if kind == "close" else 1)

    marks.sort(key=_sort_key)
    out: list[str] = []
    cursor = 0
    for pos, _kind, tag in marks:
        if pos > cursor:
            out.append(html.escape(raw[cursor:pos]))
            cursor = pos
        elif pos < cursor:
            continue
        out.append(tag)
    if cursor < n:
        out.append(html.escape(raw[cursor:]))
    return "".join(out)


def render_chapter_html(
    *,
    source_title: str,
    source_author: str | None,
    segment_title: str,
    paragraphs: Iterable[Mapping[str, Any]],
    target_paragraph_id: str | None = None,
    prev_nav: Mapping[str, str] | None = None,
    next_nav: Mapping[str, str] | None = None,
    inline_assets: bool = False,
    app_passage_id: str | None = None,
) -> str:
    paragraphs = list(paragraphs)
    title = source_title or segment_title or "Kapitel"
    author = (source_author or "").strip()
    segment = (segment_title or "").strip()
    target = (target_paragraph_id or "").strip()
    esc_title = html.escape(title)
    meta_bits = [html.escape(x) for x in (author, segment) if x]
    meta = " — ".join(meta_bits)

    if inline_assets:
        css_tag = f"<style>\n{load_book_css()}\n</style>"
        js_tag = f"<script>\n{load_reader_js()}\n</script>"
    else:
        css_tag = '<link rel="stylesheet" href="/assets/book.css" />'
        js_tag = '<script src="/assets/reader.js" defer></script>'

    body_parts: list[str] = []
    for para in paragraphs:
        pid = str(para.get("id") or "").strip()
        num = para.get("paragraph_number")
        classes = "book-paragraph"
        if target and pid == target:
            classes += " is-target"
        pid_attr = html.escape(pid, quote=True)
        body_html = annotate_text_html(
            str(para.get("text_raw") or ""),
            para.get("annotations"),
        )
        num_html = ""
        if isinstance(num, int) and num > 0 and pid:
            num_html = (
                f'<a class="p-num" href="#{pid_attr}">{html.escape(str(num))}|</a>'
            )
        body_parts.append(
            f'<section class="{classes}" id="{pid_attr}">\n'
            f'  <p class="book-paragraph-text">{num_html}{body_html}</p>\n'
            f"</section>"
        )

    nav_html = ""
    nav_bits: list[str] = []
    if prev_nav and prev_nav.get("href"):
        nav_bits.append(
            f'<a class="nav-btn" href="{html.escape(prev_nav["href"], quote=True)}">'
            f"← {html.escape(prev_nav.get('title') or 'Zurück')}</a>"
        )
    else:
        nav_bits.append("<span></span>")
    if next_nav and next_nav.get("href"):
        nav_bits.append(
            f'<a class="nav-btn" href="{html.escape(next_nav["href"], quote=True)}">'
            f"{html.escape(next_nav.get('title') or 'Weiter')} →</a>"
        )
    if prev_nav or next_nav:
        nav_html = '<nav class="chapter-nav">\n  ' + "\n  ".join(nav_bits) + "\n</nav>"

    cta = ""
    if app_passage_id:
        cta = (
            f'<a class="app-cta" href="ragapp://passage/'
            f'{html.escape(app_passage_id)}">In der App öffnen</a>'
        )

    return f"""\
<!DOCTYPE html>
<html lang="de" data-theme="default" class="book-page">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{esc_title}</title>
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="stylesheet" href="{_FONT_HREF}" />
{css_tag}
</head>
<body class="book-body">
<button id="themeToggle" class="reader-toggle reader-toggle-theme" type="button" aria-label="Theme umschalten" title="Theme umschalten">🌙</button>
<button id="sizeToggle" class="reader-toggle reader-toggle-size" type="button" aria-label="Schriftgröße umschalten" title="Schriftgröße umschalten">A</button>
<main class="book-main">
  <h1 class="book-chapter-title">{html.escape(segment or title)}</h1>
  <p class="book-meta">{meta}</p>
  {chr(10).join(body_parts)}
  {nav_html}
  {cta}
</main>
{js_tag}
</body>
</html>
"""
