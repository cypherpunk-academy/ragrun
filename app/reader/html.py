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
    "ital,wght@0,400;0,500;1,400&family=Marcellus&display=swap"
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


def _unique_parts(parts: list[str]) -> list[str]:
    seen: list[str] = []
    for part in parts:
        text = (part or "").strip()
        if text and text not in seen:
            seen.append(text)
    return seen


def _header_parts(
    *,
    source_type: str,
    author: str,
    source_title: str,
    segment: str,
    lecture_ort: str,
    lecture_datum: str,
    lecture_title: str,
    lecture_reihe: str,
) -> tuple[list[str], list[str]]:
    """Leading pieces stay fully visible. The tail ellipsizes when it is too long."""
    if source_type == "lecture":
        place = ", ".join(part for part in (lecture_ort, lecture_datum) if part)
        title = lecture_title or segment or source_title
        leading = _unique_parts([author, place])
        tail = [
            part for part in _unique_parts([lecture_reihe, title]) if part not in leading
        ]
    else:
        parts = _unique_parts([author, source_title, segment])
        leading, tail = (parts[:-1], parts[-1:]) if parts else ([], [])
    if not leading and not tail:
        return [], ["Kapitel"]
    return leading, tail


def _header_arrow(nav: Mapping[str, str] | None, *, direction: str) -> str:
    """Icon-only chapter control. A missing sibling keeps its slot so the bar stays stable."""
    glyph = "←" if direction == "prev" else "→"
    label = "Voriges Kapitel" if direction == "prev" else "Nächstes Kapitel"
    if not nav or not nav.get("href"):
        return f'<span class="book-header-arrow is-absent" aria-hidden="true">{glyph}</span>'
    title = (nav.get("title") or label).strip()
    return (
        f'<a class="book-header-arrow" href="{html.escape(str(nav["href"]), quote=True)}"'
        f' aria-label="{html.escape(label)}"'
        f' title="{html.escape(title)}">{glyph}</a>'
    )


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
    source_type: str | None = None,
    lecture_ort: str | None = None,
    lecture_datum: str | None = None,
    lecture_title: str | None = None,
    lecture_reihe: str | None = None,
) -> str:
    paragraphs = list(paragraphs)
    author = (source_author or "").strip()
    segment = (segment_title or "").strip()
    target = (target_paragraph_id or "").strip()
    kind = (source_type or "book").strip().lower()
    leading, tail = _header_parts(
        source_type=kind,
        author=author,
        source_title=(source_title or "").strip(),
        segment=segment,
        lecture_ort=(lecture_ort or "").strip(),
        lecture_datum=(lecture_datum or "").strip(),
        lecture_title=(lecture_title or "").strip(),
        lecture_reihe=(lecture_reihe or "").strip(),
    )
    visible = [*leading, *tail]
    esc_title = html.escape(" · ".join(visible))
    sep = '<span class="book-header-sep">·</span>'
    pieces: list[str] = [
        f'<span class="book-header-keep">{html.escape(part)}</span>' for part in leading
    ]
    if tail:
        tail_html = sep.join(html.escape(part) for part in tail)
        pieces.append(f'<span class="book-header-tail">{tail_html}</span>')
    header_spans = sep.join(pieces)

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

    show_chapter_nav = bool(
        (prev_nav and prev_nav.get("href")) or (next_nav and next_nav.get("href"))
    )
    prev_arrow = _header_arrow(prev_nav, direction="prev") if show_chapter_nav else ""
    next_arrow = _header_arrow(next_nav, direction="next") if show_chapter_nav else ""

    passage_attr = ""
    if target:
        passage_attr = f' data-passage-id="{html.escape(target, quote=True)}"'

    return f"""\
<!DOCTYPE html>
<html lang="de" data-theme="default" class="book-page"{passage_attr}>
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{esc_title}</title>
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="stylesheet" href="{_FONT_HREF}" />
{css_tag}
</head>
<body class="book-body">
<header class="book-header">
  {prev_arrow}
  <p class="book-header-line">{header_spans}</p>
  <div class="book-header-actions">
    <button id="themeToggle" class="reader-toggle reader-toggle-theme" type="button" aria-label="Theme umschalten" title="Theme umschalten">🌙</button>
    <button id="sizeToggle" class="reader-toggle reader-toggle-size" type="button" aria-label="Schriftgröße umschalten" title="Schriftgröße umschalten">A</button>
  </div>
  {next_arrow}
</header>
<main class="book-main">
  {chr(10).join(body_parts)}
</main>
{js_tag}
</body>
</html>
"""
