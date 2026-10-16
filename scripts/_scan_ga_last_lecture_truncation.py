#!/usr/bin/env python3
"""Find last-lecture truncations from font-height end detection (GA 79 pattern).

Scans Acrobat PDFs in GA/2026. For each band's last lecture (by start page),
reproduces lecture_pages() end detection and flags short ranges.
Also reports existing markdown word counts when present.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_ga_lecture_markdown import (
    HEIGHT_TOLERANCE,
    OUT_ROOT,
    file_stem,
    folder_name,
    is_body_height,
    load_catalog,
    load_lectures,
    pdf_path,
    section_height,
    volumes_51_to_354,
    words_in,
)

# Lookahead pages after lecture start for end detection (Hinweise usually within this).
LOOKAHEAD = 80
# Flag if last lecture ends within this many pages of its start.
SHORT_PAGES = 5
# Soft flag: range ends but remaining PDF pages look like more body (not Hinweise).
MIN_REMAINING_AFTER = 3


def pdf_page_count(pdf: Path) -> int:
    info = subprocess.run(
        ["pdfinfo", str(pdf)], check=True, capture_output=True, text=True
    )
    match = re.search(r"^Pages:\s+(\d+)", info.stdout, re.M)
    if not match:
        raise RuntimeError(f"Pages missing: {pdf}")
    return int(match.group(1))


def read_page_range(pdf: Path, start: int, end: int) -> dict[int, list]:
    result = subprocess.run(
        [
            "pdftotext",
            "-bbox-layout",
            "-enc",
            "UTF-8",
            "-f",
            str(start),
            "-l",
            str(end),
            str(pdf),
            "-",
        ],
        check=True,
        capture_output=True,
    )
    text = result.stdout.decode("utf-8", errors="replace")
    chunks = re.split(r"<page\b[^>]*>", text)[1:]
    return {start + i: words_in(chunk) for i, chunk in enumerate(chunks)}


def detect_end(
    pages: dict[int, list], start: int, pdf_last: int
) -> tuple[int, float | None, int | None, float | None]:
    """Return (end_page, lecture_height, break_page, break_height)."""
    end = pdf_last
    lecture_height: float | None = None
    break_page: int | None = None
    break_height: float | None = None
    for page in range(start, pdf_last + 1):
        height = section_height(pages.get(page, []))
        if height is None:
            continue
        if lecture_height is None:
            lecture_height = height
            continue
        if not is_body_height(height, lecture_height):
            end = page - 1
            break_page = page
            break_height = height
            break
    return end, lecture_height, break_page, break_height


def md_stats(ga: str, lecture: dict) -> tuple[Path | None, int, bool, bool]:
    md = OUT_ROOT / folder_name(ga) / f"{file_stem(lecture)}.md"
    if not md.is_file():
        return None, 0, False, False
    text = md.read_text(encoding="utf-8")
    words = len(text.split())
    has_marker = "[<- S." in text
    ends_hyphen = bool(re.search(r"[-\u00ad]\s*$", text.rstrip()))
    return md, words, has_marker, ends_hyphen


def main() -> None:
    entries = load_catalog()
    bands = volumes_51_to_354(entries)
    print(
        f"# Scan last-lecture truncation (tolerance={HEIGHT_TOLERANCE}, short≤{SHORT_PAGES} pages)",
        flush=True,
    )
    print(f"# bands={len(bands)}", flush=True)
    print(
        "ga\tlecture_id\tstart\tend\tpages\tbreak_at\th0\th_break\tdelta\tmd_words\tmd_markers\tmd_hyphen\treason",
        flush=True,
    )

    flagged = 0
    skipped = 0
    for n, ga in enumerate(bands, start=1):
        pdf = pdf_path(ga)
        if not pdf.is_file():
            skipped += 1
            continue
        lectures = load_lectures(entries, ga)
        if not lectures:
            skipped += 1
            continue
        last = lectures[-1]
        start = last["page"]
        try:
            total = pdf_page_count(pdf)
        except Exception as exc:
            print(f"{ga}\t?\t?\t?\t?\t?\t?\t?\t?\t?\t?\t?\tpdfinfo:{exc}", flush=True)
            continue
        if start > total:
            print(
                f"{ga}\t{last['id']}\t{start}\t?\t?\t?\t?\t?\t?\t?\t?\t?\tstart>{total}",
                flush=True,
            )
            continue

        end_probe = min(total, start + LOOKAHEAD - 1)
        try:
            pages = read_page_range(pdf, start, end_probe)
        except subprocess.CalledProcessError as exc:
            print(
                f"{ga}\t{last['id']}\t{start}\t?\t?\t?\t?\t?\t?\t?\t?\t?\tpdftotext:{exc.returncode}",
                flush=True,
            )
            continue

        # Pad to total so detect_end can use pdf_last; pages beyond probe are empty
        full = {i: pages.get(i, []) for i in range(start, total + 1)}
        end, h0, break_page, h_break = detect_end(full, start, total)
        n_pages = max(0, end - start + 1) if end >= start else 0
        delta = (
            abs(h_break - h0) if h0 is not None and h_break is not None else None
        )
        md_path, words, has_marker, ends_hyphen = md_stats(ga, last)

        reasons = []
        if n_pages <= SHORT_PAGES:
            reasons.append(f"short_range({n_pages})")
        if (
            break_page is not None
            and h0 is not None
            and h_break is not None
            and abs(h_break - h0) <= 2.0
            and h_break >= 14.0
            and n_pages <= 15
        ):
            # Small font delta into still-body-sized text → classic false positive
            reasons.append("false_font_break")
        if md_path and words < 800 and not has_marker:
            reasons.append("md_one_page")
        if md_path and ends_hyphen and words < 2000:
            reasons.append("md_mid_hyphen")
        if md_path and words < 500:
            reasons.append(f"md_tiny({words}w)")

        # Only report flagged or interesting
        if not reasons and n_pages > SHORT_PAGES:
            if n % 25 == 0:
                print(f"# progress {n}/{len(bands)} ok ga={ga}", flush=True)
            continue

        flagged += 1
        print(
            f"{ga}\t{last['id']}\t{start}\t{end}\t{n_pages}\t"
            f"{break_page or ''}\t{h0 if h0 is not None else ''}\t"
            f"{h_break if h_break is not None else ''}\t"
            f"{f'{delta:.2f}' if delta is not None else ''}\t"
            f"{words if md_path else ''}\t"
            f"{'yes' if has_marker else ('no' if md_path else '')}\t"
            f"{'yes' if ends_hyphen else ('no' if md_path else '')}\t"
            f"{','.join(reasons) or 'ok'}",
            flush=True,
        )

    print(f"# done flagged={flagged} skipped_no_pdf_or_lectures={skipped}", flush=True)


if __name__ == "__main__":
    main()
