#!/usr/bin/env python3
"""Refine truncation scan into a correction list with recovered page ranges."""

from __future__ import annotations

import csv
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_ga_lecture_markdown import (
    load_catalog,
    load_lectures,
    pdf_path,
    section_height,
    words_in,
    file_stem,
    folder_name,
    OUT_ROOT,
)

SCAN = Path("/Users/michael/Reniets/Ai/ragrun/ragkeep/ga/_audit/last-lecture-truncation-scan.tsv")
OUT = Path("/Users/michael/Reniets/Ai/ragrun/ragkeep/ga/_audit/last-lecture-truncation-korrekturlauf.tsv")
OUT_MD = Path("/Users/michael/Reniets/Ai/ragrun/ragkeep/ga/_audit/last-lecture-truncation-korrekturlauf.md")


def pdf_page_count(pdf: Path) -> int:
    info = subprocess.run(["pdfinfo", str(pdf)], check=True, capture_output=True, text=True)
    return int(re.search(r"^Pages:\s+(\d+)", info.stdout, re.M).group(1))


def read_page_range(pdf: Path, start: int, end: int) -> dict[int, list]:
    result = subprocess.run(
        ["pdftotext", "-bbox-layout", "-enc", "UTF-8", "-f", str(start), "-l", str(end), str(pdf), "-"],
        check=True,
        capture_output=True,
    )
    chunks = re.split(r"<page\b[^>]*>", result.stdout.decode("utf-8", errors="replace"))[1:]
    return {start + i: words_in(chunk) for i, chunk in enumerate(chunks)}


def recovered_end(pages: dict[int, list], start: int, pdf_last: int) -> tuple[int, float | None, int | None, float | None]:
    """End when height drops clearly (Hinweise); re-anchor off title page onto body."""
    lecture_height = None
    reanchored = False
    for page in range(start, pdf_last + 1):
        height = section_height(pages.get(page, []))
        if height is None:
            continue
        if lecture_height is None:
            lecture_height = height
            continue
        # Title page → body: first following page still body-sized (≥14pt) but
        # different height (often 1–5pt smaller, sometimes slightly larger).
        if (
            not reanchored
            and page <= start + 2
            and height >= 14.0
            and abs(height - lecture_height) >= 0.5
            and abs(height - lecture_height) <= 6.0
        ):
            lecture_height = height
            reanchored = True
            continue
        reanchored = True
        if height <= lecture_height - 2.5 or height < 12.5:
            return page - 1, lecture_height, page, height
        if height >= lecture_height + 3.0:
            return page - 1, lecture_height, page, height
    return pdf_last, lecture_height, None, None


def main() -> None:
    rows = []
    with SCAN.open(encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#") or line.startswith("ga\t"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 13:
                continue
            rows.append(parts)

    entries = load_catalog()
    out_rows = []
    for parts in rows:
        ga, lecture_id, start_s, end_s, pages_s, break_at, h0, h_break, delta, md_words, md_markers, md_hyphen, reason = parts[:13]
        start = int(start_s)
        old_pages = int(pages_s) if pages_s.isdigit() else 0
        pdf = pdf_path(ga)
        if not pdf.is_file():
            continue
        total = pdf_page_count(pdf)
        probe_end = min(total, start + 79)
        pages = read_page_range(pdf, start, probe_end)
        full = {i: pages.get(i, []) for i in range(start, total + 1)}
        new_end, h_body, br, br_h = recovered_end(full, start, total)
        new_pages = new_end - start + 1 if new_end >= start else 0
        gained = new_pages - old_pages
        lectures = load_lectures(entries, ga)
        lec = next((x for x in lectures if x["id"] == lecture_id), None)
        title = lec["titel"] if lec else ""
        md = OUT_ROOT / folder_name(ga) / f"{file_stem(lec)}.md" if lec else None
        md_w = int(md_words) if md_words.isdigit() else 0

        # Priority
        classic = old_pages <= 2 and (
            "false_font_break" in reason or "md_one_page" in reason or "md_mid_hyphen" in reason
        )
        if classic and gained >= 8:
            prio = "P1"
        elif old_pages <= 5 and gained >= 8:
            prio = "P1"
        elif gained >= 10:
            prio = "P2"
        elif gained >= 3 or (classic and gained >= 3):
            prio = "P3"
        elif old_pages <= 5 and md_w < 800:
            prio = "P3"
        else:
            prio = "P4"  # short range but little gain — maybe genuinely short / notes

        out_rows.append(
            {
                "prio": prio,
                "ga": ga,
                "lecture_id": lecture_id,
                "titel": title,
                "start": start,
                "end_alt": int(end_s) if end_s.isdigit() else "",
                "pages_alt": old_pages,
                "end_neu": new_end,
                "pages_neu": new_pages,
                "gewonnen": gained,
                "break_alt": break_at,
                "h0": h0,
                "h_break_alt": h_break,
                "delta_alt": delta,
                "break_neu": br or "",
                "h_body": h_body if h_body is not None else "",
                "h_break_neu": br_h if br_h is not None else "",
                "md_words": md_w,
                "md_markers": md_markers,
                "reason": reason,
                "md": str(md) if md else "",
            }
        )
        print(
            f"{prio} GA {ga} {lecture_id}: {old_pages}→{new_pages} pages (+{gained}) "
            f"S.{start}–{new_end} md={md_w}w",
            flush=True,
        )

    out_rows.sort(key=lambda r: (r["prio"], int(re.match(r"\d+", r["ga"]).group(0)), r["ga"], r["lecture_id"]))

    fields = list(out_rows[0].keys()) if out_rows else []
    with OUT.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        w.writeheader()
        w.writerows(out_rows)

    # Markdown summary
    lines = [
        "# Korrekturlauf: abgeschnittene letzte Vorträge (Schrift-Enderkennung)",
        "",
        "Ursache: `export_ga_lecture_markdown.py` setzt beim **letzten** Vortrag eines Bandes",
        "die Körperhöhe der Titelseite als Anker. Ist die Folgeseite 1–2 pt kleiner",
        f"(Acrobat-OCR), endet der Vortrag nach 1 Seite — wie bei GA 79.",
        "",
        "`end_neu` / `pages_neu`: geschätzter Bereich mit toleranterer Enderkennung",
        "(Re-Anker Titelseite→Körper, Abbruch erst bei klarerem Schriftwechsel).",
        "",
        f"Scan: `{SCAN.name}` · Liste: `{OUT.name}`",
        "",
    ]
    for prio in ("P1", "P2", "P3", "P4"):
        group = [r for r in out_rows if r["prio"] == prio]
        if not group:
            continue
        label = {
            "P1": "P1 — klarer Fehlabbruch (wenige Seiten, großer Gewinn)",
            "P2": "P2 — deutlicher Gewinn (≥10 Seiten)",
            "P3": "P3 — kleinerer Gewinn oder sehr kurze MD",
            "P4": "P4 — kurzer Bereich, wenig Gewinn (prüfen: evtl. echt kurz)",
        }[prio]
        lines.append(f"## {label} ({len(group)})")
        lines.append("")
        lines.append("| GA | id | Start–Ende (alt→neu) | Seiten | MD-Wörter | Titel |")
        lines.append("|---:|---|---|---:|---:|---|")
        for r in group:
            lines.append(
                f"| {r['ga']} | `{r['lecture_id']}` | "
                f"S.{r['start']}–{r['end_alt']} → S.{r['start']}–{r['end_neu']} | "
                f"{r['pages_alt']}→{r['pages_neu']} (+{r['gewonnen']}) | "
                f"{r['md_words']} | {r['titel'][:60]} |"
            )
        lines.append("")

    # Shell loop helper
    p1p2 = [r for r in out_rows if r["prio"] in ("P1", "P2")]
    lines.extend(
        [
            "## Re-Export (nach Fix der Enderkennung)",
            "",
            "```bash",
            "cd /Users/michael/Reniets/Ai/ragrun",
            "for ga in " + " ".join(r["ga"] for r in p1p2) + "; do",
            '  .venv/bin/python -u scripts/export_ga_lecture_markdown.py --ga "$ga"',
            "done",
            "```",
            "",
            f"P1+P2 Bände: {len(p1p2)} · alle geflaggt: {len(out_rows)}",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT}")
    print(f"Wrote {OUT_MD}")
    print(f"P1={sum(1 for r in out_rows if r['prio']=='P1')} "
          f"P2={sum(1 for r in out_rows if r['prio']=='P2')} "
          f"P3={sum(1 for r in out_rows if r['prio']=='P3')} "
          f"P4={sum(1 for r in out_rows if r['prio']=='P4')}")


if __name__ == "__main__":
    main()
