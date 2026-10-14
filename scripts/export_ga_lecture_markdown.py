#!/usr/bin/env python3
"""Phase 3: Vortragstext eines GA-Bandes als Markdown.

Seitenzahlen kommen aus dem Katalog (Feld seite_pdf, sonst seite_ga).
Ein Testlauf schreibt zusätzlich die verworfenen und die knappen Blöcke.
"""

from __future__ import annotations

import argparse
import html
import re
import statistics
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import yaml

CATALOG = Path("/Users/michael/Reniets/Ai/steineroriginals/rudolf-steiner-ga-lecture-catalog.yaml")
PDF_DIR = Path("/Users/michael/GA 180dpi/GA-Acrobat/GA/2026")
OUT_ROOT = Path("/Users/michael/Reniets/Ai/ragrun/ragkeep/ga")

# Die Vortragsschrift wird je Band gemessen. In GA 329 ist sie 15.3 pt,
# in anderen Bänden 17 pt oder nur 10 pt. Abweichung bis zu diesem Wert gilt als dieselbe Schrift.
HEIGHT_TOLERANCE = 0.8
# Kleiner als der Vortragstext, aber zwischen zwei Textzeilen: mitnehmen.
NARROW_SPAN = 4.0
INDENT = 12.0
PARAGRAPH_GAP = 32.0
SOFT_HYPHEN = "\u00ad"

_WORD = re.compile(
    r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">(.*?)</word>',
    re.S,
)


@dataclass
class Line:
    page: int
    x0: float
    x1: float
    y: float
    height: float
    text: str
    kind: str  # text, verworfen, knapp
    continues: bool = False


@dataclass
class Report:
    rows: list[str] = field(default_factory=list)

    def add(self, page: int, kind: str, line: Line) -> None:
        label = "verworfen" if kind == "verworfen" else "knapp"
        self.rows.append(
            f"S.{page}\t{label}\th={line.height:.1f}\tx={line.x0:.0f}-{line.x1:.0f}\t{line.text}"
        )


def band_code(ga: str) -> str:
    match = re.fullmatch(r"(\d+)([a-zA-Z]?)", ga)
    if not match:
        raise SystemExit(f"GA-Nummer nicht lesbar: {ga}")
    return f"{int(match.group(1)):03d}{match.group(2)}"


def folder_name(ga: str) -> str:
    """ga_329, ga_051, ga_332a."""
    return f"ga_{band_code(ga)}"


def pdf_path(ga: str) -> Path:
    """Dateiname wie im Ordner GA/2026: dreistellig, Buchstabe direkt dahinter."""
    return PDF_DIR / f"GA {band_code(ga)}.pdf"


def page_of(entry: dict, ga: str) -> int | None:
    raw = entry.get("seite_pdf", entry.get("seite_ga"))
    if raw is None:
        return None
    if isinstance(raw, dict):
        if ga not in raw and int(re.match(r"\d+", ga).group(0)) not in raw:
            # Schlüssel können 88 oder "71a" sein.
            for key, value in raw.items():
                if str(key) == ga:
                    return int(value)
            return None
        value = raw.get(ga, raw.get(int(re.match(r"\d+", ga).group(0))))
        return int(value) if value is not None else None
    return int(raw)


def belongs_to_ga(entry: dict, ga: str) -> bool:
    raw = entry.get("ga")
    if raw is None:
        return False
    parts = [part.strip() for part in str(raw).split(",")]
    return ga in parts


def load_catalog() -> list[dict]:
    entries = yaml.safe_load(CATALOG.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise SystemExit(f"Katalog ist keine Liste: {CATALOG}")
    return entries


def load_lectures(entries: list[dict], ga: str) -> list[dict]:
    lectures = []
    for entry in entries:
        if not belongs_to_ga(entry, ga):
            continue
        page = page_of(entry, ga)
        if page is None:
            continue
        lectures.append(
            {
                "id": str(entry["id"]),
                "ort": str(entry.get("ort") or "").strip(),
                "titel": " ".join(str(entry.get("vortragstitel") or "").split()),
                "page": page,
            }
        )
    lectures.sort(key=lambda item: (item["page"], item["id"]))
    # Dieselbe Anfangsseite: die Seite gehört dem früheren Vortrag.
    for index in range(1, len(lectures)):
        if lectures[index]["page"] <= lectures[index - 1]["page"]:
            lectures[index]["page"] = lectures[index - 1]["page"] + 1
    return lectures


def file_stem(lecture: dict) -> str:
    def part(value: str) -> str:
        return value.replace(" ", "_").replace("?", "_").replace("/", "_")

    stem = f"{lecture['id']}_{part(lecture['ort'])}_{part(lecture['titel'])}"
    # macOS lässt höchstens 255 Byte im Dateinamen zu.
    limit = 255 - len(".md")
    encoded = stem.encode("utf-8")
    if len(encoded) <= limit:
        return stem
    return encoded[:limit].decode("utf-8", errors="ignore")


def words_in(chunk: str) -> list[tuple[float, float, float, float, str]]:
    words = []
    for x0, y0, x1, y1, raw in _WORD.findall(chunk):
        text = html.unescape(raw.replace("\n", " ")).strip()
        if text:
            words.append((float(x0), float(y0), float(x1), float(y1), text))
    return words


def read_pdf_pages(pdf: Path) -> dict[int, list[tuple[float, float, float, float, str]]]:
    result = subprocess.run(
        ["pdftotext", "-bbox-layout", "-enc", "UTF-8", str(pdf), "-"],
        check=True,
        capture_output=True,
    )
    text = result.stdout.decode("utf-8", errors="replace")
    return {index: words_in(chunk) for index, chunk in enumerate(re.split(r"<page\b[^>]*>", text)[1:], start=1)}


def line_groups(words: list[tuple[float, float, float, float, str]]) -> list[Line]:
    ordered = sorted(words, key=lambda item: (item[1], item[0]))
    groups: list[list[tuple[float, float, float, float, str]]] = []
    for word in ordered:
        if groups and abs(word[1] - groups[-1][0][1]) <= 3:
            groups[-1].append(word)
        else:
            groups.append([word])
    lines = []
    for group in groups:
        group.sort(key=lambda item: item[0])
        text = " ".join(item[4] for item in group)
        lines.append(
            Line(
                page=0,
                x0=group[0][0],
                x1=group[-1][2],
                y=group[0][1],
                height=max(item[3] - item[1] for item in group),
                text=text,
                kind="text",
            )
        )
    return lines


def is_page_number(text: str) -> bool:
    return re.fullmatch(r"[–\-—]?\s*\d{1,4}\s*[–\-—]?", text.strip()) is not None


def height_mode(lines: list[Line]) -> tuple[float, int] | None:
    """Häufigste Höhe der längeren Zeilen auf dieser Seite, und wie viele Zeilen sie haben."""
    buckets: dict[float, int] = {}
    for line in lines:
        if is_page_number(line.text) or len(line.text) < 12:
            continue
        bucket = round(line.height * 2) / 2
        buckets[bucket] = buckets.get(bucket, 0) + 1
    if not buckets:
        return None
    height = max(buckets, key=buckets.get)
    return height, buckets[height]


def is_body_height(height: float, body_height: float) -> bool:
    return abs(height - body_height) <= HEIGHT_TOLERANCE


def classify(lines: list[Line], page: int, report: Report) -> list[Line]:
    mode = height_mode(lines)
    body_height = mode[0] if mode else 0.0
    body_indexes = [
        index
        for index, line in enumerate(lines)
        if mode and is_body_height(line.height, body_height) and not is_page_number(line.text)
    ]
    if not body_indexes:
        for line in lines:
            line.kind = "verworfen"
            report.add(page, "verworfen", line)
        return []
    first, last = body_indexes[0], body_indexes[-1]
    kept = []
    for index, line in enumerate(lines):
        if is_page_number(line.text):
            line.kind = "verworfen"
            report.add(page, "verworfen", line)
            continue
        if is_body_height(line.height, body_height):
            line.kind = "text"
            kept.append(line)
            continue
        if first < index < last and line.height >= body_height - NARROW_SPAN:
            line.kind = "knapp"
            report.add(page, "knapp", line)
            kept.append(line)
            continue
        line.kind = "verworfen"
        report.add(page, "verworfen", line)
    return kept


# Ergänzungsstrich: «Geistes- oder Körperkräfte», «vier- bis fünf». Der Bindestrich bleibt.
_HYPHEN_KEPT = {"und", "oder", "bis", "wie", "als", "noch", "bzw.", "sowie"}


def join_line_break(previous: str, following: str) -> tuple[str, str] | None:
    """Trennstrich am Zeilenende mit dem Folgewort verbinden. Echte Bindestrichwörter bleiben."""
    parts = following.split(" ", 1)
    if not parts or not parts[0][:1].islower():
        return None
    if previous.endswith(SOFT_HYPHEN):
        stem = previous[: -len(SOFT_HYPHEN)]
    elif len(previous) > 1 and previous.endswith("-") and not previous.endswith(" -"):
        follower = parts[0].lower().rstrip(".,;:")
        if follower in _HYPHEN_KEPT:
            return None
        stem = previous[:-1]
    else:
        return None
    rest = parts[1] if len(parts) > 1 else ""
    return stem + parts[0], rest


def merge_hyphens(lines: list[Line]) -> list[Line]:
    merged: list[Line] = []
    for line in lines:
        if merged:
            joined = join_line_break(merged[-1].text, line.text)
            if joined is not None:
                merged[-1].text, rest = joined
                if rest:
                    line.text = rest
                    line.continues = True
                    merged.append(line)
                continue
        merged.append(line)
    return merged


def opens_new_paragraph(lines: list[Line]) -> bool:
    """Eingezogene erste Zeile beginnt einen Absatz. Eine Folgezeile setzt den Satz fort."""
    if not lines or lines[0].continues:
        return False
    margin = statistics.median(line.x0 for line in lines)
    return lines[0].x0 >= margin + INDENT


def paragraphs(lines: list[Line]) -> list[str]:
    if not lines:
        return []
    margin = statistics.median(line.x0 for line in lines)
    blocks: list[list[str]] = [[]]
    previous_y = lines[0].y
    for line in lines:
        new_paragraph = not line.continues and (
            line.y - previous_y > PARAGRAPH_GAP or line.x0 >= margin + INDENT
        )
        if new_paragraph and blocks[-1]:
            blocks.append([])
        blocks[-1].append(line.text)
        previous_y = line.y
    return [" ".join(part) for part in blocks if part]


def section_height(words: list[tuple[float, float, float, float, str]]) -> float | None:
    """Schrift einer vollen Textseite. Titelseiten und leere Seiten zählen nicht."""
    mode = height_mode(line_groups(words))
    if mode is None or mode[1] < 8:
        return None
    return mode[0]


_APPARATUS_PAGE = re.compile(
    r"^(HINWEISE|ANMERKUNGEN|REGISTER|INHALT|I\.\s*Werke|"
    r"RUDOLF\s+STEINER\s+GESAMTAUSGABE|"
    r"Gliederung\s+nach)\b",
    re.I,
)


def is_apparatus_page(words: list[tuple[float, float, float, float, str]]) -> bool:
    """Hinweise, Werke-Liste und bibliographische Anhänge — nicht Vortragstext."""
    lines = line_groups(words)
    head = " ".join(line.text for line in lines[:8])
    head = re.sub(r"\s+", " ", head).strip()
    if _APPARATUS_PAGE.search(head):
        return True
    # Manche Bände setzen HINWEISE allein in die erste Zeile.
    if lines and _APPARATUS_PAGE.match(lines[0].text.strip()):
        return True
    return False


def lecture_pages(
    lectures: list[dict],
    index: int,
    pdf_pages: dict[int, list[tuple[float, float, float, float, str]]],
) -> list[int]:
    start = lectures[index]["page"]
    last = max(pdf_pages) if pdf_pages else 0
    if index + 1 < len(lectures):
        end = min(lectures[index + 1]["page"] - 1, last)
    else:
        end = last
        # Letzter Vortrag: Ende bei klarem Schriftwechsel (Hinweise/Register).
        # Titelseiten messen oft 1–5 pt größer als der Körper — einmal auf die
        # erste Folgeseite mit Körperschrift umankern (GA 79 / Acrobat-OCR).
        # Wenn Hinweise dieselbe Schriftgröße haben (z. B. GA 51), greift der
        # Textanker is_apparatus_page.
        lecture_height: float | None = None
        reanchored = False
        for page in range(start, last + 1):
            words = pdf_pages.get(page, [])
            if page > start and is_apparatus_page(words):
                end = page - 1
                break
            height = section_height(words)
            if height is None:
                continue
            if lecture_height is None:
                lecture_height = height
                continue
            if (
                not reanchored
                and page <= start + 2
                and height >= 14.0
                and 0.5 <= abs(height - lecture_height) <= 6.0
            ):
                lecture_height = height
                reanchored = True
                continue
            reanchored = True
            # Hinweise/Apparat: klar kleiner; Anzeigetitel: klar größer.
            # Kleine Acrobat-OCR-Schwankungen (±1–2 pt) bleiben im Vortrag.
            if height <= lecture_height - 2.5 or height < 12.5:
                end = page - 1
                break
            if height >= lecture_height + 3.0:
                end = page - 1
                break
        if lecture_height is None:
            return []
    if end < start:
        return []
    return list(range(start, end + 1))


def render(
    lecture: dict,
    pages: list[int],
    pdf_pages: dict[int, list[tuple[float, float, float, float, str]]],
    report: Report,
) -> str:
    chunks = [f"# {lecture['titel']}", ""]
    page_lines: dict[int, list[Line]] = {}
    for page in pages:
        lines = classify(line_groups(pdf_pages.get(page, [])), page, report)
        for line in lines:
            line.page = page
        page_lines[page] = merge_hyphens(lines)
    # Silbentrennung über den Seitenwechsel: das Wort endet auf der vorherigen Seite.
    for previous, following in zip(pages, pages[1:]):
        if not page_lines[previous] or not page_lines[following]:
            continue
        joined = join_line_break(page_lines[previous][-1].text, page_lines[following][0].text)
        if joined is None:
            continue
        page_lines[previous][-1].text, rest = joined
        if rest:
            page_lines[following][0].text = rest
            page_lines[following][0].continues = True
        else:
            page_lines[following].pop(0)
    paragraphs_out: list[str] = []
    previous_page: int | None = None
    for page in pages:
        page_paragraphs = paragraphs(page_lines[page])
        if previous_page is not None and page_paragraphs:
            marker = f"[<- S. {previous_page} | S. {page} ->]"
            if paragraphs_out and not opens_new_paragraph(page_lines[page]):
                paragraphs_out[-1] = f"{paragraphs_out[-1]} {marker} {page_paragraphs[0]}"
                paragraphs_out.extend(page_paragraphs[1:])
            else:
                page_paragraphs[0] = f"{marker} {page_paragraphs[0]}"
                paragraphs_out.extend(page_paragraphs)
        else:
            paragraphs_out.extend(page_paragraphs)
        previous_page = page
    for paragraph in paragraphs_out:
        chunks.append(paragraph)
        chunks.append("")
    return "\n".join(chunks).rstrip() + "\n"


def volumes_51_to_354(entries: list[dict]) -> list[str]:
    found: set[str] = set()
    for entry in entries:
        raw = entry.get("ga")
        if raw is None:
            continue
        for part in str(raw).split(","):
            part = part.strip()
            match = re.fullmatch(r"(\d+)([a-zA-Z]?)", part)
            if match and 51 <= int(match.group(1)) <= 354:
                found.add(part)
    return sorted(found, key=lambda ga: (int(re.match(r"\d+", ga).group(0)), re.fullmatch(r"\d+([a-zA-Z]?)", ga).group(1).lower()))


def export_ga(entries: list[dict], ga: str) -> int:
    pdf = pdf_path(ga)
    if not pdf.is_file():
        print(f"GA {ga}: PDF fehlt ({pdf.name}), übersprungen", flush=True)
        return 0
    lectures = load_lectures(entries, ga)
    if not lectures:
        print(f"GA {ga}: keine Vorträge mit Seite, übersprungen", flush=True)
        return 0
    out_dir = OUT_ROOT / folder_name(ga)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = Report()
    pdf_pages = read_pdf_pages(pdf)
    print(f"GA {ga}: {len(lectures)} Vorträge, PDF {pdf.name}, {len(pdf_pages)} Seiten", flush=True)
    written = 0
    for index, lecture in enumerate(lectures):
        pages = lecture_pages(lectures, index, pdf_pages)
        if not pages:
            print(f"  {lecture['id']} ohne Seiten", flush=True)
            continue
        text = render(lecture, pages, pdf_pages, report)
        path = out_dir / f"{file_stem(lecture)}.md"
        path.write_text(text, encoding="utf-8")
        written += 1
        print(f"  {path.name}  S.{pages[0]}–{pages[-1]}", flush=True)
    report_path = out_dir / "_bericht.txt"
    header = [
        f"GA {ga}",
        "Seitenfeld im Katalog: seite_ga (seite_pdf ist im Katalog nicht vorhanden)",
        "Vortragsschrift: häufigste Zeilenhöhe der jeweiligen Seite",
        "verworfen: kleinere Blöcke vor dem ersten oder nach dem letzten Vortragstext der Seite",
        "knapp: kleinere Blöcke zwischen Vortragstext, in die Markdown-Datei aufgenommen",
        "",
    ]
    report_path.write_text("\n".join(header + report.rows) + "\n", encoding="utf-8")
    old_report = out_dir / "_bericht-testlauf.txt"
    if old_report.is_file():
        old_report.unlink()
    print(f"Bericht: {report_path} ({len(report.rows)} Zeilen)", flush=True)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Vortragstext der GA-Bände als Markdown.")
    parser.add_argument("--ga", help="Ein Band, z. B. 329 oder 71a")
    parser.add_argument("--all", action="store_true", help="Bände 51 bis 354")
    args = parser.parse_args()
    if args.all == bool(args.ga):
        raise SystemExit("Genau eines von --ga und --all angeben.")
    entries = load_catalog()
    if args.ga:
        export_ga(entries, args.ga)
        return
    bands = volumes_51_to_354(entries)
    print(f"{len(bands)} Bände", flush=True)
    written = 0
    failed: list[str] = []
    for number, ga in enumerate(bands, start=1):
        print(f"[{number}/{len(bands)}]", flush=True)
        try:
            written += export_ga(entries, ga)
        except subprocess.CalledProcessError as error:
            failed.append(f"{ga}: pdftotext {error.returncode}")
            print(f"GA {ga}: pdftotext fehlgeschlagen", flush=True)
    print(f"Fertig: {written} Dateien, {len(failed)} fehlgeschlagen", flush=True)
    for line in failed:
        print(line, flush=True)


if __name__ == "__main__":
    main()
