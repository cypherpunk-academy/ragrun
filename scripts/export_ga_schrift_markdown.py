#!/usr/bin/env python3
"""Inhaltsverzeichnisse und Kapitel-Markdown der Schriften GA 1–38.

Schreibt:
  ragkeep/ga/_audit/ga-001-038-inhaltsverzeichnisse.md
  ragkeep/ga/ga_NNN/{NN}_{slug}.md  und  _kapitel.yaml  (nur Prosabände 1–37)

Briefbände GA 38 / 38_1 / 38_2 / 38_3: nur Verzeichnis, kein Markdown.
GA 4a und alles über 38: nicht behandelt.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Wiederverwendung der PDF-/Render-Helfer aus dem Vortrags-Exporter.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_ga_lecture_markdown as lec  # noqa: E402

PDF_DIR = Path("/Users/michael/GA 180dpi/GA-Acrobat/GA/2026")
OUT_ROOT = Path("/Users/michael/Reniets/Ai/ragrun/ragkeep/ga")
AUDIT_MD = OUT_ROOT / "_audit" / "ga-001-038-inhaltsverzeichnisse.md"

# Prosa: ingest. Briefe: nur Verzeichnis.
INGEST_BANDS = [str(n) for n in range(1, 38)]
TOC_ONLY_BANDS = ["38", "38_1", "38_2", "38_3"]
ALL_BANDS = INGEST_BANDS + TOC_ONLY_BANDS

_SKIP_TITLE = re.compile(
    r"^(Hin(?:weise|weife)|Himweise|Anmerkungen|"
    r"Namen-?Register|Personenregister|Gesamtregister|Register|"
    r"Anhang\b|Verzeichnis|"
    r"Zu dieser Ausgabe|Editionsgeschichte|"
    r"Vorbemerkung(en)? der Herausgeber|"
    r"Editorische Vorbemerkung|"
    r"Geleitwort von Marie Steiner|"
    r"Handschriftenwiedergaben|"
    r"Korrigenda|"
    r"Übersicht über die Rudolf Steiner Gesamtausgabe|"
    r"Übersicht über die Herausgabetätigkeit|"
    r"Alphabetisches Verzeichnis|"
    r"Chronologische Übersicht|"
    r"Nachtrag zu den Briefen|"
    r"Verwendete Zeichen|"
    r"Briefe Nr\.|"
    r"Briefe ohne genaues Datum|"
    r"Ergänzung:|"
    r"Faksimiles|"
    r"Manuskript|"
    r"Fragment)\b",
    re.I,
)

# Titel … Seite  (Punkte und/oder Leerzeichen als Führer)
_ENTRY = re.compile(
    r"^(?P<title>.+?)\s*(?:\.{2,}|\u2026+)\s*(?P<page>\d{1,4})\s*$"
)
_ENTRY_SPACES = re.compile(
    r"^(?P<title>.+?)\s{2,}(?P<page>\d{1,4})\s*$"
)
# Zeile endet mit Seitenzahl, davor Text (auch mit wenigen Punkten)
_ENTRY_TAIL = re.compile(
    r"^(?P<title>.+?)\s+(?P<page>\d{1,4})\s*$"
)


@dataclass
class TocRow:
    kind: str  # chapter | section | skip
    title: str
    page: int | None = None


@dataclass
class Chapter:
    index: int
    title: str
    page_start: int
    page_end: int
    file: str = ""
    id: str = ""
    warn: str = ""


@dataclass
class BandResult:
    ga: str
    title: str
    toc_pages: list[int] = field(default_factory=list)
    rows: list[TocRow] = field(default_factory=list)
    chapters: list[Chapter] = field(default_factory=list)
    ingest: bool = True
    error: str = ""


def band_code(ga: str) -> str:
    """1 → 001, 38_1 → 038_1."""
    if "_" in ga:
        num, rest = ga.split("_", 1)
        return f"{int(num):03d}_{rest}"
    match = re.fullmatch(r"(\d+)([a-zA-Z]?)", ga)
    if not match:
        raise SystemExit(f"GA-Nummer nicht lesbar: {ga}")
    return f"{int(match.group(1)):03d}{match.group(2)}"


def folder_name(ga: str) -> str:
    return f"ga_{band_code(ga)}"


def pdf_path(ga: str) -> Path:
    return PDF_DIR / f"GA {band_code(ga)}.pdf"


def page_plain(pdf_pages: dict[int, list], page: int) -> str:
    words = pdf_pages.get(page, [])
    return " ".join(w[4] for w in words)


def page_layout(pdf: Path, start: int, end: int) -> str:
    result = subprocess.run(
        [
            "pdftotext",
            "-f",
            str(start),
            "-l",
            str(end),
            "-enc",
            "UTF-8",
            "-layout",
            "-q",
            str(pdf),
            "-",
        ],
        check=True,
        capture_output=True,
    )
    return result.stdout.decode("utf-8", errors="replace")


def find_toc_pages(pdf: Path, pdf_pages: dict[int, list]) -> list[int]:
    """Seiten, deren Text mit INHALT beginnt; Folgezeiten mit Punktführern dazu."""
    hits: list[int] = []
    last = max(pdf_pages) if pdf_pages else 0
    for page in range(1, last + 1):
        text = page_plain(pdf_pages, page).strip()
        if not text:
            continue
        head = re.sub(r"\s+", " ", text)[:40].upper()
        if head.startswith("INHALT"):
            hits.append(page)
    if not hits:
        return []
    # Ersten Block nehmen (vorn oder hinten): zusammenhängende Seiten ab erstem Treffer
    # bzw. dem letzten Block, wenn der erste Block nur eine Titelseite ohne Einträge wäre.
    blocks: list[list[int]] = [[hits[0]]]
    for page in hits[1:]:
        if page == blocks[-1][-1] + 1:
            blocks[-1].append(page)
        else:
            blocks.append([page])
    # Bevorzuge den Block mit den meisten Einträgen (Punktführer)
    best = blocks[0]
    best_score = -1
    for block in blocks:
        start, end = block[0], block[-1]
        # TOC kann über Seiten ohne „INHALT“-Kopf weitergehen
        end_ext = end
        for page in range(end + 1, min(end + 8, last + 1)):
            raw = page_layout(pdf, page, page)
            if raw.count(".") < 15 and not re.search(r"\d{1,4}\s*$", raw, re.M):
                break
            # Stop if body text (long paragraphs without leaders)
            lines = [ln for ln in raw.splitlines() if ln.strip()]
            leader_lines = sum(1 for ln in lines if ".." in ln or ln.count(".") > 8)
            if leader_lines < 2 and page > end:
                break
            end_ext = page
        pages = list(range(start, end_ext + 1))
        layout = page_layout(pdf, pages[0], pages[-1])
        score = layout.count("...")
        if score > best_score:
            best_score = score
            best = pages
    return best


def volume_title(pdf: Path, pdf_pages: dict[int, list]) -> str:
    for page in (3, 2, 1, 4):
        text = page_plain(pdf_pages, page)
        cleaned = re.sub(r"\s+", " ", text).strip()
        if "RUDOLF STEINER" in cleaned.upper() and len(cleaned) > 40:
            # Strip leading RUDOLF STEINER / GESAMTAUSGABE noise
            t = cleaned
            t = re.sub(r"^RUDOLF STEINER\s+", "", t, flags=re.I)
            t = re.sub(r"^GESAMTAUSGABE\s+", "", t, flags=re.I)
            t = re.sub(r"\s+\d{4}\s+RUDOLF STEINER VERLAG.*$", "", t, flags=re.I)
            t = re.sub(r"\s+RUDOLF STEINER VERLAG.*$", "", t, flags=re.I)
            return " ".join(t.split())[:160]
    return ""


def normalize_title(raw: str) -> str:
    t = raw.replace("\xad", "").replace("­", "")
    t = re.sub(r"\s+", " ", t).strip(" .")
    t = re.sub(r"\s*\.{2,}\s*$", "", t)
    return t.strip()


def is_skip_title(title: str) -> bool:
    return bool(_SKIP_TITLE.search(title.strip()))


def is_section_heading(title: str, indent: int) -> bool:
    """Zwischenüberschrift ohne Seitenzahl (Teil, Band, zentrierter Block)."""
    t = title.strip()
    if not t or len(t) > 90:
        return False
    if re.fullmatch(r"[IVXLC]+\.?", t):
        return True
    if re.match(r"^[A-D]\.\s+[A-ZÄÖÜ]", t):
        return True
    if re.search(r"\bBand\b", t, re.I):
        return True
    if re.match(r"^\d{4},", t):
        return True
    # Stark eingerückt / zentriert und kurz → Teilüberschrift
    if indent >= 12 and len(t.split()) <= 10 and not re.match(r"^[IVXLC]+\.", t):
        return True
    if t.isupper() and len(t) < 60 and not re.match(r"^[IVXLC]+\.", t):
        return True
    return False


def looks_like_chapter_start(title: str) -> bool:
    """Anfang einer mehrzeiligen Kapitelzeile (römische Nummer, Meditation, …)."""
    t = title.strip()
    if re.match(r"^[IVXLC]+\.\s+\S", t):
        return True
    if re.match(r"^\d+\.\s+\S", t):
        return True
    if re.match(r"^(Erste|Zweite|Dritte|Vierte|Fünfte|Sechste|Siebente|Achte|Neunte|Zehnte)\s+", t, re.I):
        return True
    return False


def parse_entry_line(stripped: str) -> tuple[str, int] | None:
    m = _ENTRY.match(stripped)
    if not m:
        m = _ENTRY_SPACES.match(stripped)
    if not m:
        # Punkte mit Leerzeichen: «. . . 287»
        if re.search(r"(?:\.\s*){2,}\d{1,4}\s*$", stripped) or (
            ".." in stripped or re.search(r"\s{3,}\d{1,4}\s*$", stripped)
        ):
            m = _ENTRY_TAIL.match(re.sub(r"(?:\.\s*){2,}", "....", stripped))
            if not m:
                m = _ENTRY_TAIL.match(stripped)
    if not m:
        return None
    return normalize_title(m.group("title")), int(m.group("page"))


def parse_toc_layout(layout: str) -> list[TocRow]:
    rows: list[TocRow] = []
    pending_title: str | None = None

    def emit_chapter(title: str, page: int) -> None:
        title = normalize_title(title)
        if is_skip_title(title):
            rows.append(TocRow(kind="skip", title=title, page=page))
        else:
            rows.append(TocRow(kind="chapter", title=title, page=page))

    for raw in layout.splitlines():
        line = raw.replace("\xad", "").replace("­", "").rstrip()
        if not line.strip():
            continue
        stripped = line.strip()
        indent = len(line) - len(line.lstrip(" "))
        if stripped.upper() == "INHALT" or stripped.upper().startswith("INHALT "):
            continue

        parsed = parse_entry_line(stripped)
        if parsed:
            title, page = parsed
            if pending_title:
                # «XI. Weltzweck…» + «(Bestimmung…) … 184»
                title = normalize_title(f"{pending_title} {title}")
                pending_title = None
            emit_chapter(title, page)
            continue

        if is_skip_title(stripped):
            pending_title = None
            rows.append(TocRow(kind="skip", title=normalize_title(stripped), page=None))
            continue

        if is_section_heading(stripped, indent):
            pending_title = None
            rows.append(TocRow(kind="section", title=normalize_title(stripped), page=None))
            continue

        # Fortsetzungszeile ohne Seitenzahl
        if pending_title:
            pending_title = normalize_title(f"{pending_title} {stripped}")
            continue
        if looks_like_chapter_start(stripped):
            pending_title = normalize_title(stripped)
            continue
        # Untertitel unter letztem Kapitel (z. B. Meditation-Beschreibung, leicht eingerückt)
        if (
            rows
            and rows[-1].kind == "chapter"
            and 2 <= indent < 12
            and (stripped.startswith("(") or stripped[:1].islower() or indent >= 3)
        ):
            rows[-1].title = normalize_title(f"{rows[-1].title} {stripped}")
            continue
        # Sonst: nächste numerierte Zeile ergänzen
        pending_title = normalize_title(stripped)

    return rows


def slugify(title: str, max_len: int = 80) -> str:
    t = title.replace("«", "").replace("»", "").replace("„", "").replace("“", "")
    t = re.sub(r"[^\w\s\-äöüÄÖÜß]", "", t, flags=re.UNICODE)
    t = re.sub(r"\s+", "_", t.strip())
    t = t.strip("_")
    if len(t) > max_len:
        t = t[:max_len].rstrip("_")
    return t or "Kapitel"


def distinctive_word(title: str) -> str | None:
    stop = {
        "der",
        "die",
        "das",
        "und",
        "oder",
        "ein",
        "eine",
        "einer",
        "eines",
        "dem",
        "den",
        "des",
        "im",
        "in",
        "zu",
        "zur",
        "zum",
        "von",
        "vom",
        "für",
        "mit",
        "als",
        "auf",
        "an",
        "aus",
        "über",
        "nach",
        "bei",
        "ist",
        "sind",
        "was",
        "wie",
        "ob",
        "es",
        "i",
        "ii",
        "iii",
        "iv",
        "v",
        "vi",
        "vii",
        "viii",
        "ix",
        "x",
        "xi",
        "xii",
        "xiii",
        "xiv",
        "xv",
        "xvi",
        "xvii",
        "xviii",
    }
    for word in re.findall(r"[A-Za-zÄÖÜäöüß]{4,}", title):
        if word.lower() not in stop:
            return word
    return None


def title_on_page(pdf_pages: dict[int, list], page: int, title: str) -> bool:
    text = page_plain(pdf_pages, page).upper()
    word = distinctive_word(title)
    if not word:
        return True
    return word.upper() in text


def chapter_pages(
    chapters: list[dict],
    index: int,
    pdf_pages: dict[int, list],
    apparatus_page: int | None,
) -> list[int]:
    """Wie lecture_pages, zusätzlich TOC-Apparat-Seite als Obergrenze."""
    start = chapters[index]["page"]
    last = max(pdf_pages) if pdf_pages else 0
    if apparatus_page is not None:
        last = min(last, apparatus_page - 1)
    if index + 1 < len(chapters):
        end = min(chapters[index + 1]["page"] - 1, last)
        if end < start:
            return []
        return list(range(start, end + 1))
    # Letztes Kapitel: Schriftwechsel / Apparat
    fake = [{"page": start}]
    pages = lec.lecture_pages(fake, 0, {p: pdf_pages[p] for p in pdf_pages if p <= last})
    return pages


def first_apparatus_page_from_rows(rows: list[TocRow]) -> int | None:
    for row in rows:
        if row.kind == "skip" and row.page is not None:
            return row.page
    return None


def build_chapters(
    ga: str,
    rows: list[TocRow],
    pdf_pages: dict[int, list],
) -> list[Chapter]:
    apparatus = first_apparatus_page_from_rows(rows)
    chapter_dicts: list[dict] = []
    for row in rows:
        if row.kind != "chapter" or row.page is None:
            continue
        # Deduplicate same start page: keep first title, extend later if same page
        if chapter_dicts and chapter_dicts[-1]["page"] == row.page:
            # Same page start — merge titles rarely; prefer keeping separate only if different
            continue
        chapter_dicts.append({"title": row.title, "page": row.page})

    out: list[Chapter] = []
    for index, item in enumerate(chapter_dicts):
        pages = chapter_pages(chapter_dicts, index, pdf_pages, apparatus)
        if not pages:
            pages = [item["page"]]
        warn = ""
        if not title_on_page(pdf_pages, pages[0], item["title"]):
            warn = f"Titelwort nicht auf Anfangsseite S.{pages[0]}"
        num = index + 1
        slug = slugify(item["title"])
        fname = f"{num:02d}_{slug}.md"
        out.append(
            Chapter(
                index=num,
                title=item["title"],
                page_start=pages[0],
                page_end=pages[-1],
                file=fname,
                id=f"{ga}:{num:02d}",
                warn=warn,
            )
        )
    return out


def render_chapter(
    title: str,
    pages: list[int],
    pdf_pages: dict[int, list],
) -> str:
    lecture = {"titel": title, "id": "", "ort": ""}
    report = lec.Report()
    return lec.render(lecture, pages, pdf_pages, report)


def write_kapitel_yaml(path: Path, ga: str, chapters: list[Chapter]) -> None:
    lines = [f'ga: "{ga}"', "chapters:"]
    for ch in chapters:
        lines.append(f"  - index: {ch.index}")
        lines.append(f'    id: "{ch.id}"')
        lines.append(f'    title: "{ch.title.replace(chr(34), chr(39))}"')
        lines.append(f"    page_start: {ch.page_start}")
        lines.append(f"    page_end: {ch.page_end}")
        lines.append(f'    file: "{ch.file}"')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def process_band(ga: str, write_md: bool) -> BandResult:
    ingest = ga in INGEST_BANDS
    pdf = pdf_path(ga)
    result = BandResult(ga=ga, title="", ingest=ingest)
    if not pdf.is_file():
        result.error = f"PDF fehlt ({pdf.name})"
        return result
    try:
        pdf_pages = lec.read_pdf_pages(pdf)
    except subprocess.CalledProcessError as err:
        result.error = f"pdftotext fehlgeschlagen ({err.returncode})"
        return result
    result.title = volume_title(pdf, pdf_pages)
    toc_pages = find_toc_pages(pdf, pdf_pages)
    if not toc_pages:
        result.error = "Kein Inhaltsverzeichnis gefunden"
        return result
    result.toc_pages = toc_pages
    layout = page_layout(pdf, toc_pages[0], toc_pages[-1])
    result.rows = parse_toc_layout(layout)
    result.chapters = build_chapters(ga, result.rows, pdf_pages)

    if write_md and ingest and result.chapters:
        out_dir = OUT_ROOT / folder_name(ga)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Remove previous numbered chapter md files
        for old in out_dir.glob("[0-9][0-9]_*.md"):
            old.unlink()
        for ch in result.chapters:
            pages = list(range(ch.page_start, ch.page_end + 1))
            text = render_chapter(ch.title, pages, pdf_pages)
            (out_dir / ch.file).write_text(text, encoding="utf-8")
            print(
                f"  {ch.file}  S.{ch.page_start}–{ch.page_end}"
                + (f"  WARN {ch.warn}" if ch.warn else ""),
                flush=True,
            )
        write_kapitel_yaml(out_dir / "_kapitel.yaml", ga, result.chapters)
    return result


def write_audit(results: list[BandResult]) -> None:
    AUDIT_MD.parent.mkdir(parents=True, exist_ok=True)
    parts: list[str] = [
        "# Inhaltsverzeichnisse GA 1–38",
        "",
        "Seitenzahl = PDF-Seite. Endseite = Anfang des nächsten Kapitels − 1,",
        "beim letzten Kapitel Apparat/Schriftwechsel.",
        "",
        "Nicht ingestiert: Briefbände GA 38, 38/1, 38/2, 38/3.",
        "Nicht behandelt: GA 4a und alles über GA 38.",
        "",
    ]
    for res in results:
        parts.append(f"## GA {res.ga}")
        if res.title:
            parts.append(f"*{res.title}*")
        parts.append("")
        if res.error:
            parts.append(f"**Fehler:** {res.error}")
            parts.append("")
            continue
        if not res.ingest:
            parts.append("*Nur Verzeichnis — kein Kapitel-Markdown / kein Ingest.*")
            parts.append("")
        if res.toc_pages:
            parts.append(
                f"Inhaltsverzeichnis: PDF-Seiten {res.toc_pages[0]}–{res.toc_pages[-1]}"
            )
            parts.append("")
        # Sections + chapters in order from rows, with end pages from chapters map
        by_start = {ch.page_start: ch for ch in res.chapters}
        parts.append("| Kapitel | Anfang | Ende |")
        parts.append("|---|---:|---:|")
        for row in res.rows:
            if row.kind == "section":
                parts.append(f"| *{row.title}* |  |  |")
            elif row.kind == "skip":
                page = row.page if row.page is not None else ""
                parts.append(f"| _(übersprungen)_ {row.title} | {page} |  |")
            elif row.kind == "chapter":
                ch = by_start.get(row.page) if row.page is not None else None
                if ch:
                    warn = f" ⚠️ {ch.warn}" if ch.warn else ""
                    parts.append(
                        f"| {ch.index}. {ch.title}{warn} | {ch.page_start} | {ch.page_end} |"
                    )
                else:
                    parts.append(f"| {row.title} | {row.page} |  |")
        parts.append("")
        if res.chapters:
            parts.append(f"Kapitel für Ingest: {len(res.chapters)}")
            parts.append("")
    AUDIT_MD.write_text("\n".join(parts).rstrip() + "\n", encoding="utf-8")
    print(f"Audit: {AUDIT_MD}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Schriften GA 1–38: TOC + Kapitel-Markdown")
    parser.add_argument("--ga", help="Ein Band, z. B. 4 oder 38_1")
    parser.add_argument("--all", action="store_true", help="GA 1–38 (ohne 4a)")
    parser.add_argument(
        "--toc-only",
        action="store_true",
        help="Nur Inhaltsverzeichnis-MD, kein Kapitel-Export",
    )
    args = parser.parse_args()
    if args.all == bool(args.ga):
        raise SystemExit("Genau eines von --ga und --all angeben.")
    bands = ALL_BANDS if args.all else [args.ga.replace("ga_", "").lstrip("0") or "0"]
    # Normalize 38_1 style; keep leading structure
    normalized: list[str] = []
    for b in bands:
        b = b.strip()
        if re.fullmatch(r"0*\d+", b):
            normalized.append(str(int(b)))
        elif re.fullmatch(r"0*\d+_[0-9a-zA-Z]+", b):
            num, rest = b.split("_", 1)
            normalized.append(f"{int(num)}_{rest}")
        else:
            normalized.append(b)
    results: list[BandResult] = []
    for number, ga in enumerate(normalized, start=1):
        print(f"[{number}/{len(normalized)}] GA {ga}", flush=True)
        write_md = not args.toc_only
        res = process_band(ga, write_md=write_md)
        if res.error:
            print(f"  ERROR: {res.error}", flush=True)
        else:
            print(
                f"  TOC S.{res.toc_pages[0]}–{res.toc_pages[-1]}, "
                f"{len(res.chapters)} Kapitel"
                + ("" if res.ingest else " (nur Verzeichnis)"),
                flush=True,
            )
        results.append(res)
    write_audit(results)


if __name__ == "__main__":
    main()
