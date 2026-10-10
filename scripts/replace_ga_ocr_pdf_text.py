#!/usr/bin/env python3
"""Phase 1: eindeutige OCR-Wörter in der Textebene der GA-PDFs ersetzen.

Liste (a) wird als Ganzwort, nur in der notierten Schreibung, in der Textebene
unter dem Seitenbild ersetzt. Liste (b) bleibt stehen. Dieselbe Ausführung
schreibt die Kontextliste für die spätere Redaktion.

Aufruf (im ragrun-venv):

    .venv/bin/python scripts/replace_ga_ocr_pdf_text.py --dry-run --only "GA 001.pdf"
    .venv/bin/python scripts/replace_ga_ocr_pdf_text.py --only "GA 001.pdf"
    .venv/bin/python scripts/replace_ga_ocr_pdf_text.py

Ein bereits abgeschlossener Band steht in der Datei neben der Kontextliste
(Endung .done) und wird beim nächsten Lauf übersprungen. Die PDF wird über
eine Temp-Datei im selben Ordner ersetzt.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import pikepdf
from pikepdf import parse_content_stream, unparse_content_stream

DEFAULT_DIR = Path.home() / "GA 180dpi/GA-Acrobat/GA/2026"
DEFAULT_CONTEXT = (
    Path(__file__).resolve().parents[2] / "ragprep/config/ocr_liste_b_kontext.txt"
)
BACKUP_DIR = Path("/Volumes/Steiner/GA 180dpi/backup")

# Liste (a): Ganzwort, case-sensitive. Längere Formen stehen vor kürzeren,
# damit „abzusondem“ nicht von „sondem“ zerschnitten wird.
LIST_A: tuple[tuple[str, str], ...] = (
    ("Dörnachs", "Dornachs"),
    ("Dörnach", "Dornach"),
    ("DÖRNACH", "DORNACH"),
    ("Domach", "Dornach"),
    ("Maerz", "März"),
    ("Faris", "Paris"),
    ("Züridi", "Zürich"),
    ("Dornacb", "Dornach"),
    ("Kristiana", "Kristiania"),
    ("Dornadh", "Dornach"),
    ("Febtuar", "Februar"),
    ("Augast", "August"),
    ("Instmktionsstunde", "Instruktionsstunde"),
    ("Neuchaiel", "Neuchâtel"),
    ("Heisingfors", "Helsingfors"),
    ("IVien", "Wien"),
    ("CJffentlidier", "öffentlicher"),
    ("Offentlidier", "öffentlicher"),
    ("öffentlidier", "öffentlicher"),
    ("abzusondem", "abzusondern"),
    ("sondem", "sondern"),
    ("Dornadi", "Dornach"),
    ("bürgerlidie", "bürgerliche"),
    ("Vorstelhing", "Vorstellung"),
    ("gesproeben", "gesprochen"),
    ("geoflenbart", "geoffenbart"),
    ("Gließmaßen", "Gliedmaßen"),
    ("außerordeutlich", "außerordentlich"),
    ("einzelden", "einzelnen"),
    ("Menscheh", "Menschen"),
    ("Naturwirklichkelt", "Naturwirklichkeit"),
    ("dreigliedtig", "dreigliedrig"),
    ("Fordetung", "Forderung"),
    ("künstletisch", "künstlerisch"),
    ("Territotium", "Territorium"),
    ("welthistotische", "welthistorische"),
    ("Zusammenptessen", "Zusammenpressen"),
    ("Anschaung", "Anschauung"),
    ("Anwort", "Antwort"),
    ("Beipiel", "Beispiel"),
    ("anspruchloses", "anspruchsloses"),
    ("Bestelliste", "Bestellliste"),
    ("Eisschnellauf", "Eisschnelllauf"),
    ("Kontrollampe", "Kontrolllampe"),
    ("Embyologie", "Embryologie"),
    ("Erkentnis", "Erkenntnis"),
    ("Geistesfoscher", "Geistesforscher"),
    ("Geometri", "Geometrie"),
    ("Gesitesleben", "Geistesleben"),
    ("inviduelle", "individuelle"),
    ("Benachrichtung", "Benachrichtigung"),
    ("Katastophe", "Katastrophe"),
    ("Keyerling", "Keyserling"),
    ("Lebensrhytmus", "Lebensrhythmus"),
    ("Mitleidlehre", "Mitleidslehre"),
    ("Naurwissenschaft", "Naturwissenschaft"),
    ("naturwissenschaflichen", "naturwissenschaftlichen"),
    ("Organistion", "Organisation"),
    ("Rechung", "Rechnung"),
    ("Theosopie", "Theosophie"),
    ("Totzkismus", "Trotzkismus"),
    ("unbhängig", "unabhängig"),
    ("Urmenscheit", "Urmenschheit"),
    ("Veständnis", "Verständnis"),
    ("Wirtschafsleben", "Wirtschaftsleben"),
    ("Zivilsation", "Zivilisation"),
    ("zvilisierten", "zivilisierten"),
    ("religösen", "religiösen"),
    ("Universtätswesen", "Universitätswesen"),
    ("höhren", "höheren"),
    ("übigens", "übrigens"),
    ("inbesondere", "insbesondere"),
    ("lebenförderndes", "lebensförderndes"),
    ("Vorstellungwelt", "Vorstellungswelt"),
    ("Weltanschaungsschule", "Weltanschauungsschule"),
    ("Zäheit", "Zähheit"),
    ("Köpenikerstraße", "Köpenickerstraße"),
    ("dasjenig", "dasjenige"),
    ("erschaffner", "erschaffener"),
    ("meintwillen", "meinetwillen"),
    ("ununterbrochner", "ununterbrochener"),
    ("ausdrückken", "ausdrücken"),
    ("erkennnen", "erkennen"),
    ("gegrade", "gerade"),
    ("völliggen", "völligen"),
    ("Seelenverfasssung", "Seelenverfassung"),
    ("Verpflichttungen", "Verpflichtungen"),
    ("Wirtschafttsorganismus", "Wirtschaftsorganismus"),
    ("Zusammenfasssung", "Zusammenfassung"),
    ("Unsterblichkeitssidee", "Unsterblichkeitsidee"),
    ("Vernunftstrieb", "Vernunfttrieb"),
    ("Regsammachen", "Regsamachen"),
    ("Trotzkiismus", "Trotzkismus"),
    ("Metarmorphose", "Metamorphose"),
    ("kennenzulernern", "kennenzulernen"),
    ("alleröftersten", "alleröftesten"),
    ("beantbeantwortete", "beantwortete"),
    ("nanamentlich", "namentlich"),
    ("Substantantiellen", "Substantiellen"),
    ("Schmierageblätter", "Schmierblätter"),
    ("Verständnisund", "Verständnis"),
    ("Berufszusam-Zusammenhänge", "Berufszusammenhänge"),
    ("Osterreich", "Österreich"),
    ("Atherleib", "Ätherleib"),
    ("Atherisches", "Ätherisches"),
    ("gewohnliche", "gewöhnliche"),
    ("gewohnlichen", "gewöhnlichen"),
    ("Passivitat", "Passivität"),
    ("Gefahrten", "Gefährten"),
    ("Wölkenkuckucksheim", "Wolkenkuckucksheim"),
    ("Arbeitsentlöhnung", "Arbeitsentlohnung"),
    ("Geisteswissenschäft", "Geisteswissenschaft"),
    ("Naturwissenschäft", "Naturwissenschaft"),
    ("Hingäbe", "Hingabe"),
    ("Kirchengemeinschäften", "Kirchengemeinschaften"),
    ("thätigkeit", "Tätigkeit"),
    ("speci\ufb01sch", "spezifisch"),
)

REPL_A = dict(LIST_A)
_ALTS = sorted((src for src, _ in LIST_A), key=len, reverse=True)
WORD_RE = re.compile("(?<!\\w)(" + "|".join(re.escape(src) for src in _ALTS) + ")(?!\\w)")

# Liste (b): nur in die Kontextliste, nicht ersetzen.
LIST_B_SINGLE: dict[str, str] = {
    "Darnach": "Dornach",
    "Bem": "Bern",
    "Marz": "März",
    "Mär%": "März",
    "Mär^": "März",
    "Mär£": "März",
    "Mar£": "März",
    "Lin%": "Linz",
}
LIST_B_MULTI: tuple[tuple[tuple[str, str], str, str], ...] = (
    (("Vor", "trag"), "Vortrag", "Vor trag"),
    (("4p", "rz7"), "April", "4p rz7"),
)
# Randzeichen am Wort. % ^ £ gehören zu den OCR-Formen und bleiben dran.
_EDGE = ".,;:!?«»\"'()[]„“”‚‘’–—*†‡/"


def token_core(token: str) -> str:
    return token.strip(_EDGE)


def ga_sort_key(path: Path) -> tuple:
    match = re.search(r"GA\s+(\d+)([a-zA-Z]?)", path.name)
    if not match:
        return (10_000, "", path.name)
    return (int(match.group(1)), match.group(2).lower(), path.name)


def parse_tounicode(data: bytes) -> tuple[dict[bytes, str], int]:
    text = data.decode("latin1")
    width = 1
    space = re.search(r"begincodespacerange(.*?)endcodespacerange", text, re.S)
    if space:
        hexes = re.findall(r"<([0-9A-Fa-f]+)>", space.group(1))
        if hexes:
            width = max(len(item) // 2 for item in hexes)

    decoded: dict[bytes, str] = {}

    def put(code: int, value: str) -> None:
        decoded[code.to_bytes(width, "big")] = value

    def dest_text(hexes: str) -> str:
        raw = bytes.fromhex(hexes)
        if len(raw) % 2:
            raw = b"\x00" + raw
        return raw.decode("utf-16-be", errors="replace")

    for block in re.findall(r"beginbfchar(.*?)endbfchar", text, re.S):
        for src, dst in re.findall(r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>", block):
            put(int(src, 16), dest_text(dst))

    for block in re.findall(r"beginbfrange(.*?)endbfrange", text, re.S):
        rest = block
        for src, end, body in re.findall(
            r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*\[(.*?)\]", block, re.S
        ):
            start = int(src, 16)
            for offset, dst in enumerate(re.findall(r"<([0-9A-Fa-f]+)>", body)):
                put(start + offset, dest_text(dst))
            rest = rest.replace(body, " ", 1)
        for src, end, dst in re.findall(
            r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>", rest
        ):
            start, last_code = int(src, 16), int(end, 16)
            raw = bytes.fromhex(dst)
            if len(raw) % 2:
                raw = b"\x00" + raw
            prefix, base = raw[:-2], int.from_bytes(raw[-2:], "big")
            for offset, code in enumerate(range(start, last_code + 1)):
                unit = (base + offset) & 0xFFFF
                value = (prefix + unit.to_bytes(2, "big")).decode("utf-16-be", errors="replace")
                put(code, value)
    return decoded, width


class FontMap:
    def __init__(self, font: pikepdf.Object, decoded: dict[bytes, str], width: int):
        self.font = font
        self.decoded = decoded
        self.width = width
        self.encoded: dict[str, bytes] = {}
        for code, value in decoded.items():
            self.encoded.setdefault(value, code)
        self.used = {int.from_bytes(code, "big") for code in decoded}
        self.added: list[str] = []

    def encode_char(self, char: str) -> bytes | None:
        found = self.encoded.get(char)
        if found is not None:
            return found
        return self._allocate(char)

    def _allocate(self, char: str) -> bytes | None:
        if self.width not in (1, 2):
            return None
        limit = 256 if self.width == 1 else 65536
        first, last = self._char_range()
        inside = range(first, last + 1)
        outside = (i for i in range(limit) if i < first or i > last)
        code = next((i for i in list(inside) + list(outside) if i not in self.used), None)
        if code is None:
            return None
        raw = code.to_bytes(self.width, "big")
        self.used.add(code)
        self.decoded[raw] = char
        self.encoded[char] = raw
        self._write_bfchar(code, char)
        self._ensure_width(code)
        self.added.append(char)
        return raw

    def _char_range(self) -> tuple[int, int]:
        if "/FirstChar" in self.font and "/LastChar" in self.font:
            return int(self.font["/FirstChar"]), int(self.font["/LastChar"])
        if self.width == 1:
            return 0, 255
        return 0, 65535

    def _write_bfchar(self, code: int, char: str) -> None:
        stream = self.font.get("/ToUnicode")
        if stream is None:
            return
        data = stream.read_bytes()
        hex_code = code.to_bytes(self.width, "big").hex().upper()
        hex_char = char.encode("utf-16-be").hex().upper()
        addition = f"\n1 beginbfchar\n<{hex_code}> <{hex_char}>\nendbfchar\n".encode("ascii")
        marker = data.rfind(b"endcmap")
        data = data + addition if marker < 0 else data[:marker] + addition + data[marker:]
        stream.write(data)

    def _ensure_width(self, code: int) -> None:
        if "/Widths" not in self.font or "/FirstChar" not in self.font:
            return
        first = int(self.font["/FirstChar"])
        last = int(self.font["/LastChar"])
        widths = [int(item) for item in self.font["/Widths"]]
        donor = next((item for item in widths if item > 100), 500)
        if first <= code <= last:
            if widths[code - first] == 0:
                widths[code - first] = donor
                self.font["/Widths"] = pikepdf.Array(widths)
            return
        if code > last:
            widths.extend([donor] * (code - last))
            self.font["/LastChar"] = code
        else:
            widths = [donor] * (first - code) + widths
            self.font["/FirstChar"] = code
        self.font["/Widths"] = pikepdf.Array(widths)


def decode_string(raw: bytes, font_map: FontMap) -> list[tuple[str, bytes]] | None:
    width = font_map.width
    if width <= 0 or len(raw) % width:
        return None
    chars: list[tuple[str, bytes]] = []
    for index in range(0, len(raw), width):
        piece = raw[index : index + width]
        value = font_map.decoded.get(piece)
        if value is None:
            return None
        chars.append((value, piece))
    return chars


def encode_word(word: str, font_map: FontMap) -> bytes | None:
    out = bytearray()
    for char in word:
        raw = font_map.encode_char(char)
        if raw is None:
            return None
        out.extend(raw)
    return bytes(out)


def replace_chars(
    parts: list[list[tuple[str, bytes]]],
    font_map: FontMap,
    part_fonts: list[FontMap] | None = None,
) -> tuple[list[bytes], Counter]:
    """Ersetzt Liste (a). Unveränderte Buchstaben behalten ihre Font-Codes."""
    owners: list[tuple[int, int]] = []
    text_parts: list[str] = []
    for part_index, part in enumerate(parts):
        for char_index, (char, _) in enumerate(part):
            owners.append((part_index, char_index))
            text_parts.append(char)
    text = "".join(text_parts)
    rebuilt: list[list[bytes]] = [[] for _ in parts]
    counts: Counter = Counter()
    cursor = 0
    while cursor < len(text):
        match = WORD_RE.match(text, cursor)
        if match:
            source = match.group(1)
            start_part = owners[cursor][0]
            target_font = part_fonts[start_part] if part_fonts else font_map
            end = match.end()
            # Komma und Punkt im Folgestück gehören zum Wort, sonst liest pdftotext sie davor.
            spanned = any(owners[index][0] != start_part for index in range(cursor, end))
            if spanned:
                while end < len(text) and text[end] in ",.;:!?":
                    end += 1
            encoded = encode_word(REPL_A[source] + text[match.end() : end], target_font)
            if encoded is None:
                part_index, char_index = owners[cursor]
                rebuilt[part_index].append(parts[part_index][char_index][1])
                cursor += 1
                continue
            part_index, _ = owners[cursor]
            rebuilt[part_index].append(encoded)
            counts[source] += 1
            cursor = end
            continue
        part_index, char_index = owners[cursor]
        rebuilt[part_index].append(parts[part_index][char_index][1])
        cursor += 1
    return [b"".join(chunks) for chunks in rebuilt], counts


def fonts_of(page: pikepdf.Object) -> pikepdf.Object | None:
    node = page
    for _ in range(8):
        if node is None:
            return None
        resources = node.get("/Resources")
        if resources is not None and "/Font" in resources and len(resources["/Font"]):
            return resources["/Font"]
        node = node.get("/Parent")
    return None


def font_map_for(font: pikepdf.Object, cache: dict) -> FontMap | None:
    key = font.objgen if font.is_indirect else id(font)
    if key not in cache:
        stream = font.get("/ToUnicode")
        if stream is None:
            cache[key] = None
        else:
            decoded, width = parse_tounicode(stream.read_bytes())
            cache[key] = FontMap(font, decoded, width) if decoded else None
    return cache[key]


def _is_number(item: object) -> bool:
    try:
        float(item)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return not isinstance(item, pikepdf.String)


def replace_show(
    operands: list, operator: str, font_map: FontMap
) -> tuple[list, Counter]:
    if operator == "TJ":
        array = operands[0]
        strings: list[bytes] = []
        structure: list[tuple[str, object]] = []
        joinable = True
        for item in array:
            if isinstance(item, pikepdf.String):
                structure.append(("s", len(strings)))
                strings.append(bytes(item))
            elif _is_number(item):
                structure.append(("n", item))
            else:
                joinable = False
                break
        if not joinable:
            return operands, Counter()
        decoded_parts: list[list[tuple[str, bytes]]] = []
        for raw in strings:
            decoded = decode_string(raw, font_map)
            if decoded is None:
                return operands, Counter()
            decoded_parts.append(decoded)
        new_strings, counts = replace_chars(decoded_parts, font_map)
        if not counts:
            return operands, counts
        new_array = pikepdf.Array()
        for kind, value in structure:
            if kind == "n":
                new_array.append(value)
            else:
                new_array.append(pikepdf.String(new_strings[int(value)]))  # type: ignore[arg-type]
        return [new_array], counts

    index = 0 if operator in ("Tj", "'") else len(operands) - 1
    raw = bytes(operands[index])
    decoded = decode_string(raw, font_map)
    if decoded is None:
        return operands, Counter()
    new_strings, counts = replace_chars([decoded], font_map)
    if not counts:
        return operands, counts
    updated = list(operands)
    updated[index] = pikepdf.String(new_strings[0])
    return updated, counts


def decode_show(operands: list, operator: str, font_map: FontMap):
    """Teile eines Textbefehls. Rückgabe: (Buchstaben-Teile, neuer Operand)."""
    if operator == "TJ":
        array = operands[0]
        strings: list[bytes] = []
        structure: list[tuple[str, object]] = []
        for item in array:
            if isinstance(item, pikepdf.String):
                structure.append(("s", len(strings)))
                strings.append(bytes(item))
            elif _is_number(item):
                structure.append(("n", item))
            else:
                return None
        parts: list[list[tuple[str, bytes]]] = []
        for raw in strings:
            decoded = decode_string(raw, font_map)
            if decoded is None:
                return None
            parts.append(decoded)

        def rebuild(new_strings: list[bytes]) -> list:
            array_out = pikepdf.Array()
            for kind, value in structure:
                if kind == "n":
                    array_out.append(value)
                else:
                    array_out.append(pikepdf.String(new_strings[int(value)]))  # type: ignore[arg-type]
            return [array_out]

        return parts, rebuild

    index = 0 if operator in ("Tj", "'") else len(operands) - 1
    if index < 0 or index >= len(operands):
        return None
    decoded = decode_string(bytes(operands[index]), font_map)
    if decoded is None:
        return None
    original = list(operands)

    def rebuild(new_strings: list[bytes]) -> list:
        updated = list(original)
        updated[index] = pikepdf.String(new_strings[0])
        return updated

    return [decoded], rebuild


def same_word_piece(previous: dict, piece: dict) -> bool:
    """Folgestück desselben Wortes: gleiche Zeile, kein Wortabstand dazwischen."""
    if previous["bt"] != piece["bt"]:
        return False
    if previous["y"] is None or piece["y"] is None or previous["x"] is None or piece["x"] is None:
        return False
    if abs(previous["y"] - piece["y"]) > 0.8 or piece["x"] < previous["x"] - 1.0:
        return False
    size = previous["size"] or 10.0
    estimated = max(len(previous["text"]), 1) * size * 0.5
    gap = piece["x"] - (previous["x"] + estimated)
    return gap < max(size * 0.45, 4.0)


def replace_page(pdf: pikepdf.Pdf, page: pikepdf.Page, cache: dict) -> Counter:
    fonts = fonts_of(page)
    if fonts is None or page.get("/Contents") is None:
        return Counter()
    font_name: str | None = None
    font_size: float | None = None
    pos_x: float | None = None
    pos_y: float | None = None
    stack: list[tuple[str | None, float | None, float | None, float | None]] = []
    text_block = 0
    operations: list[tuple] = []
    pieces: list[dict] = []
    for operands, operator in parse_content_stream(page):
        name = str(operator)
        op_index = len(operations)
        if name == "q":
            stack.append((font_name, font_size, pos_x, pos_y))
        elif name == "Q" and stack:
            font_name, font_size, pos_x, pos_y = stack.pop()
        elif name == "BT":
            text_block += 1
        elif name == "ET":
            text_block += 1
        elif name == "Tf" and operands:
            font_name = str(operands[0])
            font_size = float(operands[1])
        elif name == "Tm" and len(operands) >= 6:
            pos_x = float(operands[4])
            pos_y = float(operands[5])
        elif name in ("Td", "TD", "T*"):
            pos_x = pos_y = None
        elif name in ("Tj", "TJ", "'", '"') and font_name:
            if name == "'":
                pos_x = pos_y = None
            key = font_name if font_name.startswith("/") else "/" + font_name
            font = fonts.get(key)
            font_map = font_map_for(font, cache) if font is not None else None
            if font_map is not None:
                decoded = decode_show(list(operands), name, font_map)
                if decoded is not None:
                    parts, rebuild = decoded
                    text = "".join(char for part in parts for char, _raw in part)
                    pieces.append(
                        {
                            "index": op_index,
                            "parts": parts,
                            "rebuild": rebuild,
                            "text": text,
                            "x": pos_x,
                            "y": pos_y,
                            "size": font_size,
                            "font_name": font_name,
                            "font_map": font_map,
                            "bt": text_block,
                        }
                    )
        operations.append((list(operands), operator))

    counts: Counter = Counter()
    runs: list[list[dict]] = []
    for piece in pieces:
        if runs and same_word_piece(runs[-1][-1], piece):
            runs[-1].append(piece)
        else:
            runs.append([piece])

    changed = False
    for run in runs:
        flat: list[list[tuple[str, bytes]]] = []
        part_fonts: list[FontMap] = []
        spans: list[tuple[dict, int]] = []
        for piece in run:
            spans.append((piece, len(flat)))
            flat.extend(piece["parts"])
            part_fonts.extend([piece["font_map"]] * len(piece["parts"]))
        new_strings, found = replace_chars(flat, run[0]["font_map"], part_fonts)
        if not found:
            continue
        counts.update(found)
        changed = True
        for piece, start in spans:
            width = len(piece["parts"])
            operations[piece["index"]] = (
                piece["rebuild"](new_strings[start : start + width]),
                operations[piece["index"]][1],
            )
    if changed:
        page["/Contents"] = pdf.make_stream(unparse_content_stream(operations))
    return counts


def replace_pdf(path: Path, dry_run: bool) -> Counter:
    pdf = pikepdf.open(path)
    cache: dict = {}
    counts: Counter = Counter()
    closed = False
    try:
        for page in pdf.pages:
            counts.update(replace_page(pdf, page, cache))
        added = sorted({char for font_map in cache.values() if font_map for char in font_map.added})
        if added:
            print(f"  neue Codes: {''.join(added)}", flush=True)
        if counts and not dry_run:
            temporary = path.with_name(path.name + ".tmp-ocr")
            try:
                pdf.save(
                    temporary,
                    recompress_flate=False,
                    normalize_content=False,
                    object_stream_mode=pikepdf.ObjectStreamMode.preserve,
                )
            except Exception:
                temporary.unlink(missing_ok=True)
                raise
            pdf.close()
            closed = True
            os.replace(temporary, path)
    finally:
        if not closed:
            pdf.close()
    return counts


_BBOX_WORD = re.compile(
    r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">(.*?)</word>',
    re.S,
)


def bbox_words(path: Path, page_number: int) -> list[dict]:
    result = subprocess.run(
        [
            "pdftotext",
            "-bbox-layout",
            "-f",
            str(page_number),
            "-l",
            str(page_number),
            "-enc",
            "UTF-8",
            str(path),
            "-",
        ],
        check=True,
        capture_output=True,
    )
    text = result.stdout.decode("utf-8", errors="replace")
    words = []
    for x_min, y_min, x_max, y_max, raw in _BBOX_WORD.findall(text):
        word = (
            raw.replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", '"')
            .replace("&apos;", "'")
        )
        words.append(
            {
                "text": word,
                "xMin": float(x_min),
                "yMin": float(y_min),
                "xMax": float(x_max),
                "yMax": float(y_max),
            }
        )
    return words


def context_window(tokens: list[str], index: int, span: int) -> str:
    end = index + span + 2
    return " ".join(tokens[max(0, index - 2) : end])


def normalize_context_token(token: str) -> str:
    """Nachbarsworte aus Liste (a) stehen im PDF schon in der neuen Schreibung."""
    core = token_core(token)
    if core not in REPL_A:
        return token
    at = token.find(core)
    if at < 0:
        return token
    return token[:at] + REPL_A[core] + token[at + len(core) :]


def normalize_context(context: str) -> str:
    return " ".join(normalize_context_token(token) for token in context.split(" "))


_DOT_LEADERS = re.compile(r"\.{2,}|…+")


def token_parts(token: str) -> list[str]:
    """Kern eines Wortes. Punktreihen in Inhaltsverzeichnissen trennen zwei Tokens."""
    token = normalize_context_token(token)
    parts = []
    for bit in _DOT_LEADERS.split(token):
        bit = bit.strip()
        if not bit:
            continue
        core = token_core(bit)
        parts.append(core if core else bit)
    return parts


def retarget_context(context: str, form: str, target: str) -> str | None:
    tokens = context.split(" ")
    parts = form.split(" ")
    cores = [token_core(token) for token in tokens]
    for index in range(len(tokens) - len(parts) + 1):
        if cores[index : index + len(parts)] == parts:
            return " ".join(tokens[:index] + [target] + tokens[index + len(parts) :])
    return None


def line_token_set(words: list[dict], index: int, dy: float = 30) -> set[str]:
    """Worte auf derselben Zeile und der Nachbarzeile, plus zusammengefügte Anfangsbuchstaben."""
    y = words[index]["yMin"]
    selected = [word for word in words if abs(word["yMin"] - y) <= dy]
    got: set[str] = set()
    for word in selected:
        got.update(token_parts(word["text"]))
    ordered = sorted(selected, key=lambda word: word["xMin"])
    for left_index, word in enumerate(ordered):
        bits = token_parts(word["text"])
        if len(bits) != 1 or len(bits[0]) > 2:
            continue
        for nxt in ordered[left_index + 1 :]:
            if nxt["xMin"] + 1 < word["xMin"]:
                continue
            if abs(nxt["yMin"] - word["yMin"]) > 4 and abs(nxt["yMax"] - word["yMax"]) > 4:
                continue
            if nxt["xMin"] - word["xMax"] > 8:
                break
            nxt_bits = token_parts(nxt["text"])
            if nxt_bits:
                got.add(bits[0] + nxt_bits[0])
            break
    return got


def find_context_hit(words: list[dict], form: str, context: str, start_at: int) -> int | None:
    tokens = [word["text"] for word in words]
    cores = [token_core(token) for token in tokens]
    parts = form.split(" ")
    span = len(parts)
    expected = normalize_context(context)
    candidates = []
    for index in range(start_at, len(tokens) - span + 1):
        if cores[index : index + span] != parts:
            continue
        if normalize_context(context_window(tokens, index, span)) == expected:
            return index
        candidates.append(index)
    if not candidates:
        return None
    ctx_set: set[str] = set()
    for token in context.split(" "):
        ctx_set.update(token_parts(token))
    scored = [(len(ctx_set & line_token_set(words, index)), index) for index in candidates]
    best = max(score for score, _index in scored)
    # Ein einziger Treffer auf der Seite: Datum und Ort reichen, auch wenn der Titel
    # in der Wortbox anders zerlegt ist. Mehrere Treffer brauchen fast den ganzen Kontext.
    need = 3 if len(candidates) == 1 else max(2, len(ctx_set) - 1)
    if best < need:
        return None
    chosen = [index for score, index in scored if score == best]
    if len(chosen) == 1 or best == len(ctx_set):
        return chosen[0]
    return None


def piece_overlaps_box(
    piece: dict, x0: float, x1: float, y0: float, y1: float, origin_top: float
) -> bool:
    """y0/y1 sind die bbox-Werte von pdftotext, Ursprung oben. Die Textebene misst von unten."""
    if piece["x"] is None or piece["y"] is None:
        return False
    pdf_bottom = origin_top - y1
    pdf_top = origin_top - y0
    if piece["y"] < pdf_bottom - 3 or piece["y"] > pdf_top + 3:
        return False
    size = piece["size"] or 10.0
    end = piece["x"] + max(len(piece["text"]), 1) * size * 0.55
    return not (end < x0 - 4 or piece["x"] > x1 + 4)


def form_pattern(form: str) -> re.Pattern[str]:
    if " " in form:
        body = r"\s*".join(re.escape(part) for part in form.split(" "))
        return re.compile(r"(?<!\w)" + body + r"(?!\w)")
    return re.compile(r"(?<!\w)" + re.escape(form) + r"(?!\w)")


def match_x(pieces: list[dict], offsets: list[int], start: int) -> float | None:
    for piece, offset in zip(pieces, offsets):
        if piece["x"] is None:
            continue
        if offset <= start < offset + len(piece["text"]):
            size = piece["size"] or 10.0
            return piece["x"] + (start - offset) * size * 0.45
    return None


def locate_form(
    pieces: list[dict], form: str, x0: float, x1: float, y0: float, y1: float, origin_top: float
):
    """Zeichenbereich der Form in den Stücken, die die Wortbox schneiden."""
    hits = [
        index
        for index, piece in enumerate(pieces)
        if piece_overlaps_box(piece, x0, x1, y0, y1, origin_top)
    ]
    if not hits:
        return None
    first, last = hits[0], hits[-1]
    chosen = pieces[first : last + 1]
    text = ""
    offsets = []
    for piece in chosen:
        offsets.append(len(text))
        text += piece["text"]
    pattern = form_pattern(form)
    found = list(pattern.finditer(text))
    if not found:
        return None
    target_x = (x0 + x1) / 2
    best = min(found, key=lambda match: abs((match_x(chosen, offsets, match.start()) or target_x) - target_x))
    return chosen, best.start(), best.end()


def apply_located_spans(page_pieces: list[dict], spans: list[tuple[list[dict], int, int, str]]) -> int:
    """Ersetzt mehrere Treffer. Höhere Position zuerst, Ausgangstext bleibt die Basis."""
    used = {id(piece) for chosen, _start, _end, _target in spans for piece in chosen}
    pieces = [piece for piece in page_pieces if id(piece) in used]
    items = []
    piece_item_ranges: dict[int, tuple[int, int]] = {}
    for piece in pieces:
        begin = len(items)
        for part_index, part in enumerate(piece["parts"]):
            for char, raw in part:
                items.append(["c", char, raw, piece, part_index])
        piece_item_ranges[id(piece)] = (begin, len(items))
    absolute = []
    for chosen, start, end, target in spans:
        base = piece_item_ranges[id(chosen[0])][0]
        absolute.append((base + start, base + end, target))
    done = 0
    for start, end, target in sorted(absolute, key=lambda item: -item[0]):
        if start < 0 or end > len(items) or start >= end:
            continue
        encoded = encode_word(target, items[start][3]["font_map"])
        if encoded is None:
            continue
        items[start:end] = [["b", encoded, None, items[start][3], items[start][4]]]
        done += 1
    grouped: dict[int, list[list[bytes]]] = {}
    for item in items:
        piece = item[3]
        part_index = item[4]
        bucket = grouped.setdefault(id(piece), [[] for _ in piece["parts"]])
        bucket[part_index].append(item[1] if item[0] == "b" else item[2])
    for piece in pieces:
        bucket = grouped.get(id(piece))
        if bucket is None:
            # Alle Zeichen dieses Stücks stecken in der Ersetzung eines früheren Stücks.
            piece["new_strings"] = [b"" for _ in piece["parts"]]
            continue
        piece["new_strings"] = [b"".join(chunks) for chunks in bucket]
    return done


def page_text_pieces(pdf: pikepdf.Pdf, page: pikepdf.Page, cache: dict):
    fonts = fonts_of(page)
    if fonts is None or page.get("/Contents") is None:
        return [], []
    font_name: str | None = None
    font_size: float | None = None
    pos_x: float | None = None
    pos_y: float | None = None
    tm = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    leading = 0.0
    stack: list[tuple] = []
    text_block = 0
    operations: list[tuple] = []
    pieces: list[dict] = []

    def apply_move(tx: float, ty: float) -> None:
        nonlocal tm, pos_x, pos_y
        a, b, c, d, e, f = tm
        e2 = a * tx + c * ty + e
        f2 = b * tx + d * ty + f
        tm = [a, b, c, d, e2, f2]
        pos_x, pos_y = e2, f2

    for operands, operator in parse_content_stream(page):
        name = str(operator)
        op_index = len(operations)
        if name == "q":
            stack.append((font_name, font_size, pos_x, pos_y, tm[:], leading))
        elif name == "Q" and stack:
            font_name, font_size, pos_x, pos_y, tm, leading = stack.pop()
        elif name == "BT":
            text_block += 1
            tm = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
            pos_x, pos_y = 0.0, 0.0
        elif name == "ET":
            text_block += 1
        elif name == "Tf" and operands:
            font_name = str(operands[0])
            font_size = float(operands[1])
        elif name == "Tm" and len(operands) >= 6:
            tm = [float(operands[i]) for i in range(6)]
            pos_x, pos_y = tm[4], tm[5]
        elif name == "TL" and operands:
            leading = float(operands[0])
        elif name in ("Td", "TD") and len(operands) >= 2:
            apply_move(float(operands[0]), float(operands[1]))
            if name == "TD":
                leading = -float(operands[1])
        elif name == "T*":
            apply_move(0.0, -leading)
        elif name in ("Tj", "TJ", "'", '"') and font_name:
            if name in ("'", '"'):
                apply_move(0.0, -leading)
            key = font_name if font_name.startswith("/") else "/" + font_name
            font = fonts.get(key)
            font_map = font_map_for(font, cache) if font is not None else None
            if font_map is not None:
                decoded = decode_show(list(operands), name, font_map)
                if decoded is not None:
                    parts, rebuild = decoded
                    text = "".join(char for part in parts for char, _raw in part)
                    pieces.append(
                        {
                            "index": op_index,
                            "parts": parts,
                            "rebuild": rebuild,
                            "text": text,
                            "x": pos_x,
                            "y": pos_y,
                            "size": font_size,
                            "font_name": font_name,
                            "font_map": font_map,
                            "bt": text_block,
                        }
                    )
        operations.append((list(operands), operator))
    return operations, pieces


def load_b_rows(path: Path) -> list[tuple[str, str, str, int, str]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 5:
            raise SystemExit(f"{path}:{line_number}: fünf Felder erwartet, {len(parts)} gefunden")
        form, target, filename, page, context = parts
        rows.append((form, target, filename, int(page), context))
    return rows


def apply_b_file(context_path: Path, directory: Path, only: list[str], dry_run: bool) -> None:
    rows = load_b_rows(context_path)
    if only:
        wanted = set(only)
        rows = [row for row in rows if row[2] in wanted]
    by_file: dict[str, list[tuple[str, str, int, str]]] = {}
    for form, target, filename, page, context in rows:
        by_file.setdefault(filename, []).append((form, target, page, context))
    print(f"{len(rows)} Zeilen in {len(by_file)} Dateien", flush=True)
    replaced = 0
    missed: list[str] = []
    for filename in sorted(by_file, key=lambda name: ga_sort_key(Path(name))):
        path = directory / filename
        if not path.is_file():
            for form, _target, page, context in by_file[filename]:
                missed.append(f"{filename} S.{page} {form}: Datei fehlt ({context})")
            continue
        pdf = pikepdf.open(path)
        cache: dict = {}
        changed = False
        file_hits = 0
        try:
            pages: dict[int, list[tuple[str, str, str]]] = {}
            for form, target, page, context in by_file[filename]:
                pages.setdefault(page, []).append((form, target, context))
            for page_number, instructions in pages.items():
                if page_number < 1 or page_number > len(pdf.pages):
                    for form, _target, context in instructions:
                        missed.append(f"{filename} S.{page_number} {form}: Seite fehlt ({context})")
                    continue
                words = bbox_words(path, page_number)
                cursor = 0
                located = []
                for form, target, context in instructions:
                    hit = find_context_hit(words, form, context, cursor)
                    if hit is None:
                        swapped = retarget_context(context, form, target)
                        done_hit = (
                            find_context_hit(words, target, swapped, cursor) if swapped is not None else None
                        )
                        if done_hit is not None:
                            cursor = done_hit + len(target.split(" "))
                            continue
                        missed.append(f"{filename} S.{page_number} {form}: Kontext nicht gefunden ({context})")
                        continue
                    span = len(form.split(" "))
                    cursor = hit + span
                    box_words = words[hit : hit + span]
                    located.append((form, target, box_words))
                if not located:
                    continue
                operations, pieces = page_text_pieces(pdf, pdf.pages[page_number - 1], cache)
                spans = []
                for form, target, box_words in located:
                    x0 = min(word["xMin"] for word in box_words)
                    x1 = max(word["xMax"] for word in box_words)
                    y0 = min(word["yMin"] for word in box_words)
                    y1 = max(word["yMax"] for word in box_words)
                    origin_top = float(pdf.pages[page_number - 1].mediabox[3])
                    found = locate_form(pieces, form, x0, x1, y0, y1, origin_top)
                    if found is None:
                        missed.append(f"{filename} S.{page_number} {form}: Textstück nicht gefunden")
                        continue
                    chosen, start, end = found
                    spans.append((chosen, start, end, target))
                if not spans or dry_run:
                    file_hits += len(spans)
                    replaced += len(spans)
                    continue
                done = apply_located_spans(pieces, spans)
                file_hits += done
                replaced += done
                changed = changed or done > 0
                if done < len(spans):
                    missed.append(f"{filename} S.{page_number}: {len(spans) - done} Kodierungen fehlgeschlagen")
                for piece in pieces:
                    if "new_strings" not in piece:
                        continue
                    operations[piece["index"]] = (
                        piece["rebuild"](piece["new_strings"]),
                        operations[piece["index"]][1],
                    )
                pdf.pages[page_number - 1]["/Contents"] = pdf.make_stream(unparse_content_stream(operations))
            if changed and not dry_run:
                temporary = path.with_name(path.name + ".tmp-ocr")
                try:
                    pdf.save(
                        temporary,
                        recompress_flate=False,
                        normalize_content=False,
                        object_stream_mode=pikepdf.ObjectStreamMode.preserve,
                    )
                except Exception:
                    temporary.unlink(missing_ok=True)
                    raise
                pdf.close()
                os.replace(temporary, path)
                pdf = None
        finally:
            if pdf is not None:
                pdf.close()
        print(f"{filename}  {file_hits}", flush=True)
    print(f"Fertig. Ersetzt: {replaced}. Nicht gefunden: {len(missed)}.", flush=True)
    for line in missed:
        print(f"  {line}", flush=True)


def read_layout(path: Path) -> str:
    result = subprocess.run(
        ["pdftotext", "-layout", "-enc", "UTF-8", str(path), "-"],
        check=True,
        capture_output=True,
    )
    return result.stdout.decode("utf-8", errors="replace")


def context_lines(name: str, layout: str) -> list[str]:
    lines: list[str] = []
    pages = layout.split("\f")
    if pages and pages[-1] == "":
        pages.pop()
    for page_number, page in enumerate(pages, start=1):
        tokens = page.split()
        index = 0
        while index < len(tokens):
            if index + 1 < len(tokens):
                pair = (token_core(tokens[index]), token_core(tokens[index + 1]))
                multi = next((item for item in LIST_B_MULTI if item[0] == pair), None)
                if multi:
                    _, target, form = multi
                    window = tokens[max(0, index - 2) : index + 4]
                    lines.append(
                        "\t".join((form, target, name, str(page_number), " ".join(window)))
                    )
                    index += 2
                    continue
            core = token_core(tokens[index])
            target = LIST_B_SINGLE.get(core)
            if target:
                window = tokens[max(0, index - 2) : index + 3]
                lines.append(
                    "\t".join((core, target, name, str(page_number), " ".join(window)))
                )
            index += 1
    return lines


def load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def pdfs_in(directory: Path, only: list[str]) -> list[Path]:
    files = [path for path in directory.glob("GA *.pdf") if not path.name.startswith("._")]
    files.sort(key=ga_sort_key)
    if only:
        wanted = set(only)
        files = [path for path in files if path.name in wanted]
        missing = wanted.difference(path.name for path in files)
        if missing:
            raise SystemExit(f"Nicht gefunden: {', '.join(sorted(missing))}")
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Liste (a) in GA-PDFs ersetzen, Liste (b) protokollieren.")
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--context", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--no-context", action="store_true")
    parser.add_argument("--apply-b", action="store_true", help="Redigierte Liste (b) in die PDFs schreiben")
    args = parser.parse_args()

    if args.apply_b:
        directory = args.dir.expanduser().resolve()
        if not args.dry_run and directory == DEFAULT_DIR and not BACKUP_DIR.is_dir():
            raise SystemExit(f"Backup fehlt, Abbruch: {BACKUP_DIR}")
        apply_b_file(args.context, directory, args.only, args.dry_run)
        return

    directory = args.dir.expanduser().resolve()
    if not directory.is_dir():
        raise SystemExit(f"Ordner fehlt: {directory}")
    if (
        not args.dry_run
        and directory == DEFAULT_DIR
        and not BACKUP_DIR.is_dir()
    ):
        raise SystemExit(f"Backup fehlt, Abbruch: {BACKUP_DIR}")

    sources = {src for src, _ in LIST_A}
    if len(sources) != len(LIST_A):
        raise SystemExit("Liste (a) enthält doppelte Formen.")

    files = pdfs_in(directory, args.only)
    done_path = args.context.with_suffix(args.context.suffix + ".done")
    done = set() if args.dry_run or args.only else load_done(done_path)
    write_context = not args.dry_run and not args.no_context
    context_file = None
    if write_context:
        args.context.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if done else "w"
        context_file = args.context.open(mode, encoding="utf-8")

    print(f"{len(files)} PDFs in {directory}", flush=True)
    if done:
        print(f"{len(done)} schon erledigt, werden übersprungen", flush=True)

    changed_files = 0
    total = Counter()
    context_count = 0
    try:
        for path in files:
            if path.name in done:
                continue
            started = time.perf_counter()
            try:
                layout = "" if args.no_context else read_layout(path)
                text_counts = Counter(WORD_RE.findall(layout)) if layout else Counter()
                counts = replace_pdf(path, args.dry_run)
            except Exception:
                print(f"{path.name}  FEHLER", flush=True)
                traceback.print_exc()
                continue
            if counts:
                changed_files += 1
                total.update(counts)
            parts = [f"{src} {counts[src]}" for src, _ in LIST_A if counts[src]]
            missed = [
                f"{src} {text_counts[src] - counts[src]}"
                for src in text_counts
                if text_counts[src] > counts[src]
            ]
            elapsed = time.perf_counter() - started
            label = ", ".join(parts) if parts else "keine Ersetzung"
            print(f"{path.name}  {elapsed:.1f}s  {label}", flush=True)
            if missed:
                print(f"  in pdftotext, nicht in einer Zeichenkette: {', '.join(missed)}", flush=True)
            if write_context and context_file is not None and layout:
                rows = context_lines(path.name, layout)
                for row in rows:
                    context_file.write(row + "\n")
                context_file.flush()
                context_count += len(rows)
            if not args.dry_run and not args.only:
                with done_path.open("a", encoding="utf-8") as handle:
                    handle.write(path.name + "\n")
    finally:
        if context_file is not None:
            context_file.close()

    print(
        f"Fertig. Geändert: {changed_files}. Ersetzungen: {sum(total.values())}. Kontextzeilen: {context_count}.",
        flush=True,
    )
    for src, _dst in LIST_A:
        if total[src]:
            print(f"  {src} -> {REPL_A[src]}  {total[src]}", flush=True)


if __name__ == "__main__":
    main()
