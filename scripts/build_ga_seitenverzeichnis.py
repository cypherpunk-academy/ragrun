#!/usr/bin/env python3
"""Seitenverzeichnisse für GA-Bände ab GA 064.

Liest die PDFs in ragkeep/ga, die Vorträge des Steiner-Katalogs und lässt
DeepSeek die PDF-Seite jedes Vortrags bestimmen. Fehlen nach dem ersten Lauf
Katalogvorträge, sucht ein zweiter Lauf gezielt nach ihnen und markiert das
Ergebnis.

Aufruf (im ragrun-venv):

    .venv/bin/python scripts/build_ga_seitenverzeichnis.py --only 64
    .venv/bin/python scripts/build_ga_seitenverzeichnis.py
    .venv/bin/python scripts/build_ga_seitenverzeichnis.py --force --only 68a

Benötigt pdftotext (poppler) und DEEPSEEK_API_KEY oder DEEPSEEK_REST_API_KEY.
Vorhandene Dateien werden übersprungen, ausser mit --force.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import yaml

RED = "\033[31m"
GREEN = "\033[32m"
RESET = "\033[0m"


def color(text: str, ansi: str, stream: object = sys.stdout) -> str:
    isatty = getattr(stream, "isatty", lambda: False)()
    if not isatty:
        return text
    return f"{ansi}{text}{RESET}"


RAGRUN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GA_DIR = RAGRUN_ROOT / "ragkeep" / "ga"
DEFAULT_CATALOG = (
    RAGRUN_ROOT.parent / "steineroriginals" / "rudolf-steiner-ga-lecture-catalog.yaml"
)

MONTHS = [
    "Januar",
    "Februar",
    "März",
    "April",
    "Mai",
    "Juni",
    "Juli",
    "August",
    "September",
    "Oktober",
    "November",
    "Dezember",
]
# „Mär%“ und „Mar£“: das z in März wurde als % oder £ gelesen.
# „4p rz7“ ist April: A als 4, i als z, l als 7.
MONTH_PATTERN = (
    r"Januar|Febtuar|Februar|Mär%|Mär\^|Mär£|Mar%|Mar£|März|Maerz|Marz|4\s*p\s*r\s*z\s*7|April|Mai|Juni|Juli|Augast|August|"
    r"September|Oktober|November|Dezember"
)
# Punkt, Komma, Bindestrich oder fehlendes Satzzeichen nach dem Tag:
# „16. November“, „16, November“, „16 November“, „13- Februar“.
# Die Ziffer 1 steht im OCR oft als Buchstabe l, I oder i, die 0 als o: „l.März 1917“, „io. Oktober 1918“.
# „26* Juni“: der Punkt nach dem Tag wurde als Stern gelesen.
# „8: September“: der Punkt nach dem Tag wurde als Doppelpunkt gelesen.
# „12» August“: der Punkt nach dem Tag wurde als » gelesen.
DAY_SEP = r"\s*[,.\-\*:»«]?\s*"
# „2 2. August“: zwischen den beiden Ziffern des Tages steht ein Leerzeichen.
# „J. Juni“: die 5 wurde als J gelesen. „2). April“: die 3 wurde als Klammer gelesen.
# „z. Febtuar“: die 3 wurde als z gelesen, das r in Februar als t.
# „!. Januar“: die 1 wurde als Ausrufezeichen gelesen.
# „//. November“: die 1 wurde als Schrägstrich gelesen.
# „n. September“: die 11 wurde als n gelesen.
# „Li“ ist 21: L die 2, i die 1. „ZI“ ist 31: Z die 3, I die 1.
# „if“ ist 15: i die 1, f die 5. „u“ ist 11, wie „n“.
# „1$“ ist 15: $ die 5.
DAY_CHARS = r"0-9lIi|oO/ZzLfFjJ$"
DAY_TOKEN = rf"[{DAY_CHARS}]\s+[{DAY_CHARS}]|[{DAY_CHARS}]\)|[{DAY_CHARS}]{{1,2}}|[Jj]|!|[Nn]|[Uu]"
# OCR setzt oft ein Apostroph vor den Monat: „14. 'November 1907“.
# „76. ,4p rz7“: vor dem zerschlagenen Monatsnamen steht ein Komma.
# „29. .April“: vor dem Monat steht ein überzähliger Punkt.
MONTH_NOISE = r"[.,''`´‘’]*\s*"
# Ganze Zeile ist nur „Ort, 29. Oktober 1914“, „Regensburg 17. Dezember 1905“
# oder „Düsseldorf\ 11. Februar 1906“. Datumsbereiche mit „bis“ bleiben außen vor.
# „Lin%“: das z am Wortende wurde als % gelesen.
PLACE_HEAD = r"[A-ZÄÖÜ][A-Za-zÄÖÜäöüß.'’\-\s()%]*?"
# „Paris> 2. Juni“, „Leipzig} 4. Juli“, „Basel» 16. November 1907“
# und „Leipzig* 5- November 1911“, „London^ 7. Mai 1913“:
# das Komma wurde als >, }, », * oder ^ gelesen.
PLACE_GAP = rf"(?:\s*[,\\;>}}»«*^]\s*|\s+)"
# „Hannover, den 18. Februar 1915“ und „Berlin am 3. April 1905“
DATE_INTRO = r"(?:(?:den|am)\s+)?"
# „20. Februar, 1917“: zwischen Monat und Jahr steht ein Komma.
MONTH_YEAR_GAP = r"(?:\s*,\s*|\s+)"
# „Berlin, 18. August (1905?)“, „16. August 1904 (?)“, „30. September 1904 (Notizen)“
# und „Berlin, 28. September 1905*“. Der Stern ist eine Fußnote und ändert das Datum nicht.
# „22. Oktober 1906, vormittags“ und „24. Februar 1908 (Kurznotizen)“.
YEAR_TAIL = (
    # „191$“ ist 1915, „19/j“ ist 1915, „19//“ ist 1911 und bleibt über die Ziffer 7 der 1917.
    r"(?:\(\s*)?([0-9$/jJlIi|]{4})\s*\??\s*\)?(?:\s*\(\s*\?\s*\))?"
    r"(?:\s*\(\s*(?:(?:Kurz)?notizen|Autoreferat)\s*\))?"
    # „19. Januar 1922'''“ und „26. Juni 1906^“: Apostroph und Zirkumflex sind Fußnoten.
    r"(?:\s*[*'´′'`^]+)?\s*\.?(?:[*'´′'`^]+)?"
    # „Den Haag, 21. März 1913 (Karfreitag)“: der Festtag in Klammern ändert das Datum nicht.
    r"(?:\s*\(\s*(?:Karfreitag|Ostern|Ostersonntag|Ostermontag|Karsamstag|Pfingsten|Pfingstsonntag|Pfingstmontag)\s*\))?"
    r"(?:,\s*(?:vormittags|nachmittags|abends))?"
    # „Stuttgart, 4. Oktober 1921 (vormittags)“: die Tageszeit steht in Klammern.
    r"(?:\s*\(\s*(?:morgens|vormittags|mittags|nachmittags|abends)\s*\))?"
    # „Basel, 11. Januar 1924, 11 Uhr vormittags“ und
    # „Dornach, 30. Dezember 1923, 8 Uhr 30 morgens im Glashaus“.
    r"(?:,\s*\d{1,2}\s*Uhr(?:\s*\d{1,2})?(?:\s*(?:morgens|vormittags|mittags|nachmittags|abends))?(?:\s+im\s+[A-Za-zÄÖÜäöüß]+)?)?"
    # „Berlin, 7. Januar 1908 (Fortsetzung vom 29. November 1907)“:
    # der Tag in der Klammer ist kein zweiter Beginn.
    r"(?:\s*\(\s*Fortsetzung\s+vom\s+[^)]+\))?"
    # „1. Januar 1912, Neujahrsfeier“: der Anlass hinter dem Jahr ändert das Datum nicht.
    r"(?:,\s*[A-Za-zÄÖÜäöüß][^,\d]{0,70})?"
    # „4. Juli 1903, Autoreferat von Rudolf Steiner,“: das Komma hinter dem Anlass bleibt.
    r",?\s*"
)
# „öffentlicher Vortrag Berlin, …“, „nicht öffentlicher Vortrag; Leipzig, …“
# und „Erster Vortrag, Paris, 25. Mai 1906“. Hinter „Vortrag“ steht oft ., > oder }.
ORDINAL = (
    "Erster|Zweiter|Dritter|Vierter|Fünfter|Fuenfter|Sechster|"
    "Siebenter|Siebter|Achter|Neunter|Zehnter|Elfter|Zwölfter|"
    "Dreizehnter|Vierzehnter|Fünfzehnter|Fuenfzehnter|Sechzehnter|"
    "Siebzehnter|Achtzehnter|Neunzehnter|Zwanzigster"
)
LECTURE_PREFIX = (
    rf"(?:(?:(?:nicht\s+)?öffentlicher\s+Vortrag\s*[,;:]?\s+"
    # „Zweiter Vortrag Stuttgart, …“: das Komma nach Vortrag fehlt.
    rf"|(?:{ORDINAL})\s+Vortrag\s*[,.;>]*\s+))?"
)
HEADING_LINE_RE = re.compile(
    rf"^{LECTURE_PREFIX}(?:({PLACE_HEAD}){PLACE_GAP}{DATE_INTRO})?"
    rf"({DAY_TOKEN}){DAY_SEP}{MONTH_NOISE}({MONTH_PATTERN}){MONTH_YEAR_GAP}{YEAR_TAIL}$",
    re.IGNORECASE,
)
# „Dornach, 10. September“: die Jahreszahl fehlt. Jahr 0 heißt, der Katalog liefert sie.
YEARLESS_HEADING_RE = re.compile(
    rf"^{LECTURE_PREFIX}({PLACE_HEAD}){PLACE_GAP}{DATE_INTRO}"
    rf"({DAY_TOKEN}){DAY_SEP}{MONTH_NOISE}({MONTH_PATTERN})\s*\.?\s*$",
    re.IGNORECASE,
)
SEASON_PATTERN = r"Frühling|Fruehling|Sommer|Herbst|Winter"
SEASON_LINE_RE = re.compile(
    rf"^{LECTURE_PREFIX}(?:({PLACE_HEAD}){PLACE_GAP})?({SEASON_PATTERN}){MONTH_YEAR_GAP}{YEAR_TAIL}$",
    re.IGNORECASE,
)
SEASON_WORD = {
    "frühling": "frühling",
    "fruehling": "frühling",
    "sommer": "sommer",
    "herbst": "herbst",
    "winter": "winter",
}
# „Berlin, 27. Februar und 15. März 1906“: zwei Monate, beide Tage.
TWO_MONTHS_RE = re.compile(
    rf"^{LECTURE_PREFIX}({PLACE_HEAD}){PLACE_GAP}"
    rf"({DAY_TOKEN})\.\s+({MONTH_PATTERN})\s+und\s+"
    rf"({DAY_TOKEN})\.\s+({MONTH_PATTERN}){MONTH_YEAR_GAP}{YEAR_TAIL}$",
    re.IGNORECASE,
)
# „München, [zwischen 17. und 20.] März 1908“: der Tag liegt in der Spanne.
BRACKET_RANGE_RE = re.compile(
    rf"^({PLACE_HEAD}){PLACE_GAP}\[\s*zwischen\s+(?:dem\s+)?"
    rf"({DAY_TOKEN})\.?\s*(?:-|und)\s*({DAY_TOKEN})\.?\s*\]\s*"
    rf"{MONTH_NOISE}({MONTH_PATTERN}){MONTH_YEAR_GAP}{YEAR_TAIL}$",
    re.IGNORECASE,
)
# „Stuttgart, Weihnachten/Sylvester 1910“ ist der 24. Dezember.
WEIHNACHT_RE = re.compile(
    rf"^({PLACE_HEAD}){PLACE_GAP}Weihnachten(?:\s*/\s*Sylvester)?\s+(\d{{4}})\s*[*'´′'`^]*\s*$",
    re.IGNORECASE,
)
# „Gehalten in Dornach am 27. Mai 1923, 23. Oktober 1923, 3. Januar 1924“.
HELD_ON_RE = re.compile(
    rf"^Gehalten\s+in\s+({PLACE_HEAD})\s+am\s+(.+)$",
    re.IGNORECASE,
)
HELD_DATE_RE = re.compile(
    rf"({DAY_TOKEN}){DAY_SEP}{MONTH_NOISE}({MONTH_PATTERN})\s+(\d{{4}})",
    re.IGNORECASE,
)
# „Berlin, 1903 oder 1904“: die Katalogangabe ist „1903, 1904“.
YEAR_SPAN_RE = re.compile(
    rf"^({PLACE_HEAD}){PLACE_GAP}(\d{{4}})\s+oder\s+(\d{{4}})\s*[*]?\s*$",
    re.IGNORECASE,
)
# „München, 10. oder 11. November 1905“ nennt beide Tage.
ODER_RE = re.compile(
    rf"^{LECTURE_PREFIX}({PLACE_HEAD}){PLACE_GAP}"
    rf"({DAY_TOKEN})\.\s+oder\s+({DAY_TOKEN})\.\s+"
    rf"{MONTH_NOISE}({MONTH_PATTERN}){MONTH_YEAR_GAP}{YEAR_TAIL}$",
    re.IGNORECASE,
)
# „Berlin, 25. und 26. April 1907“ beginnt beide Vorträge.
COMBINED_HEADING_RE = re.compile(
    rf"^{LECTURE_PREFIX}(?:({PLACE_HEAD}){PLACE_GAP}{DATE_INTRO})?"
    rf"({DAY_TOKEN})\.\s+und\s+({DAY_TOKEN})\.\s+{MONTH_NOISE}({MONTH_PATTERN}){MONTH_YEAR_GAP}{YEAR_TAIL}$",
    re.IGNORECASE,
)
# „18. und 19. Oktober 1902 [in Berlin]“: der Ort steht in Klammern hinter dem Jahr.
COMBINED_BRACKET_RE = re.compile(
    rf"^({DAY_TOKEN})\.\s+und\s+({DAY_TOKEN})\.\s+{MONTH_NOISE}({MONTH_PATTERN}){MONTH_YEAR_GAP}{YEAR_TAIL}"
    rf"\s+\[(?:in\s+)?({PLACE_HEAD})\]\s*$",
    re.IGNORECASE,
)
# Eine Ziffer der Jahreszahl, die im Druck leicht verwechselt wird: 1903 statt 1905.
CONFUSABLE_DIGITS = {
    ("3", "5"),
    ("5", "3"),
    ("3", "8"),
    ("8", "3"),
    ("8", "0"),
    ("0", "8"),
    ("1", "7"),
    ("7", "1"),
    ("6", "0"),
    ("0", "6"),
    ("3", "9"),
    ("9", "3"),
}
PLACE_NOISE_RE = re.compile(
    r"\b(vgl|siehe|vortrag|vorträge|gehalten|vortraege)\b",
    re.IGNORECASE,
)
DATE_ANY_RE = re.compile(
    rf"\b({DAY_TOKEN}){DAY_SEP}{MONTH_NOISE}({MONTH_PATTERN})(?:[ \t]*,[ \t]*|[ \t]+)(\d{{4}})\b",
    re.IGNORECASE,
)
PDF_NAME_RE = re.compile(r"^GA\s+(\d+)([a-z]?)\.pdf$", re.IGNORECASE)
STOPWORDS = {
    "einer",
    "eines",
    "einem",
    "einen",
    "eine",
    "oder",
    "und",
    "der",
    "die",
    "das",
    "den",
    "dem",
    "des",
    "vom",
    "von",
    "zur",
    "zum",
    "mit",
    "für",
    "auf",
    "aus",
    "ist",
    "sind",
    "wird",
    "eine",
    "einer",
    "im",
    "in",
    "am",
    "an",
    "als",
    "auch",
    "nach",
    "über",
    "ihre",
    "ihrer",
    "seiner",
    "seine",
    "nicht",
    "durch",
    "bei",
    "dass",
    "wird",
}

PASS1_SYSTEM = """\
Du ordnest Vorträge aus dem Rudolf-Steiner-Katalog den PDF-Seiten eines GA-Bandes zu.

Regeln:
- seite_pdf ist die 1-basierte PDF-Seite, auf der der Vortrag beginnt (Überschrift), nicht die Seite des Inhaltsverzeichnisses und nicht eine Erwähnung in Einleitung, Anhang oder Hinweisen.
- Die Liste „Vortragsanfänge“ ist maßgeblich. Die Zahl hinter „PDF“ ist die PDF-Seite. Die gedruckte Seitenzahl am Rand des Inhaltsverzeichnisses darf davon abweichen; dann gilt die PDF-Seite des Vortragsanfangs.
- Ordne über Datum und Ort zu. Katalogtitel können Tippfehler, Kürzungen oder eine andere Schreibweise haben. Der Titel im Band darf abweichen.
- Datumszeilen dürfen klein abweichen und bleiben dasselbe Datum. Nach dem Tag gelten Punkt, Komma und ein fehlendes Satzzeichen gleich, Leerzeichen darum auch. „16, November 1911“, „16. November 1911“, „16 . November 1911“ und „16 November 1911“ sind der 16. November 1911. Die Ziffer 1 ist oft der Buchstabe l oder I, und vor dem Monat kann das Leerzeichen fehlen: „l.März 1917“, „l. März 1917“ und „I. März 1917“ sind der 1. März 1917. Ein Leerzeichen zwischen den Ziffern des Tages ändert ihn nicht: „2 2. August 1910“ ist der 22. August 1910, „2 3. August 1910“ der 23. August 1910. Steht J allein an der Stelle des Tages, ist es die 5: „Kopenhagen, J. Juni 1910“ ist Kopenhagen am 5. Juni 1910. Steht z allein an der Stelle des Tages, ist es die 3: „Berlin, z. Febtuar 1913“ ist Berlin am 3. Februar 1913. Ein Ausrufezeichen an der Stelle des Tages ist die 1: „Dörnach,!. Januar 1916“ ist Dornach am 1. Januar 1916. Ein Schrägstrich an der Stelle der Ziffer ist die 1: „Darnach, //. November 1917“ ist Dornach am 11. November 1917. Steht n allein an der Stelle des Tages, ist es die 11: „Dörnach, n. September 1920“ ist Dornach am 11. September 1920. Ein Doppelpunkt nach dem Tag ist ein Punkt: „in Stuttgart am 8: September 1921“ ist Stuttgart am 8. September 1921. „Ansprache in Dornach am 6. Januar 1923 nach dem Brande des Goetheanum“ ist Dornach am 6. Januar 1923. „Ansprache und Fragenbeantwortung in Stuttgart am 14. Februar 1923“ ist Stuttgart am 14. Februar 1923. „Zweiter Vortrag vom 15. Oktober 1922“ ist der 15. Oktober 1922. „Zweiter Vortrag Stuttgart, 14. Oktober 1922“ ist Stuttgart am 14. Oktober 1922, auch ohne Komma nach Vortrag. „Erster Vortrag London, 12. November 1922“ ist London am 12. November 1922. „Prag, 29. .April 1923“ ist Prag am 29. April 1923. Steht „am 17. Juni 1924“ auf der nächsten Zeile nach „in Koberwitz“, ist es Koberwitz am 17. Juni 1924. Apostrophe hinter dem Jahr sind eine Fußnote: „Mannheim, 19. Januar 1922'''“ ist Mannheim am 19. Januar 1922. Ein Komma vor der Jahreszahl ändert das Datum nicht: „Berlin, 20. Februar, 1917“ ist Berlin am 20. Februar 1917. „Febtuar“ ist Februar. Eine schließende Klammer an der zweiten Stelle des Tages ist die 3: „Berlin, 2). April 1912“ ist Berlin am 23. April 1912. „io. Oktober 1918“ ist der 10. Oktober 1918, weil i die 1 und o die 0 ist. Ein Apostroph vor dem Monat ist Druckfehler: „14. 'November 1907“ ist der 14. November 1907. „Bremen, 5. Februar 1906“ bleibt der Vortragsanfang, auch wenn darunter „vom 8. Februar 1906“ steht.
- Zwischen Ort und Datum darf das Komma fehlen, ein Semikolon oder ein Backslash stehen: „Hannover; 25. Februar 1908“, „Regensburg 17. Dezember 1905“, „Leipzig 9. Juni 1907“, „Düsseldorf\\ 11. Februar 1906“, „Basel» 16. November 1907“, „Leipzig* 5- November 1911“, „London^ 7. Mai 1913“. „den“ oder „am“ zwischen Ort und Tag gehört zum Datum: „Hannover, den 18. Februar 1915“ ist Hannover am 18. Februar 1915. „Montag, den 5. November 1917“ ist kein Vortragsanfang: der Wochentag ist kein Ort. „Berlin am 3. April 1905“ ist Berlin am 3. April 1905. Ein Wort zwischen Ort und Datum ist kein Ort: „Berlin, Karfreitag, 21. April 1905“ ist Berlin am 21. April 1905. „Den Haag, Ostersonntag, 23. März 1913“ ist Den Haag am 23. März 1913. „Haag“ und „Den Haag“ sind derselbe Ort. „Penmaenmawr (N.Wales)“ ist Penmaenmawr. Fehlt die Jahreszahl und stimmen Ort, Tag und Monat, bleibt es dieser Vortrag: „Dornach, 10. September“ ist Dornach am 10. September. „Li. Dezember 1923“ ist der 21. Dezember 1923, weil L die 2 und i die 1 ist. „ZI. Dezember 1923“ ist der 31. Dezember 1923, weil Z die 3 und I die 1 ist. „12» August 1924“ ist der 12. August 1924. „[Karlsruhe], 15. April 1905“ ist Karlsruhe am 15. April 1905. „'Nürnberg, 12. Dezember 1909“ ist Nürnberg am 12. Dezember 1909. „Aus einem Vortrag in Berlin vom 28. April 1914“ ist Berlin am 28. April 1914. „18. und 19. Oktober 1902 [in Berlin]“ ist Berlin am 18. und am 19. Oktober 1902. „London, 4. Juli 1903, Autoreferat von Rudolf Steiner,“ ist London am 4. Juli 1903. „Vortrag von Rudolf Steiner, Berlin, 4. Juli 1904“ ist Berlin am 4. Juli 1904. „Berlin, zwischen dem 4. und 14. März 1907“ ist Berlin am 14. März 1907. „Ansprache Rudolf Steiners am 25. März 1907 in Berlin“ ist Berlin am 25. März 1907. „Berlin, 20. Oktober 1907, Motzstr. 17“ ist Berlin am 20. Oktober 1907. „Erster Tag, 18. Januar 1914“ ist der 18. Januar 1914. „Montag, den 19. Januar 1914, 1 Uhr mittags“ ist der 19. Januar 1914. „Sonnabend, den 24. Januar 1914, 10 Uhr abends“ ist der 24. Januar 1914. Steht der Wochentag allein, ist das der Vortragsanfang ohne neuen Ort. „Dornach, den 19. August 1915“ vor „Sehr geehrter Herr Dr. Steiner!“ ist der Briefkopf, kein zweiter Vortrag; „Dornach, 21. August 1915“ bleibt der Beginn. „Basler Volksblatt, 4. September 1920“ ist eine Zeitung, kein Ort; „Dornach, 5. September 1920“ bleibt der Beginn. „am 25. Dezember 1923, 10 Uhr vormittags“ ist der 25. Dezember 1923. „am 25. Dezember 1923, 2 Uhr 30 nachmittags, im Glashaus“ bleibt derselbe Tag. „am Sonntag, 23. Dezember 1923“ ist der 23. Dezember 1923. „am 24. Dezember, 10 Uhr vormittags“ ohne Jahr ist der 24. Dezember. „vom 24. Dezember 1923, 11 Uhr 15 vormittags“ ist der 24. Dezember 1923. „Vortrag von Rudolf Steiner am 31. Dezember 1923, abends“ ist der 31. Dezember 1923. „Am 31. Dezember, nachmittags 2 Uhr 30, fand im Architektenbüro …“ ist der 31. Dezember. „am 1. Januar 1924,4 Uhr 30 nachmittags“ ist der 1. Januar 1924. Ein Anführungszeichen hinter dem Jahr ändert es nicht: „Berlin, 2. Januar 1905“ mit schließendem Anführungszeichen ist Berlin am 2. Januar 1905. „Basel, 11. Januar 1924, 11 Uhr vormittags“ ist Basel am 11. Januar 1924. „Berlin, 7. Januar 1908 (Fortsetzung vom 29. November 1907)“ ist Berlin am 7. Januar 1908; der Tag in der Klammer ist kein zweiter Beginn. „München, 10. oder 11. November 1905“ ist München am 10. und am 11. November 1905. Stehen mehrere Zeilen, die nur Ort und Datum sind, auf einer Seite, ist jede ein Vortragsbeginn: „Berlin, 21. Oktober 1906“, „Berlin, 20. Oktober 1907“ und „Berlin, 26. Oktober 1908“. „Berlin, 28. Dezember 1903“ vor „Verehrtestes Fräulein“ oder „Verehrtes liebes Fräulein“ und „Berlin, 2. Januar 1905“ vor „Mein lieber Herr“ sind Briefköpfe, kein Vortrag. „Stuttgart, 29. März 1923“ vor „Liebe Leidtragende!“ bleibt der Beginn. „Instmktionsstunde Berlin, 26. Juni 1906^“ ist Berlin am 26. Juni 1906. „Stuttgart, [23.] Februar 1912“ ist Stuttgart am 23. Februar 1912. „München, [15. oder 16.] Dezember 1905“ ist München am 15. und am 16. Dezember 1905. „München, [zwischen 17. und 20.] März 1908“ ist München vom 17. bis zum 20. März 1908. „Stuttgart, [zwischen 17.-20.] Februar 1913“ ist Stuttgart vom 17. bis zum 20. Februar 1913. „Berlin, 27. Februar und 15. März 1906“ ist Berlin am 27. Februar und am 15. März 1906. „Den Haag, 21. März 1913 (Karfreitag)“ ist Den Haag am 21. März 1913. „Dornach, 30. Dezember 1923, 8 Uhr 30 morgens im Glashaus“ ist Dornach am 30. Dezember 1923. „Kristiana (Oslo)“ ist Kristiania (Oslo). „Berlin, 1903 oder 1904“ ist Berlin in den Jahren 1903 und 1904. „Esoterische Stunde in Kristiania (Oslo), 18. oder 20. Mai 1923“ ist Kristiania am 18. und am 20. Mai 1923. „Gehalten in Dornach am 27. Mai 1923, 23. Oktober 1923, 3. Januar 1924“ ist Dornach an diesen drei Tagen. „Stuttgart, Weihnachten/Sylvester 1910“ ist Stuttgart am 24. Dezember 1910. „im Vortrag Leipzig, 13. Oktober 1906“ mit „folgende Ausführungen“ danach ist der Beginn. „im Vortrag Dornach, 26. November 1916, dargelegt“ ist Dornach am 26. November 1916. „aus dem Vortrag Dornach, 4. Juni 1924 hier vorangestellt“ ist Dornach am 4. Juni 1924. „Instruktionsstunde Berlin, 15. April 1908“ vor „(Aufzeichnungen von anderer Hand)“ ist kein zweiter Beginn. „Aus Vortrag (Fragenbeantwortungen) Dörnach, 22. April 1924“ ist Dornach am 22. April 1924. „Aus dem Vortrag vom 21. September 1920“ ist der 21. September 1920. „V. Dörnach, 2. Januar 1916“ ist Dornach am 2. Januar 1916; „V.“ heißt Vortrag. „V. 15. Oktober 1911“ ist der 15. Oktober 1911. „durch Rudolf Steiner am 25. Dezember 1923, 10 Uhr vormittags“ ist der 25. Dezember 1923. „Stuttgart, 4. Oktober 1921 (vormittags)“ ist Stuttgart am 4. Oktober 1921. Steht „am 20. September 1913“ auf der nächsten Zeile nach „Ansprache zur Grundsteinlegung des Dornacher Baues“, ist es der 20. September 1913. Dieselbe Quellenzeile „V. Pforzheim, 7. März 1914“ auf der nächsten Seite ist kein neuer Beginn. Das alphabetische Register ist kein Vortragsbeginn. „Dörnach, u. April 1915“ ist Dornach am 11. April 1915, weil u die 11 ist. „Dörnach, jo. Mai 191$“ ist Dornach am 30. Mai 1915: j ist die 3, o die 0, $ die 5. „Dörnach, if. August 191$“ ist Dornach am 15. August 1915: i die 1, f die 5. „Dörnach, 4. April 19/j“ ist Dornach am 4. April 1915. „Dörnach, 2. November 19//“ ist Dornach am 2. November 1917, wenn die letzte Ziffer 7 statt 1 steht. „Fortsetzung: Dörnach 28. September 1918“ ist Dornach am 28. September 1918. „Kristiania, (Oslo), 5. Juni 1912“ ist Kristiania (Oslo) am 5. Juni 1912. Die Jahreszahl darf in Klammern stehen: „Berlin, 18. August (1905?)“ ist der 18. August 1905. „(?)“ hinter dem Katalogdatum ändert das Datum nicht. „(Notizen)“ hinter dem Datum ändert es nicht: „Berlin, 30. September 1904 (Notizen)“ ist Berlin am 30. September 1904. Ein Stern hinter der Jahreszahl ist eine Fußnote: „Berlin, 28. September 1905*“ ist Berlin am 28. September 1905. „, vormittags“, „, abends“, „, Neujahrsfeier“, „, Einleitung zum Mitgliedervortrag“ und „, Feier zum Jahresende“ gehören zum selben Datum: „Hannover, 1. Januar 1912, Neujahrsfeier“ ist Hannover am 1. Januar 1912. „Berlin, 7. Januar 1913, Einleitung zum Mitgliedervortrag“ ist Berlin am 7. Januar 1913. „Dörnach, 31. Dezember 1914, Feier zum Jahresende“ ist Dornach am 31. Dezember 1914. In GA 164 ist „Dörnach, Z. September 1915“ der Vortrag 19151003b vom 3. Oktober 1915. In GA 167 ist „Berlin, 13. April 1916“ der Vortrag 19160411 vom 11. April 1916. In GA 272 ist „Dornach, 11. April 1915“ der Vortrag 19110411 vom 11. April 1911. „Dornach, 25. AUGUST 1918“ ist Dornach am 25. August 1918. „Programm zur Aufführung Dornach, 25. August 1918“ ist derselbe Beginn, das Programm danach nicht ein zweiter. „Ansprache zur Eurythmie, Dornach, 4. April 1920“ ist Dornach am 4. April 1920; „Meine sehr verehrten Anwesenden!“ ist der Anfang der Ansprache. „Ansprache zur Eurythmie, 29. August 1915, Domach“ ist Dornach am 29. August 1915. „Fünfte Stunde, Montag, 23. August 1915, vormittags“ ist der 23. August 1915; vormittags der frühere, nachmittags der spätere Eintrag desselben Tages. „Dornacb“ ist Dornach. „Berlin, 1$. Dezember 1911“ ist Berlin am 15. Dezember 1911. „ANSPRACHE AM ELTERNABEND“ und auf der nächsten Zeile „9. Mai 1922“ ist der 9. Mai 1922. „Meine sehr verehrten Anwesenden!“ ist der Anfang der Ansprache. „Konferenz vom Donnerstag 25. September 1919, 8.30 Uhr“ ist der 25. September 1919. „am 20. August 1919 in Stuttgart“ ist Stuttgart am 20. August 1919. „Donnerstag, 5. Juli 1923“ unter „Besprechung mit dem Verwaltungsrat“ ist der 5. Juli 1923. „am 15. Mai 1920 in Dörnach*“ ist Dornach am 15. Mai 1920. „Stuttgart, 25. März 1923“ und darunter „Meine sehr verehrten Anwesenden!“ ist Stuttgart am 25. März 1923. „Kristiania (Oslo), 23. November 1921“ ist Kristiania am 23. November 1921. „Ilkley, 10. Augast 1923“ ist Ilkley am 10. August 1923. In GA 304a ist das der Vortrag 19230810b, Katalogort Bingley. „London, 30. August 1924“ und darunter „Meine sehr verehrten Damen und Herren!“ ist London am 30. August 1924. „(Kurznotizen)“ gehört zum selben Datum: „Berlin, 22. Oktober 1906, vormittags“ und „Berlin, 22. Oktober 1906, abends“ sind zwei Vorträge, unterschieden am Titel. „Berlin, 7. April 1907“ ist der 1. April 1907, wenn 7 statt 1 steht und Ort, Monat und Jahr stimmen. „13- Februar 1906“ ist der 13. Februar 1906. „Mär%“, „Mär^“, „Mär£“ und „Mar£“ sind März: „Stuttgart, 5. Mär% 1910“ ist Stuttgart am 5. März 1910. „Pforzheim, 7. Mär^ 1914“ ist Pforzheim am 7. März 1914. „München, 30. Mär£ 1914“ ist München am 30. März 1914. „London^ 7. Mai 1913“ ist London am 1. Mai 1913, wenn 7 statt 1 steht. Steht im Katalog am selben Ort auch der gedruckte Tag, bleibt es dieser Tag: „München, 13. März 1910“ ist der 13. März, nicht der 15. Ein Stern statt des Punktes nach dem Tag ändert nichts: „Kassel, 26* Juni 1909“ ist Kassel am 26. Juni 1909. Ein Punkt nach der Jahreszahl ändert nichts: „Frankfurt am Main, 18. Januar 1922.“ ist Frankfurt am 18. Januar 1922. „Frankfurt am Main“ ist Frankfurt.
- Vor dem Ort kann „öffentlicher Vortrag“ oder „nicht öffentlicher Vortrag“ stehen, auch mit Semikolon danach. „öffentlicher Vortrag Berlin, 28. August 1899“ ist Berlin am 28. August 1899. „nicht öffentlicher Vortrag Berlin, 4. April 1904“ ist Berlin am 4. April 1904. „nicht öffentlicher Vortrag; Leipzig, 28. November 1909“ ist Leipzig. Ort und Datum dürfen auf der nächsten Zeile stehen. Ein späteres „vom 20. Januar 1890“ im Zeitungsbericht ist nicht der Vortragsanfang.
- „Berlin, 25. und 26. April 1907“ ist der Beginn beider Vorträge. Beide bekommen dieselbe PDF-Seite. „vom 6./7. Januar 1906“ ist der 6. und der 7. Januar 1906. „vom 13. Januar 1906“ ist der 13. Januar 1906.
- „Erster Vortrag, Paris, 25. Mai 1906“ ist Paris am 25. Mai 1906. Die Zahl vor „Vortrag“ gehört nicht zum Ort. „Aus Vortrag Berlin, 30, Oktober 1905“, „Aus Vortrag Stuttgart, 75. September 1907“ und „Aus Vortrag Berlin, 17. Juni 1909“ sind Vortragsanfänge. 75 ist der 15. „Aus dem Vortrag Basel, 3. Februar 1909:“ ist Basel am 3. Februar 1909. „Vor trag“ ist Vortrag. Ein Doppelpunkt oder ein Zusatz hinter dem Jahr ändert das Datum nicht: „Aus dem Vortrag Berlin, 22. März 1909 (ganzer Vortrag …)“ ist Berlin am 22. März 1909. „Aus einer esoterischen (Instruktions-)Stunde in Berlin, 17. Dezember 1911“ ist Berlin am 17. Dezember 1911. „Aus demselben Vortrag“ ist kein neuer Beginn. „Notizen aus dem Vortrag, Rom, 12. April 1910“ ist Rom am 12. April 1910. „Notizen aus dem Vortrag* Köln 29.Januar 1911“ ist Köln am 29. Januar 1911. „Private Lehrstunde, Berlin-Schlachtensee, Sommer 1903“ ist Berlin im Sommer 1903.
- Im Ortsnamen kann „rn“ als „m“ gelesen sein: „Bem“ ist Bern. „ß“ und „ss“ sind gleich: „Straßburg“ ist Strassburg. „ch“ am Wortende ist oft „di“ oder „dh“: „Züridi“ ist Zürich, „Dornadh“ ist Dornach. Umlaut und Grundvokal gelten gleich: „Dörnach“ ist Dornach. „Darnach“ ist Dornach. „Neuchâtel“, „Neuchatel“ und „Neuchaiel“ sind derselbe Ort. „Heisingfors“ ist Helsingfors. „IVien“ ist Wien. „Lin%“ ist Linz.
- Vor dem Ort darf eine Bezeichnung stehen: „Erster Gesprächsabend, Dörnach, 4. Oktober 1920“ ist Dornach am 4. Oktober 1920. „nach dem Vortrag …“ ist keine solche Bezeichnung.
- Eine Zeile, die nur Ort und Datum ist, bleibt der Vortragsanfang, auch wenn sie mitten auf der Seite unter dem vorigen Text steht: „Berlin, 18. Oktober 1903“, „Berlin, 27. Oktober 1903“, „Berlin, 8. Dezember 1903“. Ein Satz wie „sprach Dr. Rudolf Steiner am 18. Oktober 1903“ ist kein Vortragsanfang.
- „Berlin-Schlachtensee, Sommer 1903“ ist Berlin im Sommer 1903. Das Katalogdatum „1903 Sommer“ ist derselbe Zeitraum. Mehrere Stunden desselben Sommers unterscheiden sich am Titel.
- Ein Tag über 31, bei dem eine Ziffer verwechselt ist, bleibt der gültige Tag. „Hamburg, 75. November 1913“ ist der 15. November 1913, weil 7 statt 1 steht.
- Weicht nur eine leicht verwechselbare Ziffer der Jahreszahl ab und stimmen Tag, Monat und Ort, ist es derselbe Vortrag. „Hamburg, 23. Januar 1903“ ist der Katalogeintrag vom 23.01.1905, „Düsseldorf, 3. Dezember 1903“ ist der vom 03.12.1905. „München, lo.März 1919“ ist München am 10. März 1913, weil 9 statt 3 steht und l die 1, o die 0 ist. „Stockholm, 76. ,4p rz7 7972“ ist Stockholm am 16. April 1912: 76 ist der 16., „4p rz7“ ist April, 7972 ist 1912. Ein anderer Monat oder ein klar anderes Jahr bleibt ein anderes Datum.
- Dasselbe Datum an verschiedenen Orten sind verschiedene Vorträge. Unterscheide sie am Ort.
- Jede Katalog-id kommt genau einmal vor. Keine zusätzlichen ids. Übernimm die id Zeichen für Zeichen.
- Fehlt der Vortrag in diesem Band, setze seite_pdf auf null.
- Mehrere Katalogeinträge dürfen dieselbe seite_pdf haben, wenn der Band sie auf einer Seite zusammenfasst.
- hinweis: höchstens zwei deutsche Sätze, nach dem Array. Sag, ob die PDF-Seite der gedruckten Seitenzahl entspricht oder um wie viele Seiten sie versetzt ist, und welche Katalogvorträge im Band fehlen oder zusammengefasst sind. Keine Nacherzählung des Inhaltsverzeichnisses.
- Antworte nur mit einem JSON-Objekt, ohne Markdown. Zuerst vortraege, danach hinweis.

Schema:
{"vortraege":[{"id":"19141029","seite_pdf":13}],"hinweis":"..."}
seite_pdf ist eine ganze Zahl oder null.
"""

PASS2_SYSTEM = """\
Einige Katalogvorträge wurden im ersten Durchgang nicht sicher einer PDF-Seite zugeordnet. Du suchst nur diese Restliste.

Regeln:
- Du darfst nur die ids der Restliste zurückgeben, jede genau einmal.
- Eine Fundstelle ist nur ein Vortragsbeginn: Überschrift mit Datum und Ort am Seitenanfang. Das Inhaltsverzeichnis, die Einführung und die Hinweise (Fußnoten, Querverweise) sind keine Fundstelle.
- Ein Suchtreffer mit anderem Datum oder anderem Ort gehört zu einem anderen Vortrag, auch wenn der Titel ähnlich ist.
- Kleine Abweichungen in der Datumszeile sind kein anderes Datum. „16, November 1911“ ist der 16. November 1911. „l.März 1917“ ist der 1. März 1917. „2 2. August 1910“ ist der 22. August 1910, „2 3. August 1910“ der 23. August 1910. „J. Juni 1910“ ist der 5. Juni 1910. „z. Febtuar 1913“ ist der 3. Februar 1913. „!. Januar 1916“ ist der 1. Januar 1916. „Darnach, //. November 1917“ ist Dornach am 11. November 1917. „n. September 1920“ ist der 11. September 1920. „8: September 1921“ ist der 8. September 1921. „in Stuttgart am 8. September 1921“ ist Stuttgart. „Ansprache in Dornach am 6. Januar 1923 nach dem Brande des Goetheanum“ ist Dornach am 6. Januar 1923. „Zweiter Vortrag vom 15. Oktober 1922“ ist der 15. Oktober 1922. „Zweiter Vortrag Stuttgart, 14. Oktober 1922“ ist Stuttgart am 14. Oktober 1922. „Erster Vortrag London, 12. November 1922“ ist London am 12. November 1922. „Prag, 29. .April 1923“ ist Prag am 29. April 1923. „in Koberwitz“ und auf der nächsten Zeile „am 17. Juni 1924“ ist Koberwitz am 17. Juni 1924. „Mannheim, 19. Januar 1922'''“ ist Mannheim am 19. Januar 1922. „Berlin, 20. Februar, 1917“ ist Berlin am 20. Februar 1917. „2). April 1912“ ist der 23. April 1912. „io. Oktober 1918“ ist der 10. Oktober 1918. „Züridi“ ist Zürich. „Dornadh“ ist Dornach. „Darnach“ ist Dornach. „Dörnach“ ist Dornach. „Neuchâtel“, „Neuchatel“ und „Neuchaiel“ sind derselbe Ort. „Heisingfors“ ist Helsingfors. „IVien“ ist Wien. „Lin%“ ist Linz. „Erster Gesprächsabend, Dörnach, 4. Oktober 1920“ ist Dornach am 4. Oktober 1920. „14. 'November 1907“ ist der 14. November 1907. „Bremen, 5. Februar 1906“ bleibt der Vortragsanfang, auch wenn darunter „vom 8. Februar 1906“ steht. „Hannover; 25. Februar 1908“, „Regensburg 17. Dezember 1905“, „Leipzig 9. Juni 1907“ und „Düsseldorf\\ 11. Februar 1906“, „Basel» 16. November 1907“, „Leipzig* 5- November 1911“, „Kristiania, (Oslo), 5. Juni 1912“ und „Hannover, den 18. Februar 1915“ und „Frankfurt am Main, 18. Januar 1922.“ sind gültige Vortragsanfänge. „Montag, den 5. November 1917“ ist kein Vortragsanfang: der Wochentag ist kein Ort. „Frankfurt am Main“ ist Frankfurt. „Berlin, 27. Oktober 1903“ mitten auf der Seite ist der Vortragsanfang. „Berlin-Schlachtensee, Sommer 1903“ ist Berlin im Sommer 1903, Katalogdatum „1903 Sommer“. „Berlin am 3. April 1905“ ist Berlin. „Berlin, Karfreitag, 21. April 1905“ ist Berlin am 21. April 1905. „Den Haag, Ostersonntag, 23. März 1913“ ist Den Haag am 23. März 1913. „Haag“ ist Den Haag. „Penmaenmawr (N.Wales)“ ist Penmaenmawr. Fehlt die Jahreszahl und stimmen Ort, Tag und Monat, bleibt es dieser Vortrag: „Dornach, 10. September“ ist Dornach am 10. September. „Li. Dezember 1923“ ist der 21. Dezember 1923, weil L die 2 und i die 1 ist. „ZI. Dezember 1923“ ist der 31. Dezember 1923, weil Z die 3 und I die 1 ist. „12» August 1924“ ist der 12. August 1924. „[Karlsruhe], 15. April 1905“ ist Karlsruhe am 15. April 1905. „'Nürnberg, 12. Dezember 1909“ ist Nürnberg am 12. Dezember 1909. „Aus einem Vortrag in Berlin vom 28. April 1914“ ist Berlin am 28. April 1914. „18. und 19. Oktober 1902 [in Berlin]“ ist Berlin am 18. und am 19. Oktober 1902. „London, 4. Juli 1903, Autoreferat von Rudolf Steiner,“ ist London am 4. Juli 1903. „Vortrag von Rudolf Steiner, Berlin, 4. Juli 1904“ ist Berlin am 4. Juli 1904. „Berlin, zwischen dem 4. und 14. März 1907“ ist Berlin am 14. März 1907. „Ansprache Rudolf Steiners am 25. März 1907 in Berlin“ ist Berlin am 25. März 1907. „Berlin, 20. Oktober 1907, Motzstr. 17“ ist Berlin am 20. Oktober 1907. „Erster Tag, 18. Januar 1914“ ist der 18. Januar 1914. „Montag, den 19. Januar 1914, 1 Uhr mittags“ ist der 19. Januar 1914. „Sonnabend, den 24. Januar 1914, 10 Uhr abends“ ist der 24. Januar 1914. Steht der Wochentag allein, ist das der Vortragsanfang ohne neuen Ort. „Dornach, den 19. August 1915“ vor „Sehr geehrter Herr Dr. Steiner!“ ist der Briefkopf, kein zweiter Vortrag; „Dornach, 21. August 1915“ bleibt der Beginn. „Basler Volksblatt, 4. September 1920“ ist eine Zeitung, kein Ort; „Dornach, 5. September 1920“ bleibt der Beginn. „am 25. Dezember 1923, 10 Uhr vormittags“ ist der 25. Dezember 1923. „am 25. Dezember 1923, 2 Uhr 30 nachmittags, im Glashaus“ bleibt derselbe Tag. „am Sonntag, 23. Dezember 1923“ ist der 23. Dezember 1923. „am 24. Dezember, 10 Uhr vormittags“ ohne Jahr ist der 24. Dezember. „vom 24. Dezember 1923, 11 Uhr 15 vormittags“ ist der 24. Dezember 1923. „Vortrag von Rudolf Steiner am 31. Dezember 1923, abends“ ist der 31. Dezember 1923. „Am 31. Dezember, nachmittags 2 Uhr 30, fand im Architektenbüro …“ ist der 31. Dezember. „am 1. Januar 1924,4 Uhr 30 nachmittags“ ist der 1. Januar 1924. Ein Anführungszeichen hinter dem Jahr ändert es nicht: „Berlin, 2. Januar 1905“ mit schließendem Anführungszeichen ist Berlin am 2. Januar 1905. „Basel, 11. Januar 1924, 11 Uhr vormittags“ ist Basel am 11. Januar 1924. „Berlin, 7. Januar 1908 (Fortsetzung vom 29. November 1907)“ ist Berlin am 7. Januar 1908; der Tag in der Klammer ist kein zweiter Beginn. „München, 10. oder 11. November 1905“ ist München am 10. und am 11. November 1905. Stehen mehrere Zeilen, die nur Ort und Datum sind, auf einer Seite, ist jede ein Vortragsbeginn: „Berlin, 21. Oktober 1906“, „Berlin, 20. Oktober 1907“ und „Berlin, 26. Oktober 1908“. „Berlin, 28. Dezember 1903“ vor „Verehrtestes Fräulein“ oder „Verehrtes liebes Fräulein“ und „Berlin, 2. Januar 1905“ vor „Mein lieber Herr“ sind Briefköpfe, kein Vortrag. „Stuttgart, 29. März 1923“ vor „Liebe Leidtragende!“ bleibt der Beginn. „Instmktionsstunde Berlin, 26. Juni 1906^“ ist Berlin am 26. Juni 1906. „Stuttgart, [23.] Februar 1912“ ist Stuttgart am 23. Februar 1912. „München, [15. oder 16.] Dezember 1905“ ist München am 15. und am 16. Dezember 1905. „München, [zwischen 17. und 20.] März 1908“ ist München vom 17. bis zum 20. März 1908. „Stuttgart, [zwischen 17.-20.] Februar 1913“ ist Stuttgart vom 17. bis zum 20. Februar 1913. „Berlin, 27. Februar und 15. März 1906“ ist Berlin am 27. Februar und am 15. März 1906. „Den Haag, 21. März 1913 (Karfreitag)“ ist Den Haag am 21. März 1913. „Dornach, 30. Dezember 1923, 8 Uhr 30 morgens im Glashaus“ ist Dornach am 30. Dezember 1923. „Kristiana (Oslo)“ ist Kristiania (Oslo). „Berlin, 1903 oder 1904“ ist Berlin in den Jahren 1903 und 1904. „Esoterische Stunde in Kristiania (Oslo), 18. oder 20. Mai 1923“ ist Kristiania am 18. und am 20. Mai 1923. „Gehalten in Dornach am 27. Mai 1923, 23. Oktober 1923, 3. Januar 1924“ ist Dornach an diesen drei Tagen. „Stuttgart, Weihnachten/Sylvester 1910“ ist Stuttgart am 24. Dezember 1910. „im Vortrag Leipzig, 13. Oktober 1906“ mit „folgende Ausführungen“ danach ist der Beginn. „im Vortrag Dornach, 26. November 1916, dargelegt“ ist Dornach am 26. November 1916. „aus dem Vortrag Dornach, 4. Juni 1924 hier vorangestellt“ ist Dornach am 4. Juni 1924. „Instruktionsstunde Berlin, 15. April 1908“ vor „(Aufzeichnungen von anderer Hand)“ ist kein zweiter Beginn. „Aus Vortrag (Fragenbeantwortungen) Dörnach, 22. April 1924“ ist Dornach am 22. April 1924. „Aus dem Vortrag vom 21. September 1920“ ist der 21. September 1920. „V. Dörnach, 2. Januar 1916“ ist Dornach am 2. Januar 1916; „V.“ heißt Vortrag. „V. 15. Oktober 1911“ ist der 15. Oktober 1911. „durch Rudolf Steiner am 25. Dezember 1923, 10 Uhr vormittags“ ist der 25. Dezember 1923. „Stuttgart, 4. Oktober 1921 (vormittags)“ ist Stuttgart am 4. Oktober 1921. Steht „am 20. September 1913“ auf der nächsten Zeile nach „Ansprache zur Grundsteinlegung des Dornacher Baues“, ist es der 20. September 1913. Dieselbe Quellenzeile „V. Pforzheim, 7. März 1914“ auf der nächsten Seite ist kein neuer Beginn. Das alphabetische Register ist kein Vortragsbeginn. „Dörnach, u. April 1915“ ist Dornach am 11. April 1915, weil u die 11 ist. „Dörnach, jo. Mai 191$“ ist Dornach am 30. Mai 1915: j ist die 3, o die 0, $ die 5. „Dörnach, if. August 191$“ ist Dornach am 15. August 1915: i die 1, f die 5. „Dörnach, 4. April 19/j“ ist Dornach am 4. April 1915. „Dörnach, 2. November 19//“ ist Dornach am 2. November 1917, wenn die letzte Ziffer 7 statt 1 steht. „Fortsetzung: Dörnach 28. September 1918“ ist Dornach am 28. September 1918. „Berlin, 18. August (1905?)“ ist der 18. August 1905. „(?)“ hinter dem Katalogdatum ändert das Datum nicht. „Berlin, 30. September 1904 (Notizen)“ ist Berlin am 30. September 1904. „Berlin, 28. September 1905*“ ist Berlin am 28. September 1905. „Berlin, 22. Oktober 1906, vormittags“ und „..., abends“ sind zwei Vorträge desselben Tages. „(Kurznotizen)“ ändert das Datum nicht. „Hannover, 1. Januar 1912, Neujahrsfeier“, „Berlin, 7. Januar 1913, Einleitung zum Mitgliedervortrag“ und „Dörnach, 31. Dezember 1914, Feier zum Jahresende“ behalten Ort und Datum. In GA 164 ist „Z. September 1915“ der Vortrag 19151003b vom 3. Oktober 1915. In GA 167 ist „Berlin, 13. April 1916“ der Vortrag 19160411 vom 11. April 1916. In GA 272 ist „Dornach, 11. April 1915“ der Vortrag 19110411 vom 11. April 1911. „Dornach, 25. AUGUST 1918“ ist Dornach am 25. August 1918. „Programm zur Aufführung Dornach, 25. August 1918“ ist derselbe Beginn, das Programm danach nicht ein zweiter. „Ansprache zur Eurythmie, Dornach, 4. April 1920“ ist Dornach am 4. April 1920; „Meine sehr verehrten Anwesenden!“ ist der Anfang der Ansprache. „Ansprache zur Eurythmie, 29. August 1915, Domach“ ist Dornach am 29. August 1915. „Fünfte Stunde, Montag, 23. August 1915, vormittags“ ist der 23. August 1915; vormittags der frühere, nachmittags der spätere Eintrag desselben Tages. „Dornacb“ ist Dornach. „Berlin, 1$. Dezember 1911“ ist Berlin am 15. Dezember 1911. „ANSPRACHE AM ELTERNABEND“ und auf der nächsten Zeile „9. Mai 1922“ ist der 9. Mai 1922. „Meine sehr verehrten Anwesenden!“ ist der Anfang der Ansprache. „Konferenz vom Donnerstag 25. September 1919, 8.30 Uhr“ ist der 25. September 1919. „am 20. August 1919 in Stuttgart“ ist Stuttgart am 20. August 1919. „Donnerstag, 5. Juli 1923“ unter „Besprechung mit dem Verwaltungsrat“ ist der 5. Juli 1923. „am 15. Mai 1920 in Dörnach*“ ist Dornach am 15. Mai 1920. „Stuttgart, 25. März 1923“ und darunter „Meine sehr verehrten Anwesenden!“ ist Stuttgart am 25. März 1923. „Kristiania (Oslo), 23. November 1921“ ist Kristiania am 23. November 1921. „Ilkley, 10. Augast 1923“ ist Ilkley am 10. August 1923. In GA 304a ist das der Vortrag 19230810b, Katalogort Bingley. „London, 30. August 1924“ und darunter „Meine sehr verehrten Damen und Herren!“ ist London am 30. August 1924. „7. April 1907“ in Berlin ist der 1. April 1907. „13- Februar 1906“ ist der 13. Februar 1906. „vom 6./7. Januar 1906“ gilt für den 6. und den 7. Januar. „Erster Vortrag, Paris, 25. Mai 1906“ ist Paris am 25. Mai 1906. „Aus Vortrag Stuttgart, 75. September 1907“ ist Stuttgart am 15. September 1907. „Aus Vortrag Berlin, 17. Juni 1909“ ist Berlin am 17. Juni 1909. „Aus dem Vortrag Basel, 3. Februar 1909:“ ist Basel am 3. Februar 1909. „Vor trag“ ist Vortrag. „Aus dem Vortrag Berlin, 22. März 1909 (ganzer Vortrag …)“ ist Berlin am 22. März 1909. „Aus einer esoterischen Stunde in Berlin, 17. Dezember 1911“ ist Berlin am 17. Dezember 1911. „Aus demselben Vortrag“ ist kein neuer Beginn. „Notizen aus dem Vortrag, Rom, 12. April 1910“ ist Rom am 12. April 1910. „Notizen aus dem Vortrag* Köln 29.Januar 1911“ ist Köln am 29. Januar 1911. „Mär%“, „Mär^“ und „Mär£“ sind März. „London^ 7. Mai 1913“ ist London am 1. Mai 1913. „München, 13. März 1910“ bleibt der 13. März, wenn der Katalog den 13. und den 15. führt. „Kassel, 26* Juni 1909“ ist Kassel am 26. Juni 1909. „öffentlicher Vortrag Berlin, 28. August 1899“ und „nicht öffentlicher Vortrag Berlin, 4. April 1904“ sind Vortragsanfänge; der Zusatz vor dem Ort gehört nicht zum Ortsnamen. „25. und 26. April 1907“ gilt für beide Tage. „Bem“ ist Bern. „Straßburg“ ist Strassburg. „75. November 1913“ ist der 15. November 1913. „23. Januar 1903“ in Hamburg ist der 23.01.1905, „3. Dezember 1903“ in Düsseldorf ist der 03.12.1905, „München, lo.März 1919“ ist München am 10. März 1913, wenn Tag, Monat und Ort sonst stimmen. „76. ,4p rz7 7972“ ist der 16. April 1912. Nur ein anderer Tag, Monat oder Ort, oder ein klar anderes Jahr, ist ein anderer Vortrag.
- Findest du den Vortrag, setze seite_pdf auf die PDF-Seite des Beginns. Sonst seite_pdf null und grund als kurzen deutschen Satz, warum er in diesem Band nicht steht.
- Antworte nur mit einem JSON-Objekt, ohne Markdown.

Schema:
{"vortraege":[{"id":"19150305","seite_pdf":null,"grund":"Im Band gibt es diesen Berliner Vortrag nicht."}]}
"""


@dataclass
class Lecture:
    id: str
    datum: str
    ort: str
    titel: str
    reihe: str | None


@dataclass
class Assignment:
    seite_pdf: int | None = None
    grund: str | None = None
    markierung: str | None = None
    rejected_page: int | None = None
    reject_reason: str | None = None


@dataclass
class VolumeText:
    pages: list[str]
    ga: str = ""
    toc_indexes: set[int] = field(default_factory=set)
    toc_block: str = ""
    anchors: list[str] = field(default_factory=list)


def load_env() -> None:
    env_file = RAGRUN_ROOT / ".env"
    if not env_file.is_file():
        return
    values: dict[str, str] = {}
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        values[key.strip()] = val.strip().strip('"').strip("'")
    for key, val in values.items():
        if key in os.environ:
            continue
        match = re.fullmatch(r"\$\{([^}]+)\}", val)
        if match:
            val = os.environ.get(match.group(1), "")
        if val:
            os.environ[key] = val


def api_key() -> str:
    for name in ("DEEPSEEK_API_KEY", "DEEPSEEK_REST_API_KEY", "RAGRUN_DEEPSEEK_API_KEY"):
        value = os.environ.get(name, "").strip()
        if value and not value.startswith("${"):
            return value
    raise SystemExit(
        "DeepSeek-Schlüssel fehlt. Setze DEEPSEEK_API_KEY oder DEEPSEEK_REST_API_KEY."
    )


def normalize_ga_token(raw: str) -> str | None:
    text = raw.strip().lower().replace(" ", "")
    match = re.fullmatch(r"0*(\d+)([a-z])?", text)
    if not match:
        return None
    return f"{int(match.group(1))}{match.group(2) or ''}"


def ga_tokens(value: object) -> list[str]:
    tokens: list[str] = []
    for part in re.split(r"[,;]", str(value or "")):
        token = normalize_ga_token(part)
        if token and token not in tokens:
            tokens.append(token)
    return tokens


def load_catalog(path: Path) -> dict[str, list[Lecture]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    lectures = data.get("lectures") if isinstance(data, dict) else None
    if not isinstance(lectures, list):
        raise SystemExit(f"Katalog ohne lectures-Liste: {path}")
    by_ga: dict[str, list[Lecture]] = {}
    for entry in lectures:
        if not isinstance(entry, dict):
            continue
        lecture_id = str(entry.get("id") or "").strip()
        if not lecture_id:
            continue
        lecture = Lecture(
            id=lecture_id,
            datum=str(entry.get("datum") or "").strip(),
            ort=str(entry.get("ort") or "").strip(),
            titel=str(entry.get("vortragstitel") or "").strip(),
            reihe=(str(entry["reihe"]).strip() if entry.get("reihe") else None),
        )
        for token in ga_tokens(entry.get("ga")):
            by_ga.setdefault(token, []).append(lecture)
    return by_ga


def pdf_token(path: Path) -> tuple[int, str, str] | None:
    match = PDF_NAME_RE.match(path.name)
    if not match:
        return None
    number = int(match.group(1))
    suffix = match.group(2).lower()
    return number, suffix, f"{number}{suffix}"


def list_pdfs(ga_dir: Path, start: int, only: set[str]) -> list[Path]:
    files = []
    for path in ga_dir.iterdir():
        parsed = pdf_token(path)
        if not parsed:
            continue
        number, _suffix, token = parsed
        if only:
            if token not in only and f"{number:03d}{_suffix}" not in only:
                continue
        elif number < start:
            continue
        files.append(path)
    files.sort(key=lambda path: pdf_token(path) or (0, "", ""))
    return files


def extract_pages(pdf: Path) -> list[str]:
    try:
        text = subprocess.check_output(
            ["pdftotext", "-layout", str(pdf), "-"],
            stderr=subprocess.DEVNULL,
            timeout=180,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"pdftotext fehlgeschlagen ({exc.returncode})") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("pdftotext Zeitüberschreitung") from exc
    parts = text.decode("utf-8", errors="replace").split("\f")
    if parts and not parts[-1].strip():
        parts.pop()
    return [apply_safe_replacements(part) for part in parts]


def toc_like(page: str) -> bool:
    lines = [line.strip() for line in page.splitlines() if line.strip()]
    if not lines:
        return False
    leaders = 0
    for line in lines:
        if re.search(r"\.{4,}", line):
            leaders += 1
        elif re.search(r"\s\d{1,4}$", line) and len(line) < 140:
            leaders += 1
    return leaders >= 3


def find_toc_indexes(pages: list[str]) -> set[int]:
    start = None
    for index, page in enumerate(pages[:25]):
        head = "\n".join(page.splitlines()[:8])
        if re.search(r"(?im)^\s*(INHALT|INHALTSVERZEICHNIS)\s*$", head):
            start = index
            break
    if start is None:
        return set()
    indexes = {start}
    cursor = start + 1
    while cursor < len(pages) and cursor < start + 20 and toc_like(pages[cursor]):
        indexes.add(cursor)
        cursor += 1
    return indexes


def squash(text: str) -> str:
    text = re.sub(r"-\s*\n\s*", "", text)
    return re.sub(r"\s+", " ", text)


def month_index(name: str) -> int | None:
    key = re.sub(r"\s+", "", name.lower().replace("ä", "a"))
    aliases = {
        "januar": 1,
        "februar": 2,
        "febtuar": 2,
        "marz": 3,
        "maerz": 3,
        "mar%": 3,
        "mar£": 3,
        "mar^": 3,
        "april": 4,
        "4prz7": 4,
        "mai": 5,
        "juni": 6,
        "juli": 7,
        "august": 8,
        "augast": 8,
        "september": 9,
        "oktober": 10,
        "november": 11,
        "dezember": 12,
    }
    return aliases.get(key)


def parse_year_span(datum: str) -> tuple[int, int] | None:
    """„1903, 1904“: die Stunde ist einem der beiden Jahre zugeordnet."""
    match = re.fullmatch(r"(\d{4})\s*,\s*(\d{4})", datum.strip())
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def parse_datum(datum: str) -> tuple[int, int, int] | None:
    text = re.sub(r"\s*\(\s*\?\s*\)\s*$", "", datum.strip())
    match = re.fullmatch(r"(\d{1,2})\.(\d{1,2})\.(\d{4})", text)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def long_datum(datum: str) -> str:
    parsed = parse_datum(datum)
    if not parsed:
        return datum
    day, month, year = parsed
    return f"{day}. {MONTHS[month - 1]} {year}"


def mismatch_line(lecture: Lecture) -> str:
    placed = f"{lecture.ort}, {long_datum(lecture.datum)}" if lecture.ort else long_datum(lecture.datum)
    return f"  stimmt nicht: {placed} {lecture.datum} {lecture.id} {lecture.titel}"


# Ganze Wörter, die es so nicht gibt. „Darnach“ bleibt stehen: das ist ein Adverb.
# Eine verwechselte Jahresziffer wird nicht ersetzt; sie gilt nur zusammen mit Tag, Monat und Ort.
SAFE_WORD_REPLACEMENTS = (
    ("Dörnach", "Dornach"),
    ("Dornadh", "Dornach"),
    ("Züridi", "Zürich"),
    ("Bem", "Bern"),
)
DATE_TOKEN_RE = re.compile(
    rf"\b({DAY_TOKEN}){DAY_SEP}{MONTH_NOISE}({MONTH_PATTERN})\b",
    re.IGNORECASE,
)


def apply_safe_replacements(text: str) -> str:
    """Sichere OCR-Formen im Seitentext, bevor Überschriften und der Katalog verglichen werden."""
    for src, dst in SAFE_WORD_REPLACEMENTS:
        text = re.sub(rf"\b{re.escape(src)}\b", dst, text, flags=re.IGNORECASE)
    return DATE_TOKEN_RE.sub(_replace_date_token, text)


def _replace_date_token(match: re.Match[str]) -> str:
    day = ocr_day(match.group(1))
    if day is None:
        return match.group(0)
    return f"{day}. {match.group(2)}"


def ocr_day(token: str) -> int | None:
    """„l“ und „I“ im Tag sind die Ziffer 1.

    Ein unmöglicher Tag mit einer verwechselten Ziffer wird der gültige Tag:
    „75“ ist der 15., weil 7 statt 1 gelesen wurde.
    """
    mapped = re.sub(r"\s+", "", token).replace(")", "3")
    mapped = (
        mapped.replace("l", "1")
        .replace("I", "1")
        .replace("i", "1")
        .replace("|", "1")
        .replace("/", "1")
        .replace("o", "0")
        .replace("O", "0")
        .replace("L", "2")
        .replace("Z", "3")
        .replace("z", "3")
        .replace("f", "5")
        .replace("F", "5")
        .replace("$", "5")
    )
    if mapped in {"J", "j"}:
        return 5
    # „jo“ ist 30: das j ist eine 3, nicht die alleinstehende 5.
    if len(mapped) > 1:
        mapped = mapped.replace("j", "3").replace("J", "3")
    if mapped in {"Z", "z"}:
        return 3
    if mapped == "!":
        return 1
    if mapped in {"N", "n", "U", "u"}:
        return 11
    if not mapped.isdigit():
        return None
    day = int(mapped)
    if 1 <= day <= 31:
        return day
    valid: set[int] = set()
    for index, char in enumerate(mapped):
        for src, dst in CONFUSABLE_DIGITS:
            if char != src:
                continue
            number = int(mapped[:index] + dst + mapped[index + 1 :])
            if 1 <= number <= 31:
                valid.add(number)
    if len(valid) == 1:
        return valid.pop()
    return None


def days_compatible(left: int, right: int) -> bool:
    """Eine Ziffer des Tages ist verwechselt: „7. April“ ist der 1. April."""
    if left == right:
        return True
    a, b = f"{left:02d}", f"{right:02d}"
    diffs = [(x, y) for x, y in zip(a, b) if x != y]
    return len(diffs) == 1 and diffs[0] in CONFUSABLE_DIGITS


def years_compatible(left: int, right: int) -> bool:
    if left == right:
        return True
    a, b = f"{left:04d}", f"{right:04d}"
    diffs = [(x, y) for x, y in zip(a, b) if x != y]
    if len(diffs) == 1 and diffs[0] in CONFUSABLE_DIGITS:
        return True
    # „7972“ ist 1912: beide Einsen wurden als 7 gelesen.
    return len(diffs) == 2 and all(set(pair) == {"1", "7"} for pair in diffs)


def year_pattern(year: int) -> str:
    text = f"{year:04d}"
    options = {text}
    for index, char in enumerate(text):
        for src, dst in CONFUSABLE_DIGITS:
            if char == src:
                options.add(text[:index] + dst + text[index + 1 :])
    return "(?:" + "|".join(sorted(options)) + ")"


def day_pattern(day: int) -> str:
    parts: list[str] = []
    for char in str(day):
        if char == "1":
            parts.append(r"(?:1|[lIi|])")
        elif char == "0":
            parts.append(r"(?:0|[oO])")
        else:
            parts.append(char)
    return "".join(parts)


def datum_regex(datum: str) -> re.Pattern[str] | None:
    parsed = parse_datum(datum)
    if not parsed:
        return None
    day, month, year = parsed
    month_name = MONTHS[month - 1]
    if month_name == "März":
        month_pat = r"(?:März|Maerz|Marz)"
    else:
        month_pat = month_name
    return re.compile(
        rf"\b{day_pattern(day)}{DAY_SEP}{MONTH_NOISE}{month_pat}[ \t]+{year_pattern(year)}\b",
        re.IGNORECASE,
    )


# „vom 8. Februar 1906“ ist das Datum des Zeitungsberichts, kein Ort.
PLACE_FUNCTION = frozenset({
    "vom",
    "von",
    "den",
    "dem",
    "der",
    "die",
    "das",
    "des",
    "am",
    "im",
    "zum",
    "zur",
    "und",
    "aus",
    "bei",
    "bis",
    "wie",
    "nach",
    "vor",
    "für",
    "fuer",
    "über",
    "ueber",
    "mit",
    "auf",
    "ein",
    "eine",
    "einem",
    "einen",
    "in",
    "an",
    "zu",
    "ab",
    # „Montag, den 5. November 1917“ ist die Ankündigung eines Termins, kein Ort.
    "montag",
    "dienstag",
    "mittwoch",
    "donnerstag",
    "freitag",
    "samstag",
    "sonnabend",
    "sonntag",
    # „Gestorben am 17. Februar 1907“ ist ein Sterbedatum, kein Ort.
    "gestorben",
})


def _place_ok(place: str) -> bool:
    """Ein Ort ist kurz. „sprach Dr. Rudolf Steiner am“ ist ein Satz, kein Ort."""
    place = place.strip(" ,\\;")
    if not place:
        return True
    if PLACE_NOISE_RE.search(place) or fold_place(place) in PLACE_FUNCTION:
        return False
    words = re.findall(r"[A-Za-zÄÖÜäöüß]+", place)
    if len(words) > 4:
        return False
    if words and fold_place(words[-1]) in PLACE_FUNCTION:
        return False
    # „… sei (Berlin, 2. Januar 1906).“ ist ein Satz. „Kristiania (Oslo)“ bleibt ein Ort.
    if "(" in place and len(re.findall(r"[A-Za-zÄÖÜäöüß]+", place.split("(")[0])) > 1:
        return False
    # „Basler Volksblatt, 4. September 1920“ ist eine Zeitung, kein Ort.
    if re.search(r"(blatt|zeitung|courier|anzeiger)$", fold_place(place)):
        return False
    return True


def ocr_year(token: str) -> int | None:
    """„191$“ ist 1915, „19/j“ ist 1915. Ein Schrägstrich in der Jahreszahl ist die 1."""
    if token == "0":
        return 0
    mapped = (
        token.replace("$", "5")
        .replace("/", "1")
        .replace("j", "5")
        .replace("J", "5")
        .replace("l", "1")
        .replace("I", "1")
        .replace("i", "1")
        .replace("|", "1")
    )
    if len(mapped) == 4 and mapped.isdigit():
        return int(mapped)
    return None


def _heading_from_match(place: str, day_token: str, month_name: str, year_token: str) -> tuple[str, int, int, int] | None:
    place = place.strip(" ,\\;")
    if not _place_ok(place):
        return None
    month = month_index(month_name)
    day = ocr_day(day_token)
    year = ocr_year(year_token)
    if month is None or day is None or year is None:
        return None
    return place, day, month, year


# „9. Mai 1922“ ist nur das Datum. „8. September 1919, am Tage nach …“ ist ein Satz.
_BARE_DATE_RE = re.compile(
    rf"^({DAY_TOKEN}){DAY_SEP}{MONTH_NOISE}({MONTH_PATTERN}){MONTH_YEAR_GAP}"
    r"(?:\(\s*)?([0-9$/jJlIi|]{4})\s*\??\s*\)?"
    r"(?:\s*[*'´′'`^]+)?"
    r"(?:\s*\(\s*(?:Karfreitag|Ostern|Ostersonntag|Ostermontag|Karsamstag|Pfingsten|Pfingstsonntag|Pfingstmontag|Autoreferat)\s*\))?"
    r"(?:,\s*(?:vormittags|nachmittags|abends))?"
    r"(?:\s*\(\s*(?:morgens|vormittags|mittags|nachmittags|abends)\s*\))?"
    r"(?:,\s*\d{1,2}\s*Uhr(?:\s*\d{1,2})?(?:\s*(?:morgens|vormittags|mittags|nachmittags|abends))?)?"
    r"\s*$",
    re.IGNORECASE,
)


def _heading_lines_exact(line: str) -> list[tuple[str, int, int, int]]:
    two = TWO_MONTHS_RE.match(line)
    if two:
        first = _heading_from_match(two.group(1) or "", two.group(2), two.group(3), two.group(6))
        second = _heading_from_match(two.group(1) or "", two.group(4), two.group(5), two.group(6))
        return [hit for hit in (first, second) if hit is not None]
    oder = ODER_RE.match(line)
    if oder:
        first = _heading_from_match(oder.group(1) or "", oder.group(2), oder.group(4), oder.group(5))
        second = _heading_from_match(oder.group(1) or "", oder.group(3), oder.group(4), oder.group(5))
        return [hit for hit in (first, second) if hit is not None]
    combined = COMBINED_HEADING_RE.match(line)
    if combined:
        first = _heading_from_match(combined.group(1) or "", combined.group(2), combined.group(4), combined.group(5))
        second = _heading_from_match(combined.group(1) or "", combined.group(3), combined.group(4), combined.group(5))
        return [hit for hit in (first, second) if hit is not None]
    bracket = COMBINED_BRACKET_RE.match(line)
    if bracket:
        place = bracket.group(5) or ""
        first = _heading_from_match(place, bracket.group(1), bracket.group(3), bracket.group(4))
        second = _heading_from_match(place, bracket.group(2), bracket.group(3), bracket.group(4))
        return [hit for hit in (first, second) if hit is not None]
    match = HEADING_LINE_RE.match(line)
    if match:
        place = match.group(1) or ""
        # „8. September 1919, am Tage nach …“ ist ein Satz, kein Beginn.
        # „9. Mai 1922“ allein bleibt der Beginn.
        if not place.strip() and _BARE_DATE_RE.match(line.strip()) is None:
            match = None
        else:
            hit = _heading_from_match(place, match.group(2), match.group(3), match.group(4))
            return [hit] if hit else []
    yearless = YEARLESS_HEADING_RE.match(line)
    if not yearless:
        return []
    hit = _heading_from_match(yearless.group(1) or "", yearless.group(2), yearless.group(3), "0")
    return [hit] if hit else []


# „Aus dem Vortrag Basel, 3. Februar 1909:“ und „Aus dem Vor trag Kassel, …“.
# Der Doppelpunkt und ein Zusatz hinter dem Jahr bleiben außen vor.
# „Notizen aus dem Vortrag, Rom, 12. April 1910“ und „Aus dem Vor trag …“.
AUS_VORTRAG_RE = re.compile(
    rf"^(?:Notizen\s+aus\s+dem\s+|Aus\s+(?:dem\s+)?)Vor\s*trag\s*[,;:*]?\s*"
    # „Aus Vortrag (Fragenbeantwortungen) Dörnach, 22. April 1924“
    rf"(?:\([^)]*\)\s*)?"
    rf"({PLACE_HEAD}){PLACE_GAP}{DATE_INTRO}"
    rf"({DAY_TOKEN}){DAY_SEP}{MONTH_NOISE}({MONTH_PATTERN}){MONTH_YEAR_GAP}{YEAR_TAIL}",
    re.IGNORECASE,
)
# „Aus der Diskussion vom 6. August 1923“: der Ort fehlt, der Tag steht im Anhang.
DISKUSSION_VOM_RE = re.compile(
    rf"^Aus\s+der\s+Diskussion\s+vom\s+"
    rf"({DAY_TOKEN}){DAY_SEP}{MONTH_NOISE}({MONTH_PATTERN}){MONTH_YEAR_GAP}{YEAR_TAIL}$",
    re.IGNORECASE,
)
# „Aus dem Vortrag vom 21. September 1920“: der Ort fehlt.
AUS_VOM_RE = re.compile(
    rf"^Aus\s+dem\s+Vor\s*trag\s+vom\s+"
    rf"({DAY_TOKEN}){DAY_SEP}{MONTH_NOISE}({MONTH_PATTERN}){MONTH_YEAR_GAP}{YEAR_TAIL}$",
    re.IGNORECASE,
)
# „durch Rudolf Steiner am 25. Dezember 1923, 10 Uhr vormittags“: kein Ortsname.
DURCH_STEINER_RE = re.compile(
    rf"^durch\s+Rudolf\s+Steiner\s+am\s+"
    rf"({DAY_TOKEN}){DAY_SEP}{MONTH_NOISE}({MONTH_PATTERN}){MONTH_YEAR_GAP}{YEAR_TAIL}$",
    re.IGNORECASE,
)
ESOTERIC_RE = re.compile(
    rf"^Aus\s+einer\s+esoterischen\b.*?\bin\s+({PLACE_HEAD}){PLACE_GAP}{DATE_INTRO}"
    rf"({DAY_TOKEN}){DAY_SEP}{MONTH_NOISE}({MONTH_PATTERN}){MONTH_YEAR_GAP}{YEAR_TAIL}",
    re.IGNORECASE,
)
# „vom 6./7. Januar 1906“ ist der 6. und der 7. Januar, auch am Ende einer längeren Zeile.
VOM_SLASH_RE = re.compile(
    rf"vom\s+({DAY_TOKEN})\./\s*({DAY_TOKEN})\.\s*{MONTH_NOISE}({MONTH_PATTERN}){MONTH_YEAR_GAP}{YEAR_TAIL}$",
    re.IGNORECASE,
)
VOM_DATE_RE = re.compile(
    rf"vom\s+({DAY_TOKEN}){DAY_SEP}{MONTH_NOISE}({MONTH_PATTERN}){MONTH_YEAR_GAP}{YEAR_TAIL}$",
    re.IGNORECASE,
)
# „Zweiter Vortrag vom 15. Oktober 1922“ und „vom 13. Januar 1906“.
VOM_ORDINAL_RE = re.compile(
    rf"^(?:(?:{ORDINAL})\s+Vortrag\s+)?vom\s+"
    rf"({DAY_TOKEN}){DAY_SEP}{MONTH_NOISE}({MONTH_PATTERN}){MONTH_YEAR_GAP}{YEAR_TAIL}$",
    re.IGNORECASE,
)
# „in Stuttgart am 8: September 1921“
# „Ansprache in Dornach am 6. Januar 1923 nach dem Brande des Goetheanum“
# „Ansprache und Fragenbeantwortung in Stuttgart am 14. Februar 1923“
ADDRESS_RE = re.compile(
    rf"^Ansprache(?:\s+und\s+[A-Za-zÄÖÜäöüß]+)?\s+in\s+"
    rf"({PLACE_HEAD})\s+am\s+"
    rf"({DAY_TOKEN}){DAY_SEP}{MONTH_NOISE}({MONTH_PATTERN}){MONTH_YEAR_GAP}{YEAR_TAIL}"
    rf"(?:\s+\S.*)?$",
    re.IGNORECASE,
)
# „Erster Tag, 18. Januar 1914“ und „Montag, den 19. Januar 1914, 1 Uhr mittags“.
# Der Wochentag ist kein Ort. Steht er allein, gilt das Datum ohne Ort.
WEEKDAY_NAME = "Montag|Dienstag|Mittwoch|Donnerstag|Freitag|Samstag|Sonnabend|Sonntag"
STUNDE_WORD = (
    "Erste|Zweite|Dritte|Vierte|Fünfte|Fuenfte|Sechste|Siebte|Achte|Neunte|"
    "Zehnte|Elfte|Zwölfte|Dreizehnte|Vierzehnte|Fünfzehnte|Fuenfzehnte|Sechzehnte|"
    "Siebzehnte|Achtzehnte|Neunzehnte|Zwanzigste|Einundzwanzigste|Zweiundzwanzigste"
)
# „Konferenz vom Donnerstag 25. September 1919, 8.30 Uhr“ ist der 25. September 1919.
# „8, September“, „19201“, „1922t“ und „Dienstag'18.“ bleiben derselbe Tag.
KONFERENZ_VOM_RE = re.compile(
    rf"^Konferenz\s+vom\s+(?:(?:{WEEKDAY_NAME})\s*['’]?\s*)?"
    rf"({DAY_TOKEN}){DAY_SEP}{MONTH_NOISE}({MONTH_PATTERN})\s+(\d{{4}})",
    re.IGNORECASE,
)
# „am 20. August 1919 in Stuttgart“: der Ort steht hinter dem Jahr.
AM_DATE_IN_PLACE_RE = re.compile(
    rf"^am\s+({DAY_TOKEN}){DAY_SEP}{MONTH_NOISE}({MONTH_PATTERN})\s+(\d{{4}})\s+in\s+({PLACE_HEAD})\s*[*'´′'`^]*\s*$",
    re.IGNORECASE,
)
# „Fünfte Stunde, Montag, 23. August 1915, vormittags“ ist der 23. August 1915.
STUNDE_HEAD_RE = re.compile(
    rf"^(?:{STUNDE_WORD})\s+Stunde\s*,\s*(?:{WEEKDAY_NAME})\s*,\s*"
    rf"({DAY_TOKEN}){DAY_SEP}{MONTH_NOISE}({MONTH_PATTERN})\s+(\d{{4}})"
    rf"(?:,\s*(?:vormittags|nachmittags|abends))?\s*$",
    re.IGNORECASE,
)
# „29. August 1915, Domach“ und „29. August 1915, 15 Uhr, Dörnach“: der Ort steht hinter dem Datum.
TAIL_PLACE_RE = re.compile(
    rf"^({DAY_TOKEN}){DAY_SEP}{MONTH_NOISE}({MONTH_PATTERN})\s+(\d{{4}})"
    rf"(?:\s*,\s*\d{{1,2}}\s*Uhr)?"
    rf"\s*,\s*({PLACE_HEAD})\s*$",
    re.IGNORECASE,
)
WEEKDAY_HEAD_RE = re.compile(
    rf"^(?:(?:{ORDINAL})\s+Tag|{WEEKDAY_NAME})\s*,\s*(?:den\s+)?"
    rf"({DAY_TOKEN}){DAY_SEP}{MONTH_NOISE}({MONTH_PATTERN})\s+(\d{{4}})"
    rf"(?:,\s*(?:gegen\s+)?\d{{1,2}}\s*Uhr(?:\s*(?:morgens|vormittags|mittags|nachmittags|abends))?)?"
    # „Mittwoch, 6. September 1922, nachmittags“ und „…, vormittags und nachmittags“.
    rf"(?:,\s*(?:vor-\s*und\s+)?(?:vormittags|nachmittags|abends|mittags|morgens)"
    rf"(?:\s+und\s+(?:vormittags|nachmittags|abends))?)?"
    rf"\s*,?\s*$",
    re.IGNORECASE,
)
# „Sehr geehrter Herr …“: die Zeile davor ist der Briefkopf, kein Vortragsanfang.
# „Liebe Leidtragende!“ und „Meine lieben Schwestern“ sind der Beginn der Rede, kein Brief.
SALUTATION_RE = re.compile(
    r"^(?:Sehr\s+geehrte\w*|Geehrte\w*|Hochverehrte\w*|Hochzuverehrende\w*"
    r"|Verehrte\w*"
    r"|Mein(?:e)?\s+sehr\s+verehrte\w*"
    r"|Mein(?:e)?\s+liebe[rs]?\s+(?:Herr|Frau|Fräulein)\b"
    r"|Liebe[rs]?\s+(?:Herr|Frau|Fräulein)\b"
    r"|Werte[rs]?\b)",
    re.IGNORECASE,
)
# „am 17. Juni 1924“ allein auf der Folgezeile.
# Nur der umbrochene Tag, ohne Uhrzeit. „am 28. Dezember 1923, nachmittags …“
# ist schon die ganze Überschrift und darf nicht an die Titelzeile davor gehängt werden.
AM_DATE_LINE_RE = re.compile(
    rf"^am\s+({DAY_TOKEN}){DAY_SEP}{MONTH_NOISE}({MONTH_PATTERN}){MONTH_YEAR_GAP}"
    rf"(?:\(\s*)?(\d{{4}})\s*\??\s*\)?\s*\.?\s*$",
    re.IGNORECASE,
)
# „am 25. Dezember 1923, 10 Uhr vormittags“, „am Sonntag, 23. Dezember 1923“,
# „am 24. Dezember, 10 Uhr vormittags“ ohne Jahr, „vom 24. Dezember 1923, 11 Uhr 15 vormittags“.
_CLOCK = r"\d{1,2}\s*Uhr(?:\s*\d{1,2})?(?:\s*(?:morgens|vormittags|mittags|nachmittags|abends))?"
_DAYTIME = r"morgens|vormittags|mittags|nachmittags|abends"
AM_SESSION_RE = re.compile(
    rf"^(?:Vortrag\s+von\s+Rudolf\s+Steiner\s+)?(?:am|vom)\s+"
    rf"(?:(?:{WEEKDAY_NAME})\s*,\s*)?"
    rf"({DAY_TOKEN}){DAY_SEP}{MONTH_NOISE}({MONTH_PATTERN})"
    rf"(?:\s+(\d{{4}}))?"
    rf"(?:,\s*(?:{_CLOCK}|(?:{_DAYTIME})(?:\s+\d{{1,2}}\s*Uhr(?:\s*\d{{1,2}})?)?))?"
    rf"(?:\s*,?\s*im\s+[A-Za-zÄÖÜäöüß]+)?"
    rf"(?:\s*,\s*fand\b.*)?\s*$",
    re.IGNORECASE,
)
# „Aus einem Vortrag in Berlin vom 28. April 1914“.
AUS_EINEM_RE = re.compile(
    rf"^Aus\s+einem\s+Vortrag\s+in\s+({PLACE_HEAD})\s+vom\s+"
    rf"({DAY_TOKEN}){DAY_SEP}{MONTH_NOISE}({MONTH_PATTERN}){MONTH_YEAR_GAP}{YEAR_TAIL}\s*$",
    re.IGNORECASE,
)
# „Berlin, zwischen dem 4. und 14. März 1907“: der Vortrag ist der spätere Tag.
BETWEEN_RE = re.compile(
    rf"^({PLACE_HEAD}){PLACE_GAP}zwischen\s+dem\s+"
    rf"({DAY_TOKEN})\.\s+und\s+({DAY_TOKEN})\.\s+"
    rf"{MONTH_NOISE}({MONTH_PATTERN}){MONTH_YEAR_GAP}{YEAR_TAIL}$",
    re.IGNORECASE,
)
# „Ansprache Rudolf Steiners am 25. März 1907 in Berlin“: der Ort steht hinter dem Datum.
AM_IN_PLACE_RE = re.compile(
    rf"^Ansprache\b.*?\bam\s+"
    rf"({DAY_TOKEN}){DAY_SEP}{MONTH_NOISE}({MONTH_PATTERN}){MONTH_YEAR_GAP}{YEAR_TAIL}"
    rf"\s+in\s+({PLACE_HEAD})\s*$",
    re.IGNORECASE,
)
# „… in Koberwitz am 17. Juni 1924“ am Zeilenende.
IN_AM_TAIL_RE = re.compile(
    rf"\bin\s+({PLACE_HEAD})\s+am\s+"
    rf"({DAY_TOKEN}){DAY_SEP}{MONTH_NOISE}({MONTH_PATTERN}){MONTH_YEAR_GAP}{YEAR_TAIL}$",
    re.IGNORECASE,
)


def _pair_from_days(day_a: str, day_b: str, month_name: str, year_token: str) -> list[tuple[str, int, int, int]]:
    first = _heading_from_match("", day_a, month_name, year_token)
    second = _heading_from_match("", day_b, month_name, year_token)
    return [hit for hit in (first, second) if hit is not None]


# „Berlin, Karfreitag, 21. April 1905“: ein Wort zwischen Ort und Datum, der Ort bleibt Berlin.
# „Den Haag, Ostersonntag, 23. März 1913“: der Festtag ist kein Ort.
FEAST_WORD = re.compile(
    r"Ostersonntag|Ostermontag|Karfreitag|Karsamstag|Pfingstsonntag|Pfingstmontag",
    re.IGNORECASE,
)
INSERTED_WORD_RE = re.compile(
    rf"^{LECTURE_PREFIX}({PLACE_HEAD}){PLACE_GAP}"
    rf"([A-ZÄÖÜ][A-Za-zÄÖÜäöüß-]*)"
    rf"{PLACE_GAP}{DATE_INTRO}"
    rf"({DAY_TOKEN}){DAY_SEP}{MONTH_NOISE}({MONTH_PATTERN}){MONTH_YEAR_GAP}{YEAR_TAIL}$",
    re.IGNORECASE,
)


def _inserted_word_heading(line: str) -> list[tuple[str, int, int, int]]:
    match = INSERTED_WORD_RE.match(line.strip())
    if not match:
        return []
    place = match.group(1) or ""
    middle = match.group(2) or ""
    words = re.findall(r"[A-Za-zÄÖÜäöüß]+", place)
    # „Basler Volksblatt, 4. September 1920“: das Blatt ist kein eingeschobenes Wort.
    if re.search(r"(blatt|zeitung|courier|anzeiger)$", fold_place(middle)):
        return []
    # „Stuttgart, Dienstag 23. Dezember 1919“ steht in der Zeittafel.
    # „Stuttgart, Dienstag, den 22. April 1919“ ist der Vortrag: „den“ folgt dem Wochentag.
    if re.match(rf"(?:{WEEKDAY_NAME})", middle, re.IGNORECASE) and not re.search(
        rf"{re.escape(middle)}\s*,\s*den\s+",
        line,
        re.IGNORECASE,
    ):
        return []
    feast = FEAST_WORD.fullmatch(middle) is not None
    if len(words) != 1 and not (feast and 1 <= len(words) <= 3):
        return []
    hit = _heading_from_match(place, match.group(3), match.group(4), match.group(5))
    return [hit] if hit else []


def _bracket_range(line: str) -> list[tuple[str, int, int, int]]:
    """„[zwischen 17. und 20.]“ und „[zwischen 17.-20.]“ nennen jeden Tag der Spanne."""
    match = BRACKET_RANGE_RE.match(line.strip())
    if not match:
        return []
    first = ocr_day(match.group(2))
    last = ocr_day(match.group(3))
    if first is None or last is None or not 0 < last - first <= 5:
        return []
    found: list[tuple[str, int, int, int]] = []
    for day in range(first, last + 1):
        hit = _heading_from_match(match.group(1) or "", str(day), match.group(4), match.group(5))
        if hit:
            found.append(hit)
    return found


def _held_on(line: str) -> list[tuple[str, int, int, int]]:
    """„Gehalten in Dornach am 27. Mai 1923, 23. Oktober 1923, 3. Januar 1924“."""
    match = HELD_ON_RE.match(line.strip())
    if not match:
        return []
    found: list[tuple[str, int, int, int]] = []
    for day_token, month_name, year_token in HELD_DATE_RE.findall(match.group(2)):
        hit = _heading_from_match(match.group(1) or "", day_token, month_name, year_token)
        if hit:
            found.append(hit)
    return found if len(found) >= 2 else []


def heading_lines(line: str) -> list[tuple[str, int, int, int]]:
    # „'Nürnberg, …“ und „[Karlsruhe], …“: Anführungszeichen und Klammern gehören nicht zum Ort.
    # „Berlin, 2. Januar 1905"“: das Anführungszeichen hinter dem Jahr auch nicht.
    line = line.lstrip("'\"„“»").rstrip("'\"“”»")
    # „CJffentlidier“, „Offentlidier“, „öffentlidier“ sind „öffentlicher“:
    # Ö wurde CJ oder O, ch wurde di.
    line = re.sub(
        r"(?:CJ|[OoöÖ])ffentl(?:ich|idi)er",
        "öffentlicher",
        line,
    )
    # „[Karsamstag]“ und „7. Juni [Pfingstsamstag] 1919“: der Festtag ändert das Datum nicht.
    line = re.sub(
        r"\s*[\[(]\s*(?:Karfreitag|Karsamstag|Ostern|Ostersonntag|Ostermontag|Pfingsten|Pfingstsamstag|Pfingstsonntag|Pfingstmontag)\s*[\])]",
        " ",
        line,
        flags=re.IGNORECASE,
    )
    # „stunde in Basel am 1. Juni 1914.“ ist der umbrochene Rest, kein Beginn.
    if re.match(r"stunde\b", line, re.IGNORECASE):
        return []
    line = re.sub(r"^\[([^\]]+)\]", r"\1", line)
    # „Berlin, 20. Oktober 1907, Motzstr. 17“: die Straße hinter dem Jahr ist kein neues Datum.
    line = re.sub(r",\s*[A-ZÄÖÜ][A-Za-zÄÖÜäöüß.]{0,24}\s+\d+\s*$", "", line)
    # „Kristiania, (Oslo), 5. Juni 1912“ ist Kristiania (Oslo) am 5. Juni 1912.
    line = re.sub(r",\s*(\([^)]+\))", r" \1", line)
    # „* München, …“ und „Instmktionsstunde“: Aufzählungsstern und verlesenes ru.
    line = re.sub(r"^\*\s+", "", line)
    line = re.sub(r"Instmktionsstunde", "Instruktionsstunde", line, flags=re.IGNORECASE)
    # „V. Dörnach, …“ ist Vortrag. „E.S. Berlin“ und „E,S. Köln“ sind die esoterische Stunde.
    line = re.sub(r"^(?:V\.|E[.,]\s*S\.)\s*", "", line)
    # „Fortsetzung: Dörnach 28. September 1918“ ist der Beginn dieses Tages.
    line = re.sub(r"^Fortsetzung:\s*", "", line, flags=re.IGNORECASE)
    # „Programm zur Aufführung Dornach, …“ und „Ansprache zur Eurythmie, Dornach, …“:
    # der Ort steht hinter der Bezeichnung.
    line = re.sub(
        r"^(?:Programm\s+zur\s+(?:Aus|Auf)f[uü]hrung(?:\s+vom)?"
        r"|Plakat\s+für\s+die\s+Aufführungen?"
        r"|Notizen\s+zur\s+Ansprache"
        r"|Ansprache\s+zur\s+Eurythmie"
        r"|Votum"
        r"|Prospekt)\s*,?\s*",
        "",
        line,
        flags=re.IGNORECASE,
    )
    spanned = _bracket_range(line)
    if spanned:
        return spanned
    # „[23.] Februar“ und „[15. oder 16.] Dezember“: die Klammer macht den Tag unsicher.
    line = re.sub(r"\[(\d{1,2}\.)\]", r"\1", line)
    line = re.sub(
        r"\[(\d{1,2}\.)\s+oder\s+(\d{1,2}\.)\]",
        r"\1 oder \2",
        line,
    )
    line = re.sub(r"^Esoterische\s+Stunde\s+in\s+", "", line, flags=re.IGNORECASE)
    weihnacht = WEIHNACHT_RE.match(line.strip())
    if weihnacht:
        hit = _heading_from_match(weihnacht.group(1) or "", "24", "Dezember", weihnacht.group(2))
        return [hit] if hit else []
    held = _held_on(line)
    if held:
        return held
    konferenz = KONFERENZ_VOM_RE.match(line.strip())
    if konferenz:
        hit = _heading_from_match("", konferenz.group(1), konferenz.group(2), konferenz.group(3))
        return [hit] if hit else []
    stunde = STUNDE_HEAD_RE.match(line.strip())
    if stunde:
        hit = _heading_from_match("", stunde.group(1), stunde.group(2), stunde.group(3))
        return [hit] if hit else []
    tail_place = TAIL_PLACE_RE.match(line.strip())
    if tail_place:
        hit = _heading_from_match(
            tail_place.group(4), tail_place.group(1), tail_place.group(2), tail_place.group(3)
        )
        return [hit] if hit else []
    weekday = WEEKDAY_HEAD_RE.match(line.strip())
    if weekday:
        hit = _heading_from_match("", weekday.group(1), weekday.group(2), weekday.group(3))
        return [hit] if hit else []
    in_place = AM_DATE_IN_PLACE_RE.match(line.strip())
    if in_place:
        hit = _heading_from_match(in_place.group(4), in_place.group(1), in_place.group(2), in_place.group(3))
        return [hit] if hit else []
    session = AM_SESSION_RE.match(line.strip())
    if session:
        hit = _heading_from_match("", session.group(1), session.group(2), session.group(3) or "0")
        return [hit] if hit else []
    durch = DURCH_STEINER_RE.match(line.strip())
    if durch:
        hit = _heading_from_match("", durch.group(1), durch.group(2), durch.group(3))
        return [hit] if hit else []
    aus_vom = AUS_VOM_RE.match(line.strip())
    if aus_vom:
        hit = _heading_from_match("", aus_vom.group(1), aus_vom.group(2), aus_vom.group(3))
        return [hit] if hit else []
    diskussion = DISKUSSION_VOM_RE.match(line.strip())
    if diskussion:
        hit = _heading_from_match("", diskussion.group(1), diskussion.group(2), diskussion.group(3))
        return [hit] if hit else []
    parsed = _heading_lines_exact(line)
    if parsed:
        return parsed
    inserted = _inserted_word_heading(line)
    if inserted:
        return inserted
    aus = AUS_VORTRAG_RE.match(line.strip())
    if aus:
        hit = _heading_from_match(aus.group(1) or "", aus.group(2), aus.group(3), aus.group(4))
        return [hit] if hit else []
    einem = AUS_EINEM_RE.match(line.strip())
    if einem:
        hit = _heading_from_match(einem.group(1) or "", einem.group(2), einem.group(3), einem.group(4))
        return [hit] if hit else []
    esoteric = ESOTERIC_RE.match(line.strip())
    if esoteric:
        hit = _heading_from_match(esoteric.group(1) or "", esoteric.group(2), esoteric.group(3), esoteric.group(4))
        return [hit] if hit else []
    slash = VOM_SLASH_RE.search(line.strip())
    if slash:
        return _pair_from_days(slash.group(1), slash.group(2), slash.group(3), slash.group(4))
    vom = VOM_ORDINAL_RE.match(line.strip())
    if vom:
        hit = _heading_from_match("", vom.group(1), vom.group(2), vom.group(3))
        return [hit] if hit else []
    address = ADDRESS_RE.match(line.strip())
    if address:
        hit = _heading_from_match(address.group(1) or "", address.group(2), address.group(3), address.group(4))
        return [hit] if hit else []
    # „… in Koberwitz am 17. Juni 1924“: der Ort steht vor „am“, nicht am Zeilenanfang.
    in_am = IN_AM_TAIL_RE.search(line.strip())
    if in_am:
        hit = _heading_from_match(in_am.group(1) or "", in_am.group(2), in_am.group(3), in_am.group(4))
        return [hit] if hit else []
    between = BETWEEN_RE.match(line.strip())
    if between:
        hit = _heading_from_match(between.group(1) or "", between.group(3), between.group(4), between.group(5))
        return [hit] if hit else []
    am_in = AM_IN_PLACE_RE.match(line.strip())
    if am_in:
        hit = _heading_from_match(am_in.group(4) or "", am_in.group(1), am_in.group(2), am_in.group(3))
        return [hit] if hit else []
    # „Erster Gesprächsabend, Dörnach, 4. Oktober 1920“: die Bezeichnung vor dem Ort.
    # „Vortrag von Rudolf Steiner, Berlin, 4. Juli 1904“ bleibt ein Vortragsanfang.
    named_talk = re.match(r"^Vortrag\s+von\s+Rudolf\s+Steiner\b", line, re.IGNORECASE)
    if not named_talk and re.search(r"\b(vortrag|vgl|siehe|nach dem|anschlu[sß])\b", line, re.IGNORECASE):
        return []
    for match in re.finditer(r"[A-ZÄÖÜ]", line):
        start = match.start()
        if start == 0 or not re.search(r",\s*$", line[:start]):
            continue
        parsed = _heading_lines_exact(line[start:])
        if parsed and parsed[0][0]:
            return parsed
    return []


def lecture_heading_lines(lines: list[str]) -> list[str]:
    """Hängt eine Folgezeile an, wenn Ort oder Datum umbrechen."""
    merged: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        wraps = re.search(r"öffentlicher\s+Vortrag", line, re.IGNORECASE) and not re.search(
            r"\d{4}\s*$", line
        )
        # „… in Koberwitz“ / „am 17. Juni 1924“: der Tag steht auf der nächsten Zeile.
        # Nur „in Koberwitz“ / „am 17. Juni 1924“. „Ansprache … Baues“ / „am 20. September 1913“
        # bleibt getrennt: die zweite Zeile ist schon der ganze Beginn.
        place_tail = re.search(
            r"\bin\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöüß.'’-]*\s*$", line
        ) and not re.search(r"\d{4}", line)
        date_wrap = (
            index + 1 < len(lines)
            and AM_DATE_LINE_RE.match(lines[index + 1].strip())
            and place_tail
        )
        if (wraps or date_wrap) and index + 1 < len(lines):
            merged.append(f"{line} {lines[index + 1]}")
            index += 2
            continue
        merged.append(line)
        index += 1
    return merged


_EXCERPT_LINE = re.compile(
    r"Aus\s+(?:dem\s+)?Vor\s*trag\b|Aus\s+einer\s+esoterischen\b",
    re.IGNORECASE,
)


def _only_excerpt_dates(page: str) -> bool:
    """True, wenn jede datierte Zeile ein Auszug ist und mindestens eine vorkommt."""
    saw = False
    for raw in page.splitlines():
        line = raw.strip()
        if not heading_lines(line):
            continue
        saw = True
        if not _EXCERPT_LINE.match(line.lstrip('"„“»')):
            return False
    return saw


def _excerpt_hits(page: str) -> list[tuple[str, int, int, int]]:
    """„Aus dem Vortrag …“ und eine esoterische Stunde, auch mitten auf der Seite."""
    found: list[tuple[str, int, int, int]] = []
    for raw in page.splitlines():
        line = raw.strip()
        if not _EXCERPT_LINE.match(line):
            continue
        found.extend(heading_lines(line))
    return found


def _same_hour_continued(lines: list[str], index: int) -> bool:
    """„Instruktionsstunde …“ und darunter „(Aufzeichnungen von anderer Hand)“."""
    return (
        index + 1 < len(lines)
        and re.match(r"\(Aufzeichnungen von anderer Hand\)", lines[index + 1], re.IGNORECASE) is not None
    )


def _quoted_lecture(lines: list[str]) -> list[tuple[str, int, int, int]] | None:
    """Der Vortragstext folgt auf „im Vortrag …, dargelegt“ oder „vorangestellt“."""
    cite = re.compile(
        rf"(?:im|aus\s+dem)\s+Vortrag\s+({PLACE_HEAD}){PLACE_GAP}"
        rf"({DAY_TOKEN}){DAY_SEP}{MONTH_NOISE}({MONTH_PATTERN})\s+(\d{{4}})",
        re.IGNORECASE,
    )
    found: list[tuple[str, int, int, int]] = []
    for index, line in enumerate(lines):
        match = cite.search(line)
        if not match:
            continue
        window = line if index + 1 >= len(lines) else f"{line} {lines[index + 1]}"
        if re.search(r"vorangestellt|folgenden Ausführungen|dargelegt", window, re.IGNORECASE) is None:
            continue
        hit = _heading_from_match(match.group(1), match.group(2), match.group(3), match.group(4))
        if hit:
            found.append(hit)
    if len(found) == 1:
        return found
    return None


def _register_page(page: str) -> bool:
    """Alphabetisches Register: viele Zeilen enden mit der gedruckten Seitenzahl."""
    lines = [line.strip() for line in page.splitlines() if line.strip()]
    if len(lines) < 15:
        return False
    numbered = sum(
        1
        for line in lines
        if len(line) < 140 and re.search(r"\s\d{2,3}\s*$", line) and not re.search(r"\d{4}\s*$", line)
    )
    return numbered >= 10 and numbered / len(lines) >= 0.4


def _letter_before_salutation(lines: list[str], index: int) -> bool:
    """Nur der Briefkopf „Ort, Datum“ vor „Sehr geehrte …“.

    „Stuttgart, 25. März 1923“ und darunter „Meine sehr verehrten Anwesenden!“
    ist der Vortrag. „Meine sehr verehrten Damen und Herren!“ ebenso.
    „Verehrte Anwesende, verehrte Kommilitonen!“ ist die Anrede des Vortrags.
    """
    if index + 1 >= len(lines):
        return False
    nxt = lines[index + 1]
    if re.match(
        r"^(?:Meine?\s+sehr\s+verehrten\s+(?:Anwesenden|Damen)\b|Verehrte\s+Anwesende\b)",
        nxt,
        re.IGNORECASE,
    ):
        return False
    if SALUTATION_RE.match(nxt) is None:
        return False
    if not _bare_place_date(lines[index]):
        return False
    parsed = heading_lines(lines[index])
    return bool(parsed) and all(hit[0] for hit in parsed)


def _same_days(groups: list[list[tuple[str, int, int, int]]]) -> bool:
    signatures = {
        tuple(sorted((hit[1], hit[2], hit[3]) for hit in group)) for group in groups
    }
    return len(signatures) == 1


def _cleanest_group(groups: list[list[tuple[str, int, int, int]]]) -> list[tuple[str, int, int, int]]:
    """Bei demselben Tag bleibt die kurze Ortszeile, nicht „Programm zur Aufführung …“."""

    def rank(group: list[tuple[str, int, int, int]]) -> tuple[int, int, int, int]:
        place = group[0][0]
        bad = 1 if re.match(
            r"^(?:Programm|Plakat|Notizen|Ankündigung|Ansprache|Auszug)\b",
            place,
            re.IGNORECASE,
        ) else 0
        empty = 0 if place.strip() else 1
        return (bad, empty, len(place.split()), len(place))

    return min(groups, key=rank)


def _heading_opens_line(line: str, parsed: list[tuple[str, int, int, int]]) -> bool:
    """Die Orts- oder Datumszeile beginnt die Zeile, nicht einen Hinweis mitten im Satz."""
    stripped = line.strip().lstrip("'\"„“»«")
    if re.match(
        rf"^(?:(?:{ORDINAL})\s+Vortrag\b|Aus\s+(?:der\s+Diskussion|dem\s+Vortrag)\b|Ansprache\b|Fortsetzung\b)",
        stripped,
        re.IGNORECASE,
    ):
        return True
    place = parsed[0][0] if parsed else ""
    if place and stripped.lower().startswith(place.lower()):
        return True
    return re.match(rf"^{DAY_TOKEN}{DAY_SEP}", stripped) is not None


def _appendix_midpage(lines: list[str], index: int) -> bool:
    """Diskussion, Autoreferat oder das Datum unter „Zur Ausstellung“."""
    line = lines[index].strip()
    if DISKUSSION_VOM_RE.match(line):
        return True
    if re.search(r"\(\s*Autoreferat\s*\)", line, re.IGNORECASE):
        return True
    previous = " ".join(lines[max(0, index - 2) : index])
    return bool(re.search(r"\bZur Ausstellung\b", previous, re.IGNORECASE)) and bool(
        heading_lines(line)
    )


def heading_hits(page: str) -> list[tuple[str, int, int, int]]:
    """Ort, Tag, Monat, Jahr von Überschriftszeilen im Seitenkopf.

    Eine Liste mehrerer Datumszeilen (Titelblatt, Inhaltsverzeichnis) ist kein
    Vortragsbeginn. Mehrere Zeilen, die jede für sich nur Ort und Datum sind,
    sind dagegen jeweils ein Beginn. Eine Zeile „25. und 26. April 1907“ ist der Beginn beider Tage.
    Mehrere Auszüge „Aus dem Vortrag …“ auf einer Seite sind jeweils ein Beginn,
    wenn die Seite sonst kein Ortsdatum nennt.
    """
    if _register_page(page):
        return []
    groups: list[list[tuple[str, int, int, int]]] = []
    prefixed: list[list[tuple[str, int, int, int]]] = []
    prefix_re = re.compile(r"(?:nicht\s+)?öffentlicher\s+Vortrag\b", re.IGNORECASE)
    lines = lecture_heading_lines(
        [line.strip() for line in page.splitlines() if line.strip()]
    )
    # „Konferenz vom Donnerstag 25. September 1919, 8.30 Uhr“ ist der Beginn,
    # auch wenn die Zeile erst unter dem vorigen Text steht.
    conferences = [
        heading_lines(line)
        for line in lines
        if KONFERENZ_VOM_RE.match(line.strip()) and heading_lines(line)
    ]
    if len(conferences) == 1:
        return conferences[0]
    for index, line in enumerate(lines[:12]):
        # „Dornach, den 19. August 1915“ / „Sehr geehrter Herr Dr. Steiner!“ ist der Brief.
        # „Ansprache zur Eurythmie, …“ / „Meine sehr verehrten Anwesenden!“ bleibt der Beginn.
        if _letter_before_salutation(lines, index):
            continue
        if _same_hour_continued(lines, index):
            continue
        parsed = heading_lines(line)
        if not parsed:
            continue
        groups.append(parsed)
        if prefix_re.search(line):
            prefixed.append(parsed)
    # Ein Wochentag ohne Ort bleibt nur dann der Beginn, wenn die Seite kein Ortsdatum nennt.
    placed = [group for group in groups if any(hit[0] for hit in group)]
    if placed:
        groups = placed
    # Programmzeile und Ortszeile nennen denselben Tag. Es bleibt die Ortszeile.
    if len(groups) > 1 and _same_days(groups):
        groups = [_cleanest_group(groups)]
    # Ein Zeitungsbericht nennt oft noch „vom 20. Januar 1890“. Der Vortrag
    # beginnt an der Zeile mit „öffentlicher Vortrag“ oder mit Ort und Datum.
    if len(prefixed) == 1:
        return prefixed[0]
    if prefixed:
        return []
    groups = _drop_bare_vom(groups)
    # „Plakat …“ und „Ansprache zur Eurythmie, …“ auf einer Seite: die Ansprache ist der Beginn.
    speeches = [
        (line, heading_lines(line))
        for line in lines
        if re.match(r"Ansprache\s+zur\s+Eurythmie\b", line, re.IGNORECASE) and heading_lines(line)
    ]
    if len(speeches) == 1:
        speech_days = {(hit[1], hit[2], hit[3]) for hit in speeches[0][1]}
        others = [
            line
            for line in lines
            if heading_lines(line) and not re.match(r"Ansprache\s+zur\s+Eurythmie\b", line, re.IGNORECASE)
        ]
        poster = re.compile(r"^(?:Programm|Plakat|Ankündigung)\b", re.IGNORECASE)
        if all(
            poster.match(line)
            or {(hit[1], hit[2], hit[3]) for hit in heading_lines(line)} <= speech_days
            for line in others
        ):
            return speeches[0][1]
    several = _several_place_dates(lines)
    if several:
        return several
    # „Sonntag, 10. September“ und darunter „Montag, 11. September“ ist die Chronik, kein Beginn.
    bare_weekdays: set[tuple[int, int, int]] = set()
    for line in lines:
        if not WEEKDAY_HEAD_RE.match(line.strip()):
            continue
        for hit in heading_lines(line):
            if not hit[0]:
                bare_weekdays.add((hit[1], hit[2], hit[3]))
    if len(bare_weekdays) > 1:
        return []
    if len(groups) == 1 and not _only_excerpt_dates(page):
        return groups[0]
    if _only_excerpt_dates(page):
        return _excerpt_hits(page)
    # Mehrere Ortsdaten im Kopf sind eine Liste. Nur Wochentage ohne Ort:
    # die Ansprache mit dem Ort steht oft erst darunter.
    if groups and any(hit[0] for group in groups for hit in group):
        return []
    # Der neue Vortrag beginnt oft erst unter dem Text des vorigen, mitten auf der Seite.
    later: list[list[tuple[str, int, int, int]]] = []
    stunden: list[list[tuple[str, int, int, int]]] = []
    rest = lines[12:]
    for index, line in enumerate(rest):
        if _letter_before_salutation(rest, index):
            continue
        if _same_hour_continued(rest, index):
            continue
        parsed = heading_lines(line)
        # Eine Datumszeile ohne Ort mitten im Text ist eine Erwähnung, kein Beginn.
        # „am Sonntag, 23. Dezember 1923“ unter der Einleitung bleibt der Beginn.
        # „V. 15. Oktober 1911“ unter dem Spruch ist die Quellenzeile, kein Satz.
        # „Fünfte Stunde, …“ unter den Notizen bleibt der Beginn, wenn sonst kein Ortsdatum folgt.
        session_line = AM_SESSION_RE.match(line.strip()) is not None
        source_mark = re.match(r"^V\.\s", line.strip()) is not None
        stunde_line = STUNDE_HEAD_RE.match(line.strip()) is not None
        # „Donnerstag, 5. Juli 1923“ unter der Besprechung bleibt der Beginn.
        weekday_line = WEEKDAY_HEAD_RE.match(line.strip()) is not None
        # „Aus der Diskussion vom …“ und „8. August 1923“ unter „Zur Ausstellung“
        # stehen oft erst unter dem vorigen Text.
        appendix_line = _appendix_midpage(rest, index)
        if parsed and stunde_line and not any(hit[0] for hit in parsed):
            stunden.append(parsed)
            continue
        # „…», Arnheim, 17. Juli 1924, in“ in den Hinweisen ist eine Literaturangabe.
        # Ein Vortragsanfang mitten auf der Seite beginnt die Zeile.
        opens_line = _heading_opens_line(line, parsed)
        if parsed and (
            session_line
            or source_mark
            or weekday_line
            or appendix_line
            or (opens_line and any(hit[0] for hit in parsed))
        ):
            later.append(parsed)
    later = _drop_bare_vom(later)
    if len(later) == 1:
        return later[0]
    if not later and len(stunden) == 1:
        return stunden[0]
    quoted = _quoted_lecture(lines)
    if quoted:
        return quoted
    return []


def _bare_place_date(line: str) -> bool:
    """Die Zeile ist nur Ort und Datum, kein Satz und kein Registereintrag."""
    cleaned = line.strip().lstrip("'\"„“»").rstrip("'\"“”»")
    cleaned = re.sub(r"^\[([^\]]+)\]", r"\1", cleaned)
    cleaned = re.sub(r",\s*[A-ZÄÖÜ][A-Za-zÄÖÜäöüß.]{0,24}\s+\d+\s*$", "", cleaned)
    return HEADING_LINE_RE.match(cleaned) is not None or ODER_RE.match(cleaned) is not None


def _several_place_dates(lines: list[str]) -> list[tuple[str, int, int, int]] | None:
    """„Berlin, 21. Oktober 1906“ und „Berlin, 20. Oktober 1907“ auf einer Seite."""
    # Eine Literaturliste nennt viele Jahre. Das sind keine Vortragsanfänge.
    dated = re.compile(rf"(?:{MONTH_PATTERN})\s+\d{{4}}", re.IGNORECASE)
    if sum(1 for line in lines if dated.search(line)) > 6:
        return None
    found: list[tuple[str, int, int, int]] = []
    for index, line in enumerate(lines):
        if _letter_before_salutation(lines, index):
            continue
        if _same_hour_continued(lines, index):
            continue
        parsed = heading_lines(line)
        if not parsed:
            continue
        if not _bare_place_date(line) or not all(hit[0] for hit in parsed):
            return None
        found.extend(parsed)
    if len(found) < 2 or len(found) > 8:
        return None
    return found


def _drop_bare_vom(groups: list[list[tuple[str, int, int, int]]]) -> list[list[tuple[str, int, int, int]]]:
    """„vom 8. Februar 1906“ unter einem Ortsdatum ist der Zeitungsbericht."""
    placed = [group for group in groups if any(hit[0] for hit in group)]
    return placed or groups


def fold_place(text: str) -> str:
    folded = text.lower().replace("ß", "ss").replace("%", "z")
    folded = folded.replace("ä", "a").replace("ö", "o").replace("ü", "u")
    for src, dst in (
        ("â", "a"), ("à", "a"), ("á", "a"),
        ("ê", "e"), ("é", "e"), ("è", "e"), ("ë", "e"),
        ("î", "i"), ("ï", "i"),
        ("ô", "o"),
        ("û", "u"), ("ù", "u"),
        ("ç", "c"),
    ):
        folded = folded.replace(src, dst)
    # „Darnach“ ist Dornach: das o wurde als a gelesen.
    folded = folded.replace("darnach", "dornach")
    # „Dornacb“: das h am Ende wurde als b gelesen.
    folded = folded.replace("dornacb", "dornach")
    # „Kristiana“ ist Kristiania: ein i fehlt.
    folded = re.sub(r"kristiana(?!i)", "kristiania", folded)
    # „Arnheim“ ist Arnhem: der Band schreibt die deutsche, der Katalog die niederländische Form.
    folded = folded.replace("arnheim", "arnhem")
    # „Goetheanum, den 11. März 1924“ ist Dornach.
    folded = folded.replace("goetheanum", "dornach")
    # „rn“ nur, wo die Buchstaben wirklich nebeneinander stehen.
    # „Penmaenmawr (N.Wales)“ bleibt Penmaenmawr; die Klammer klebt kein rn an.
    folded = folded.replace("rn", "m")
    # „Penmaenmawr (England)“ und „Penmaenmawr (N.Wales)“ sind derselbe Ort.
    folded = re.sub(r"\([^)]*\)", " ", folded)
    # „Freiburg i. Br.“ und „Freiburg im Breisgau“ sind derselbe Ort.
    folded = re.sub(r"freiburg\s+im\s+breisgau", "freiburg", folded)
    folded = re.sub(r"freiburg\s+i\.?\s*br\.?", "freiburg", folded)
    folded = re.sub(r"[^a-z]", "", folded)
    # „ch“ am Wortende wird oft „di“ oder „dh“: „Züridi“ ist Zürich, „Dornadh“ ist Dornach.
    return re.sub(r"d[ih]$", "ch", folded)


def places_compatible(lecture_ort: str, heading_place: str) -> bool:
    if not lecture_ort or not heading_place:
        return True
    left = fold_place(lecture_ort)
    right = fold_place(heading_place)
    if not left or not right:
        return True
    if left in right or right in left:
        return True
    # „IVien“ ist Wien: das W wurde als IV gelesen.
    def as_w(value: str) -> str:
        if value.startswith(("iv", "vv")):
            return "w" + value[2:]
        return value
    if len(left) > 3 and len(right) > 3 and as_w(left) == as_w(right):
        return True
    # „Faris“ ist Paris: der erste Buchstabe P wurde als F gelesen.
    if len(left) > 3 and left[1:] == right[1:] and {left[0], right[0]} == {"f", "p"}:
        return True
    # „Neuchaiel“ ist Neuchâtel, „Heisingfors“ ist Helsingfors:
    # ein t oder ein l wurde als i gelesen.
    if len(left) == len(right) and len(left) > 5:
        diffs = [pair for pair in zip(left, right) if pair[0] != pair[1]]
        if len(diffs) == 1 and set(diffs[0]) in ({"t", "i"}, {"l", "i"}):
            return True
    return False


def sibling_days(lecture: Lecture, lectures: list[Lecture]) -> set[int]:
    """Andere Katalogtage am selben Ort im selben Monat. Die sind kein OCR-Fehler."""
    target = parse_datum(lecture.datum)
    if target is None:
        return set()
    _day, month, year = target
    found: set[int] = set()
    for other in lectures:
        if other.id == lecture.id:
            continue
        other_date = parse_datum(other.datum)
        if other_date is None:
            continue
        other_day, other_month, other_year = other_date
        if other_month == month and other_year == year and places_compatible(lecture.ort, other.ort):
            found.add(other_day)
    return found


def hit_matches(
    hit: tuple[str, int, int, int],
    lecture: Lecture,
    target: tuple[int, int, int],
    *,
    blocked_days: set[int] | None = None,
) -> bool:
    day, month, year = target
    place, hit_day, hit_month, hit_year = hit
    if hit_month != month:
        return False
    # Jahr 0: die Überschrift nennt kein Jahr. Tag, Monat und Ort müssen stimmen.
    if hit_year == 0:
        if hit_day != day:
            return False
        if place:
            return places_compatible(lecture.ort, place)
        # „am 24. Dezember, 10 Uhr vormittags“: kein Ort und kein Jahr.
        return True
    if hit_day == day and hit_year == year:
        return places_compatible(lecture.ort, place)
    if not lecture.ort or not place:
        return False
    if not places_compatible(lecture.ort, place):
        return False
    if hit_day == day:
        return years_compatible(year, hit_year)
    if hit_year == year and days_compatible(day, hit_day):
        if blocked_days and hit_day in blocked_days:
            return False
        return True
    return False


def title_tokens(text: str) -> set[str]:
    words = re.findall(r"[a-zäöüß]{5,}", text.lower())
    return {word for word in words if word not in STOPWORDS}


def title_overlap(titel: str, page: str) -> float:
    wanted = title_tokens(titel)
    if len(wanted) < 3:
        return 0.0
    head = squash("\n".join(page.splitlines()[:20]))
    found = title_tokens(head)
    return len(wanted & found) / len(wanted)


def _hour_role(
    lecture: Lecture,
    page: str,
    blocked_days: set[int] | None,
) -> str:
    """Unterscheidet die Gedächtnisnotizen, einen neuen Abschnitt und einen späteren Auszug."""
    target = parse_datum(lecture.datum)
    if target is None:
        return ""
    previous = ""
    role = ""
    for raw in page.splitlines():
        line = raw.strip()
        if not line:
            continue
        parsed = heading_lines(line)
        if parsed and any(hit_matches(hit, lecture, target, blocked_days=blocked_days) for hit in parsed):
            if re.search(r"\bvon\s+der\s+Instruktionsstunde\b", line, re.IGNORECASE):
                return "notes"
            if re.fullmatch(r"[IVX]{1,4}", previous):
                return "roman"
            if re.search(r"\bAus\s+Instruktionsstunde\b", line, re.IGNORECASE):
                role = role or "excerpt"
            else:
                return "other"
        previous = line
    return role


def _program_sheet(page: str) -> bool:
    """Die Datumszeile leitet das Programm ein, nicht die Ansprache."""
    lines = [line.strip() for line in page.splitlines() if line.strip()]
    for index, line in enumerate(lines[:25]):
        if re.match(r"Programm\s+zur\s+", line, re.IGNORECASE) and heading_lines(line):
            return True
        if not heading_lines(line):
            continue
        if any(re.fullmatch(r"PROGRAMM", item, re.IGNORECASE) for item in lines[index + 1:index + 4]):
            return True
    return False


def _daytime_page(
    lecture: Lecture,
    lectures: list[Lecture],
    pages: list[int],
    volume: VolumeText,
) -> list[int]:
    """Vormittags ist der frühere, nachmittags der spätere Eintrag desselben Tages.

    Mehrere Nachmittagsanfänge gehören zum späteren Katalogeintrag; genommen wird der erste.
    """
    if len(pages) < 2:
        return pages

    def daytime(page_no: int) -> str:
        head = "\n".join(volume.pages[page_no - 1].splitlines()[:30])
        later = bool(re.search(r"\b(?:nachmittags|abends)\b", head, re.IGNORECASE))
        earlier = bool(re.search(r"\bvormittags\b", head, re.IGNORECASE))
        # „vormittags und nachmittags“ ist eine Zusammenkunft, kein Paar.
        if later and earlier:
            return ""
        if later:
            return "nachmittags"
        if earlier:
            return "vormittags"
        return ""

    roles = {page_no: daytime(page_no) for page_no in pages}
    if set(roles.values()) != {"vormittags", "nachmittags"}:
        return pages
    twins = [
        other
        for other in lectures
        if other.datum == lecture.datum and places_compatible(lecture.ort, other.ort)
    ]
    twins.sort(key=lambda other: other.id)
    if len(twins) == 2:
        want = "vormittags" if lecture.id == twins[0].id else "nachmittags"
    elif len(twins) == 1:
        want = "vormittags"
    else:
        return pages
    chosen = [page_no for page_no, role in roles.items() if role == want]
    # „24. Juli, nachmittags“ steht über dem neunten Vortrag und noch einmal über dem Schlusswort.
    # Der Katalog führt beides als einen Eintrag. Der Beginn ist die frühere Seite.
    if want == "nachmittags" and len(chosen) > 1 and len(twins) <= 2:
        return [min(chosen)]
    return chosen or pages


def fill_unique_headings(
    lectures: list[Lecture],
    volume: VolumeText,
    assignments: dict[str, Assignment],
) -> list[tuple[Lecture, int]]:
    """Setzt eine Seite, wenn genau ein Vortragsanfang zum Katalogeintrag passt."""
    filled: list[tuple[Lecture, int]] = []
    for lecture in lectures:
        current = assignments.get(lecture.id)
        if current is None or current.seite_pdf is not None:
            continue
        occupied = sibling_days(lecture, lectures)
        pages = [
            index + 1
            for index, page in enumerate(volume.pages)
            if index not in volume.toc_indexes
            and page_matches(
                lecture,
                page,
                allow_title=False,
                blocked_days=occupied,
                ga=volume.ga,
                prior_page=volume.pages[index - 1] if index else None,
            )[0]
        ]
        if len(pages) > 1:
            placed = [
                page_no
                for page_no in pages
                if any(
                    hit[0] and places_compatible(lecture.ort, hit[0])
                    for hit in heading_hits(volume.pages[page_no - 1])
                )
            ]
            if len(placed) == 1:
                pages = placed
        if len(pages) > 1:
            # Die Gedächtnisnotizen und ein Abschnitt nach „II“ sind der Beginn.
            # „Aus Instruktionsstunde …“ weiter hinten ist ein Auszug derselben Stunde.
            roles = {
                page_no: _hour_role(lecture, volume.pages[page_no - 1], occupied)
                for page_no in pages
            }
            notes = [page_no for page_no, role in roles.items() if role == "notes"]
            roman = [page_no for page_no, role in roles.items() if role == "roman"]
            if len(notes) == 1 and all(role in {"notes", "excerpt"} for role in roles.values()):
                pages = notes
            elif len(roman) == 1 and all(role in {"roman", "excerpt"} for role in roles.values()):
                pages = roman
        if len(pages) > 1:
            speeches = [page_no for page_no in pages if not _program_sheet(volume.pages[page_no - 1])]
            if len(speeches) == 1:
                pages = speeches
            addressed = [
                page_no
                for page_no in pages
                if re.search(r"(?m)^\s*Ansprache\s+zur\s+Eurythmie\b", volume.pages[page_no - 1], re.IGNORECASE)
            ]
            if len(addressed) == 1:
                pages = addressed
        if len(pages) > 1 and lecture.titel.strip():
            titled = [
                page_no
                for page_no in pages
                if title_overlap(lecture.titel, volume.pages[page_no - 1]) >= 0.55
            ]
            if len(titled) == 1:
                pages = titled
        if len(pages) > 1:
            pages = _daytime_page(lecture, lectures, pages, volume)
        if len(pages) != 1:
            continue
        current.seite_pdf = pages[0]
        current.markierung = None
        current.grund = None
        current.rejected_page = None
        current.reject_reason = None
        filled.append((lecture, pages[0]))
    return filled


def parse_season(datum: str) -> tuple[str, int] | None:
    text = " ".join(datum.strip().lower().split())
    match = re.fullmatch(r"(\d{4})\s+(frühling|fruehling|sommer|herbst|winter)", text)
    if match:
        return SEASON_WORD[match.group(2)], int(match.group(1))
    match = re.fullmatch(r"(frühling|fruehling|sommer|herbst|winter)\s+(\d{4})", text)
    if match:
        return SEASON_WORD[match.group(1)], int(match.group(2))
    return None


def season_hit(line: str) -> tuple[str, str, int] | None:
    match = SEASON_LINE_RE.match(line.strip())
    if not match:
        return None
    place = (match.group(1) or "").strip(" ,\\;")
    year = ocr_year(match.group(3))
    if not _place_ok(place) or year is None:
        return None
    return place, SEASON_WORD[match.group(2).lower()], year


def season_title_ok(titel: str, page: str) -> bool:
    """Unterscheidet mehrere Stunden desselben Sommers am Titelanfang."""
    words = [word for word in re.findall(r"[a-zäöüß]{5,}", titel.lower()) if word not in STOPWORDS]
    if not words:
        return False
    opening = words[:2]
    head = squash("\n".join(page.splitlines()[:30])).lower()
    found = [word for word in opening if re.search(rf"\b{re.escape(word)}\b", head)]
    if len(opening) == 1:
        return len(found) == 1 and len(opening[0]) >= 8
    return len(found) == len(opening)


def _repeats_previous_source(
    lecture: Lecture,
    page: str,
    prior_page: str | None,
    target: tuple[int, int, int],
    blocked_days: set[int] | None,
) -> bool:
    """„V. Pforzheim, 7. März 1914“ auf der nächsten Seite ist dieselbe Quelle, kein neuer Beginn."""
    if not prior_page:
        return False

    def marked(text: str) -> bool:
        for raw in text.splitlines():
            line = raw.strip()
            if not re.match(r"^V\.\s", line):
                continue
            parsed = heading_lines(line)
            if parsed and any(hit_matches(hit, lecture, target, blocked_days=blocked_days) for hit in parsed):
                return True
        return False

    return marked(page) and marked(prior_page)


def _dateline_continued(
    page: str,
    prior_page: str | None,
    lecture: Lecture,
    target: tuple[int, int, int],
    blocked_days: set[int] | None,
) -> bool:
    """Dieselbe Orts- und Datumszeile weiter unten ist der laufende Text, kein neuer Beginn."""
    if not prior_page:
        return False
    if not any(
        hit_matches(hit, lecture, target, blocked_days=blocked_days)
        for hit in heading_hits(prior_page)
    ):
        return False
    opening = [raw.strip() for raw in page.splitlines() if raw.strip()][:8]
    for raw in opening:
        for hit in heading_lines(raw):
            if hit_matches(hit, lecture, target, blocked_days=blocked_days):
                return False
    return True


def page_matches(
    lecture: Lecture,
    page: str,
    *,
    allow_title: bool,
    blocked_days: set[int] | None = None,
    ga: str = "",
    prior_page: str | None = None,
) -> tuple[bool, str]:
    season = parse_season(lecture.datum)
    if season:
        for raw in page.splitlines():
            line = raw.strip()
            if not line:
                continue
            candidates = [line]
            for mark in re.finditer(r",\s+", line):
                candidates.append(line[mark.end() :])
            hit = None
            for candidate in candidates:
                hit = season_hit(candidate)
                if hit is not None and hit[0]:
                    break
            if hit is None:
                continue
            place, name, year = hit
            if name == season[0] and year == season[1] and places_compatible(lecture.ort, place):
                if season_title_ok(lecture.titel, page):
                    return True, "jahreszeit"
        return False, "Jahreszeit oder Titel nicht am Seitenanfang"
    span = parse_year_span(lecture.datum)
    if span:
        for raw in page.splitlines():
            match = YEAR_SPAN_RE.match(raw.strip().lstrip("* "))
            if not match:
                continue
            years = {int(match.group(2)), int(match.group(3))}
            if years == set(span) and places_compatible(lecture.ort, match.group(1) or ""):
                return True, "jahresspanne"
        return False, "Jahresspanne nicht am Seitenanfang"
    # GA 164 druckt „Z. September 1915“ für den Vortrag 19151003b vom 3. Oktober 1915.
    # Im Seitentext steht danach „3. September“, weil Z als Tag 3 gelesen wird.
    if (
        ga == "164"
        and lecture.id == "19151003b"
        and re.search(r"D[oö]rnach,\s*(?:Z|3)\.\s*September\s+1915", page, re.IGNORECASE)
    ):
        return True, "ausnahme"
    # GA 167 druckt „Berlin, 13. April 1916“ für den Vortrag 19160411 vom 11. April 1916.
    if (
        ga == "167"
        and lecture.id == "19160411"
        and re.search(r"Berlin,\s*13\.\s*April\s+1916", page, re.IGNORECASE)
    ):
        return True, "ausnahme"
    # GA 272 druckt „Dornach, u. April 1915“ für den Vortrag 19110411 vom 11. April 1911.
    # Im Seitentext steht danach „11. April“, weil u als Tag 11 gelesen wird.
    if (
        ga == "272"
        and lecture.id == "19110411"
        and re.search(
            r"D[oö]rnach,\s*(?:11|[un])\.\s*April\s+191(?:5|\$)",
            page,
            re.IGNORECASE,
        )
    ):
        return True, "ausnahme"
    # GA 332b druckt „Domach, 1. April 1922“ für 19220402c vom 2. April 1922.
    # Das Inhaltsverzeichnis des Bandes nennt den 2. April.
    if (
        ga == "332b"
        and lecture.id == "19220402c"
        and re.search(r"(?m)^\s*Domach,\s*1\.\s*April\s+1922\s*$", page, re.IGNORECASE)
    ):
        return True, "ausnahme"
    # GA 332b druckt „Domach, 31. Dezember 1923, vormittags“ für 19240101f vom 1. Januar 1924.
    if (
        ga == "332b"
        and lecture.id == "19240101f"
        and re.search(r"(?m)^\s*Domach,\s*31\.\s*Dezember\s+1923\b", page, re.IGNORECASE)
    ):
        return True, "ausnahme"
    # GA 304a druckt „Ilkley, 10. Augast 1923“ für den Vortrag 19230810b, Katalogort Bingley.
    if ga == "304a" and lecture.id == "19230810b":
        ilkley = [
            hit
            for hit in heading_hits(page)
            if hit[1] == 10 and hit[2] == 8 and hit[3] == 1923 and fold_place(hit[0]) == "ilkley"
        ]
        if ilkley:
            return True, "ausnahme"
    hits = heading_hits(page)
    target = parse_datum(lecture.datum)
    if target and hits:
        same = [hit for hit in hits if hit_matches(hit, lecture, target, blocked_days=blocked_days)]
        if same:
            if _repeats_previous_source(lecture, page, prior_page, target, blocked_days):
                return False, "dieselbe Quellenzeile wie auf der Seite davor"
            # „Goetheanum, den 11. März 1924“ steht auf der nächsten Seite noch einmal
            # über der Fragenbeantwortung desselben Rundbriefs.
            if _dateline_continued(page, prior_page, lecture, target, blocked_days):
                return False, "dieselbe Datumszeile wie auf der Seite davor"
            return True, "datum"
        if any(hit[1] != target[0] or hit[2] != target[1] for hit in hits):
            return False, "anderes datum am Seitenanfang"
        return False, "ort oder jahr widerspricht"
    if hits:
        return False, "Datum am Seitenanfang passt nicht"
    if allow_title and title_overlap(lecture.titel, page) >= 0.55:
        return True, "titel"
    return False, "Datum oder Titel nicht am Seitenanfang"


def format_toc(pages: list[str], indexes: set[int]) -> str:
    if not indexes:
        return "(kein Inhaltsverzeichnis in den ersten 25 PDF-Seiten erkannt)"
    chunks: list[str] = []
    for index in sorted(indexes):
        body = pages[index].strip()
        chunks.append(f"--- PDF-Seite {index + 1} (Inhaltsverzeichnis) ---\n{body}")
    text = "\n\n".join(chunks)
    if len(text) > 24000:
        text = text[:24000] + "\n[Inhaltsverzeichnis gekürzt]"
    return text


def format_anchors(pages: list[str], toc_indexes: set[int]) -> list[str]:
    anchors: list[str] = []
    for index, page in enumerate(pages):
        if index in toc_indexes:
            continue
        hits = heading_hits(page)
        if not hits:
            continue
        lines = [line.strip() for line in page.splitlines() if line.strip()][:6]
        preview = " / ".join(lines)
        if len(preview) > 320:
            preview = preview[:320] + "…"
        anchors.append(f"PDF {index + 1} | {preview}")
    return anchors


def catalog_block(lectures: list[Lecture]) -> str:
    lines = []
    for lecture in lectures:
        reihe = f" | Reihe: {lecture.reihe}" if lecture.reihe else ""
        lines.append(
            f"- id: {lecture.id} | {lecture.datum} | {lecture.ort} | {lecture.titel}{reihe}"
        )
    return "\n".join(lines)


def build_volume(pdf: Path) -> VolumeText:
    pages = extract_pages(pdf)
    toc_indexes = find_toc_indexes(pages)
    token = pdf_token(pdf)
    return VolumeText(
        pages=pages,
        ga=token[2] if token else "",
        toc_indexes=toc_indexes,
        toc_block=format_toc(pages, toc_indexes),
        anchors=format_anchors(pages, toc_indexes),
    )


def pass1_user(pdf: Path, lectures: list[Lecture], volume: VolumeText) -> str:
    token = pdf_token(pdf)
    ga = token[2] if token else pdf.stem
    anchors = "\n".join(volume.anchors) if volume.anchors else "(keine Vortragsanfänge erkannt)"
    return (
        f"GA {ga}, Datei {pdf.name}, {len(volume.pages)} PDF-Seiten.\n"
        f"Katalogvorträge in diesem Band: {len(lectures)}. Jede id muss genau einmal vorkommen.\n\n"
        f"Katalog:\n{catalog_block(lectures)}\n\n"
        f"Inhaltsverzeichnis:\n{volume.toc_block}\n\n"
        f"Vortragsanfänge (PDF-Seite und erste Zeilen):\n{anchors}\n"
    )


def search_hits(lecture: Lecture, volume: VolumeText, limit: int = 5) -> list[str]:
    pattern = datum_regex(lecture.datum)
    ranked: list[tuple[int, str]] = []
    for index, page in enumerate(volume.pages):
        squashed = squash(page)
        date_hit = bool(pattern and pattern.search(squashed))
        overlap = title_overlap(lecture.titel, page)
        if not date_hit and overlap < 0.45:
            continue
        if index in volume.toc_indexes:
            kind = "inhaltsverzeichnis"
            rank = 2
        elif heading_hits(page):
            kind = "vortragsbeginn"
            rank = 0
        else:
            kind = "erwaehnung"
            rank = 1
        lines = [line.strip() for line in page.splitlines() if line.strip()][:8]
        preview = " / ".join(lines)
        if len(preview) > 420:
            preview = preview[:420] + "…"
        ranked.append((rank, f"PDF {index + 1} | {kind} | {preview}"))
    ranked.sort(key=lambda item: item[0])
    return [line for _rank, line in ranked[:limit]]


def pass2_user(
    lectures: list[Lecture],
    pending: list[Lecture],
    assignments: dict[str, Assignment],
    volume: VolumeText,
) -> str:
    by_id = {lecture.id: lecture for lecture in lectures}
    blocks: list[str] = []
    for lecture in pending:
        current = assignments.get(lecture.id)
        reason = ""
        if current and current.reject_reason:
            reason = f"\nErster Lauf verworfen: {current.reject_reason}"
            if current.rejected_page:
                reason += f" (Seite {current.rejected_page})"
        hits = search_hits(lecture, volume)
        hit_text = "\n".join(hits) if hits else "(keine Suchtreffer im PDF)"
        row = by_id[lecture.id]
        blocks.append(
            f"id: {row.id} | {row.datum} | {row.ort} | {row.titel}{reason}\n"
            f"Suchtreffer:\n{hit_text}"
        )
    return (
        f"Restliste, {len(pending)} Vorträge. Nur diese ids zurückgeben.\n\n"
        + "\n\n".join(blocks)
    )


def salvage_json_object(text: str) -> dict | None:
    """Liest vollständige Vortragsobjekte aus einer abgeschnittenen Antwort."""
    rows: list[dict] = []
    for match in re.finditer(r"\{[^{}]*\}", text):
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("id") and ("seite_pdf" in obj or "seite" in obj):
            rows.append(obj)
    if not rows:
        return None
    return {"vortraege": rows, "hinweis": ""}


def deepseek_json(
    client: httpx.Client,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
) -> tuple[dict, dict]:
    payload: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "thinking": {"type": "disabled"},
    }
    last_error = ""
    repaired = False
    for _attempt in range(5):
        response = client.post("/chat/completions", json=payload)
        if response.status_code == 400 and "thinking" in payload:
            payload.pop("thinking", None)
            continue
        if response.status_code == 400 and "response_format" in payload:
            payload.pop("response_format", None)
            continue
        if response.status_code in {429, 500, 502, 503}:
            last_error = response.text[:400]
            time.sleep(2 ** attempt)
            continue
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("DeepSeek lieferte keine choices")
        message = choices[0].get("message") or {}
        usage = data.get("usage") or {}
        candidates = [
            message.get("content"),
            message.get("reasoning_content"),
        ]
        parsed: dict | None = None
        broken = ""
        for candidate in candidates:
            if not isinstance(candidate, str) or not candidate.strip():
                continue
            try:
                parsed = parse_json_object(candidate)
                break
            except (json.JSONDecodeError, ValueError) as exc:
                broken = str(exc)
                salvaged = salvage_json_object(candidate)
                if salvaged and choices[0].get("finish_reason") != "length":
                    parsed = salvaged
                    break
        if parsed is not None:
            return parsed, usage
        if not repaired:
            repaired = True
            print("  Antwort kein gültiges JSON, neuer Versuch", file=sys.stderr)
            payload["messages"] = [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": (
                        user
                        + "\n\nDie vorige Antwort war kein gültiges JSON. "
                        + "Gib nur das Schema zurück: zuerst vortraege, danach hinweis "
                        + "in höchstens zwei Sätzen."
                    ),
                },
            ]
            continue
        salvaged = None
        for candidate in candidates:
            if isinstance(candidate, str):
                salvaged = salvage_json_object(candidate)
                if salvaged:
                    break
        if salvaged:
            return salvaged, usage
        raise RuntimeError(f"Antwort kein gültiges JSON: {broken}")
    raise RuntimeError(f"DeepSeek nicht erreichbar: {last_error}")


def parse_json_object(text: str) -> dict:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    if start < 0:
        raise ValueError("Antwort enthält kein JSON-Objekt")
    obj, _end = json.JSONDecoder().raw_decode(cleaned[start:])
    if not isinstance(obj, dict):
        raise ValueError("JSON-Antwort ist kein Objekt")
    return obj


def parse_page(value: object, page_count: int) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "null", "none"}:
            return None
        value = text
    try:
        number = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if number < 1 or number > page_count:
        return None
    return number


def read_assignments(payload: dict, page_count: int) -> dict[str, tuple[int | None, str | None]]:
    rows = payload.get("vortraege") or payload.get("lectures") or []
    found: dict[str, tuple[int | None, str | None]] = {}
    if not isinstance(rows, list):
        return found
    for row in rows:
        if not isinstance(row, dict):
            continue
        lecture_id = str(row.get("id") or "").strip()
        if not lecture_id:
            continue
        page = parse_page(row.get("seite_pdf", row.get("seite")), page_count)
        grund = row.get("grund")
        found[lecture_id] = (page, str(grund).strip() if isinstance(grund, str) else None)
    return found


def apply_pass(
    lectures: list[Lecture],
    pending_ids: set[str],
    raw: dict[str, tuple[int | None, str | None]],
    volume: VolumeText,
    assignments: dict[str, Assignment],
    *,
    second: bool,
) -> None:
    by_id = {lecture.id: lecture for lecture in lectures}
    for lecture_id in pending_ids:
        lecture = by_id[lecture_id]
        page, grund = raw.get(lecture_id, (None, None))
        current = assignments.setdefault(lecture_id, Assignment())
        if page is None:
            current.seite_pdf = None
            current.grund = grund or "Im Band nicht gefunden."
            if second:
                current.markierung = "nicht_im_band"
            continue
        if page - 1 in volume.toc_indexes:
            current.rejected_page = page
            current.reject_reason = "Seite gehört zum Inhaltsverzeichnis"
            current.seite_pdf = None
            current.grund = current.reject_reason
            if second:
                current.markierung = "nicht_im_band"
            continue
        ok, why = page_matches(
            lecture,
            volume.pages[page - 1],
            allow_title=second,
            blocked_days=sibling_days(lecture, lectures),
            ga=volume.ga,
            prior_page=volume.pages[page - 2] if page > 1 else None,
        )
        if ok:
            current.seite_pdf = page
            current.grund = None
            current.markierung = "zweiter_lauf" if second else None
            current.rejected_page = None
            current.reject_reason = None
            continue
        current.rejected_page = page
        current.reject_reason = why
        current.seite_pdf = None
        current.grund = grund or why
        if second:
            current.markierung = "nicht_im_band"


def printed_pages_from_toc(volume: VolumeText) -> dict[tuple[int, int, int], int]:
    found: dict[tuple[int, int, int], int] = {}
    for index in sorted(volume.toc_indexes):
        for line in volume.pages[index].splitlines():
            stripped = line.strip()
            date_match = DATE_ANY_RE.search(stripped)
            page_match = re.search(r"(\d{1,4})\s*$", stripped)
            if not date_match or not page_match or page_match.start() < date_match.end():
                continue
            month = month_index(date_match.group(2))
            day = ocr_day(date_match.group(1))
            if month is None or day is None:
                continue
            year = int(date_match.group(3))
            printed = int(page_match.group(1))
            if printed == year or printed > 2000:
                continue
            # Mehrere Zeilen desselben Tages: der Beginn ist die kleinste Seitenzahl.
            # Das Schlusswort danach darf die Zahl des Vormittags nicht ersetzen.
            key = (day, month, year)
            previous = found.get(key)
            if previous is None or printed < previous:
                found[key] = printed
    return found


def offset_sentence(
    lectures: list[Lecture],
    assignments: dict[str, Assignment],
    volume: VolumeText,
) -> str:
    printed = printed_pages_from_toc(volume)
    earliest: dict[tuple[int, int, int], int] = {}
    for lecture in lectures:
        # Anhangsstücke (Diskussion, Ausstellung, Autoreferat) stehen nicht
        # auf der Seitenzahl des Hauptvortrags im Inhaltsverzeichnis.
        if _appendix_role(lecture):
            continue
        assignment = assignments.get(lecture.id)
        parsed = parse_datum(lecture.datum)
        if not assignment or assignment.seite_pdf is None or not parsed:
            continue
        if parsed not in printed:
            continue
        previous = earliest.get(parsed)
        if previous is None or assignment.seite_pdf < previous:
            earliest[parsed] = assignment.seite_pdf
    deltas = [page - printed[parsed] for parsed, page in earliest.items()]
    if not deltas:
        return ""
    if all(delta == 0 for delta in deltas):
        return "PDF-Seite entspricht der gedruckten Seitenzahl."
    if len(set(deltas)) == 1:
        delta = deltas[0]
        signed = f"+{delta}" if delta > 0 else str(delta)
        return f"PDF-Seite = gedruckte Seitenzahl {signed}."
    return "PDF-Seite und gedruckte Seitenzahl gehen auseinander; genommen wurde der Vortragsanfang im PDF."


def lecture_already_named(lecture: Lecture, text: str) -> bool:
    if lecture.datum and lecture.datum in text:
        return True
    parsed = parse_datum(lecture.datum)
    if not parsed:
        return False
    day, month, year = parsed
    return f"{day}. {MONTHS[month - 1]} {year}" in text


def build_hinweis(
    model_hinweis: str,
    lectures: list[Lecture],
    assignments: dict[str, Assignment],
    volume: VolumeText,
) -> str:
    parts: list[str] = []
    offset = offset_sentence(lectures, assignments, volume)
    if offset:
        parts.append(offset)
    elif model_hinweis:
        parts.append(re.sub(r"\s+", " ", model_hinweis).strip())
    missing = [
        lecture
        for lecture in lectures
        if assignments.get(lecture.id) and assignments[lecture.id].seite_pdf is None
    ]
    second = [
        lecture
        for lecture in lectures
        if assignments.get(lecture.id)
        and assignments[lecture.id].markierung == "zweiter_lauf"
    ]
    if second:
        dates = ", ".join(lecture.datum for lecture in second)
        parts.append(
            f"Im zweiten Lauf gefunden und mit markierung zweiter_lauf gekennzeichnet: {dates}."
        )
    covered = " ".join(parts)
    for lecture in missing:
        if lecture_already_named(lecture, covered):
            continue
        label = lecture.titel
        if lecture.ort and lecture.ort.lower() not in label.lower():
            label = f"{label}, {lecture.ort}" if label else lecture.ort
        parts.append(f"Der Vortrag vom {lecture.datum} ({label}) ist im Band nicht enthalten.")
        covered = " ".join(parts)
    shared: dict[int, list[Lecture]] = {}
    for lecture in lectures:
        assignment = assignments.get(lecture.id)
        if assignment and assignment.seite_pdf is not None:
            shared.setdefault(assignment.seite_pdf, []).append(lecture)
    for page, group in shared.items():
        if len(group) > 1:
            dates = ", ".join(lecture.datum for lecture in group)
            parts.append(f"Die Vorträge vom {dates} beginnen gemeinsam auf PDF-Seite {page}.")
    text = " ".join(part for part in parts if part)
    return text or "Seiten aus den Vortragsanfängen des PDF."


def yaml_quote(value: str) -> str:
    if re.fullmatch(r"\d+[a-z]?", value):
        return value
    if re.search(r"[:#\n\"'\\]|^\s|\s$|^(?:true|false|null|yes|no|~)$", value, re.I):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if value.startswith(("-", "?", "&", "*", "!", "|", ">", "%", "@", "`", "{", "[")):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return value


def render_yaml(
    pdf: Path,
    lectures: list[Lecture],
    assignments: dict[str, Assignment],
    hinweis: str,
    stats: str,
) -> str:
    token = pdf_token(pdf)
    ga_value = token[2] if token else pdf.stem
    lines = [
        f"# Abgleich: {stats}",
        f"ga: {ga_value}",
        f"datei: {pdf.name.replace(' ', '_')}",
        f"hinweis: {yaml_quote(hinweis)}",
        "vortraege:",
    ]
    for lecture in lectures:
        assignment = assignments.get(lecture.id) or Assignment(
            grund="nicht geprüft", markierung="nicht_im_band"
        )
        lines.append(f"- datum: {yaml_quote(lecture.datum) if lecture.datum else 'null'}")
        lines.append(f"  id: {yaml_quote(lecture.id)}")
        lines.append(f"  titel: {yaml_quote(lecture.titel) if lecture.titel else '\"\"'}")
        if lecture.reihe:
            lines.append(f"  reihe: {yaml_quote(lecture.reihe)}")
        if assignment.seite_pdf is None:
            lines.append("  seite_pdf: null")
        else:
            lines.append(f"  seite_pdf: {assignment.seite_pdf}")
        if assignment.markierung:
            lines.append(f"  markierung: {assignment.markierung}")
    lines.append("")
    return "\n".join(lines)


def output_path(pdf: Path) -> Path:
    return pdf.with_name(f"{pdf.stem.lower()}-seitenverzeichnis.yaml")


def usage_note(usages: list[dict]) -> str:
    prompt = sum(int(item.get("prompt_tokens") or 0) for item in usages)
    completion = sum(int(item.get("completion_tokens") or 0) for item in usages)
    return f"{prompt}+{completion} Tokens"


def _appendix_role(lecture: Lecture) -> str | None:
    """Fragenbeantwortung, Ausstellungsworte und Autoreferat stehen oft im Anhang."""
    title = lecture.titel.lower()
    if "fragenbeantwortung" in title or re.search(r"\bdiskussion\b", title):
        return "diskussion"
    if "ausstellung" in title:
        return "ausstellung"
    if not lecture.titel.strip():
        return "autoreferat"
    return None


def appendix_openings(volume: VolumeText) -> dict[tuple[str, int, int, int], list[int]]:
    """Anhangseiten: Diskussion, Ausstellung, Autoreferat. Inhaltsverzeichnis zählt nicht."""
    found: dict[tuple[str, int, int, int], list[int]] = {}
    bare_date = re.compile(
        rf"^{DAY_TOKEN}{DAY_SEP}{MONTH_NOISE}{MONTH_PATTERN}{MONTH_YEAR_GAP}{YEAR_TAIL}$",
        re.IGNORECASE,
    )
    for index, page in enumerate(volume.pages):
        if index in volume.toc_indexes:
            continue
        lines = [line.strip() for line in page.splitlines() if line.strip()]
        for line_index, line in enumerate(lines):
            if re.search(r"\.{4,}", line):
                continue
            role = ""
            parsed: list[tuple[str, int, int, int]] = []
            if DISKUSSION_VOM_RE.match(line):
                role = "diskussion"
                parsed = heading_lines(line)
            elif re.search(r"\(\s*Autoreferat\s*\)", line, re.IGNORECASE):
                role = "autoreferat"
                parsed = heading_lines(line)
            elif re.match(r"^Zur Ausstellung\b", line, re.IGNORECASE):
                for nxt in lines[line_index + 1 : line_index + 4]:
                    if re.search(r"\.{4,}", nxt) or not bare_date.match(nxt):
                        continue
                    role = "ausstellung"
                    parsed = heading_lines(nxt)
                    break
            if not role or not parsed or parsed[0][3] == 0:
                continue
            key = (role, parsed[0][1], parsed[0][2], parsed[0][3])
            bucket = found.setdefault(key, [])
            page_no = index + 1
            if page_no not in bucket:
                bucket.append(page_no)
    return found


def _shares_lecture_opening(
    lecture: Lecture,
    lectures: list[Lecture],
    assignments: dict[str, Assignment],
) -> bool:
    current = assignments.get(lecture.id)
    if current is None or current.seite_pdf is None:
        return False
    for other in lectures:
        if other.id == lecture.id or other.datum != lecture.datum:
            continue
        if _appendix_role(other) is not None:
            continue
        other_assignment = assignments.get(other.id)
        if other_assignment and other_assignment.seite_pdf == current.seite_pdf:
            return True
    return False


def reassign_appendix_openings(
    lectures: list[Lecture],
    volume: VolumeText,
    assignments: dict[str, Assignment],
) -> list[tuple[Lecture, int | None]]:
    """Legt Anhangsstücke auf ihre Überschrift, nicht auf die erste Seite des Hauptvortrags."""
    openings = appendix_openings(volume)
    has_discussion = any(key[0] == "diskussion" for key in openings)
    changes: list[tuple[Lecture, int | None]] = []
    for lecture in lectures:
        role = _appendix_role(lecture)
        when = parse_datum(lecture.datum)
        if role is None or when is None:
            continue
        pages = openings.get((role, *when), [])
        others = [
            other
            for other in lectures
            if other.id != lecture.id
            and _appendix_role(other) == role
            and parse_datum(other.datum) == when
        ]
        current = assignments.setdefault(lecture.id, Assignment())
        if len(pages) == 1 and not others and current.seite_pdf != pages[0]:
            current.seite_pdf = pages[0]
            current.grund = None
            current.markierung = "zweiter_lauf"
            current.rejected_page = None
            current.reject_reason = None
            changes.append((lecture, pages[0]))
            continue
        if (
            role == "diskussion"
            and not pages
            and has_discussion
            and _shares_lecture_opening(lecture, lectures, assignments)
        ):
            current.seite_pdf = None
            current.grund = "Im Band nicht gefunden."
            current.markierung = "nicht_im_band"
            changes.append((lecture, None))
    return changes


def process_pdf(
    pdf: Path,
    lectures: list[Lecture],
    client: httpx.Client | None,
    model: str,
    dry_run: bool,
) -> int:
    volume = build_volume(pdf)
    print(
        f"{pdf.name}: {len(volume.pages)} Seiten, "
        f"Inhaltsverzeichnis {[i + 1 for i in sorted(volume.toc_indexes)] or '—'}, "
        f"{len(volume.anchors)} Vortragsanfänge, {len(lectures)} Katalogvorträge"
    )
    if dry_run:
        for anchor in volume.anchors:
            print(f"  {anchor}")
        return 0

    assert client is not None
    usages: list[dict] = []
    payload, usage = deepseek_json(
        client,
        model,
        PASS1_SYSTEM,
        pass1_user(pdf, lectures, volume),
        max_tokens=8192,
    )
    usages.append(usage)
    assignments: dict[str, Assignment] = {}
    apply_pass(
        lectures,
        {lecture.id for lecture in lectures},
        read_assignments(payload, len(volume.pages)),
        volume,
        assignments,
        second=False,
    )
    pending = [
        lecture
        for lecture in lectures
        if assignments.get(lecture.id) is None or assignments[lecture.id].seite_pdf is None
    ]
    first_found = len(lectures) - len(pending)
    second_found = 0
    if pending:
        print(
            f"  erster Lauf: {first_found}/{len(lectures)}; "
            f"zweiter Lauf für {len(pending)}: "
            + ", ".join(lecture.id for lecture in pending)
        )
        payload2, usage2 = deepseek_json(
            client,
            model,
            PASS2_SYSTEM,
            pass2_user(lectures, pending, assignments, volume),
            max_tokens=4096,
        )
        usages.append(usage2)
        before = {lecture.id: assignments[lecture.id].seite_pdf for lecture in pending}
        apply_pass(
            lectures,
            {lecture.id for lecture in pending},
            read_assignments(payload2, len(volume.pages)),
            volume,
            assignments,
            second=True,
        )
        second_found = sum(
            1
            for lecture in pending
            if assignments[lecture.id].seite_pdf is not None and before[lecture.id] is None
        )
    else:
        print(color(f"  erster Lauf vollständig: {len(lectures)}/{len(lectures)}", GREEN))

    for lecture, page in fill_unique_headings(lectures, volume, assignments):
        print(color(
            f"  Seitenanfang: {mismatch_line(lecture).removeprefix('  stimmt nicht: ')} → Seite {page}",
            GREEN,
        ))

    for lecture, page in reassign_appendix_openings(lectures, volume, assignments):
        if page is None:
            continue
        print(color(
            f"  Seitenanfang: {mismatch_line(lecture).removeprefix('  stimmt nicht: ')} → Seite {page}",
            GREEN,
        ))

    missing = [
        lecture
        for lecture in lectures
        if assignments[lecture.id].seite_pdf is None
    ]
    if pending and missing:
        print(color(f"  nach dem zweiten Lauf stimmen nicht: {len(missing)}", RED))
        for lecture in missing:
            print(color(mismatch_line(lecture), RED))
    model_hinweis = payload.get("hinweis") if isinstance(payload.get("hinweis"), str) else ""
    hinweis = build_hinweis(model_hinweis, lectures, assignments, volume)
    second_found = sum(
        1
        for lecture in lectures
        if assignments[lecture.id].markierung == "zweiter_lauf"
    )
    stats = (
        f"katalog={len(lectures)} erster_lauf={first_found} "
        f"zweiter_lauf={second_found} nicht_im_band={len(missing)}"
    )
    text = render_yaml(pdf, lectures, assignments, hinweis, stats)
    target = output_path(pdf)
    temporary = target.with_suffix(".yaml.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(target)
    print(f"  {stats} | {usage_note(usages)}")
    print(f"  {target}")
    for lecture in lectures:
        mark = assignments[lecture.id].markierung
        if mark == "zweiter_lauf":
            print(
                f"  zweiter Lauf: {lecture.datum} {lecture.id} "
                f"→ Seite {assignments[lecture.id].seite_pdf}"
            )
    return len(missing)


def parse_only(values: list[str]) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        for part in value.split(","):
            token = normalize_ga_token(part)
            if token:
                tokens.add(token)
    return tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ga-dir", type=Path, default=DEFAULT_GA_DIR)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--from", dest="start", default="64", help="Erste GA-Nummer, Standard 64")
    parser.add_argument("--only", action="append", default=[], help="Nur diese GA, z.B. 64 oder 68a")
    parser.add_argument("--force", action="store_true", help="Vorhandene Seitenverzeichnisse neu schreiben")
    parser.add_argument("--dry-run", action="store_true", help="Nur PDF und Katalog zeigen, kein API-Aufruf")
    parser.add_argument("--limit", type=int, default=0, help="Höchstens N Bände verarbeiten")
    parser.add_argument("--model", default=os.environ.get("RAGRUN_DEEPSEEK_CHAT_MODEL", ""))
    args = parser.parse_args()

    load_env()
    model = args.model or os.environ.get("RAGRUN_DEEPSEEK_CHAT_MODEL") or "deepseek-v4-flash"
    start_token = normalize_ga_token(args.start)
    if not start_token or not start_token.isdigit():
        raise SystemExit(f"--from muss eine GA-Nummer sein, nicht {args.start!r}")
    if not args.ga_dir.is_dir():
        raise SystemExit(f"PDF-Verzeichnis fehlt: {args.ga_dir}")
    if not args.catalog.is_file():
        raise SystemExit(f"Katalog fehlt: {args.catalog}")
    if shutil_which("pdftotext") is None:
        raise SystemExit("pdftotext (poppler) nicht gefunden")

    catalog = load_catalog(args.catalog)
    pdfs = list_pdfs(args.ga_dir, int(start_token), parse_only(args.only))
    if args.limit > 0:
        pdfs = pdfs[: args.limit]
    if not pdfs:
        raise SystemExit("Keine passenden PDFs")

    client = None
    if not args.dry_run:
        base = os.environ.get("RAGRUN_DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
        client = httpx.Client(
            base_url=base,
            headers={"Authorization": f"Bearer {api_key()}"},
            timeout=httpx.Timeout(180.0, connect=20.0),
        )

    done = 0
    skipped = 0
    failed = 0
    interrupted = False
    try:
        for pdf in pdfs:
            parsed = pdf_token(pdf)
            if not parsed:
                continue
            token = parsed[2]
            lectures = catalog.get(token, [])
            target = output_path(pdf)
            if not lectures:
                print(f"{pdf.name}: keine Katalogvorträge, übersprungen")
                skipped += 1
                continue
            if target.exists() and not args.force:
                print(f"{pdf.name}: {target.name} vorhanden, übersprungen")
                skipped += 1
                continue
            try:
                failed += process_pdf(pdf, lectures, client, model, args.dry_run)
                done += 1
            except KeyboardInterrupt:
                interrupted = True
                print("Abbruch.", file=sys.stderr)
                break
            except Exception as exc:  # noqa: BLE001 — ein Band soll die Reihe nicht abbrechen
                failed += 1
                print(color(f"{pdf.name}: FEHLER {exc}", RED, sys.stderr), file=sys.stderr)
    finally:
        if client is not None:
            client.close()
    label = "geprüft" if args.dry_run else "geschrieben"
    summary = f"fertig: {done} {label}, {skipped} übersprungen, {failed} Fehler"
    print(color(summary, RED if failed or interrupted else GREEN))
    if interrupted:
        raise SystemExit(130)


def shutil_which(name: str) -> str | None:
    from shutil import which

    return which(name)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Abbruch.", file=sys.stderr)
        raise SystemExit(130) from None
