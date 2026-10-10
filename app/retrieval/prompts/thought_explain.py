"""Prompts for thought-explain graph."""
from __future__ import annotations

from typing import Mapping, Sequence


def build_thought_explain_messages(
    *,
    text: str,
    quote: str,
    theme: str,
    alignment: str,
    writing_style: str,
    context: str,
) -> list[dict[str, str]]:
    style = (writing_style or "").strip() or (
        "Lebendig, präzise, unaufgeregt; keine Ironie; keine Fremdworte, die Steiner nicht verwendet."
    )
    system = f"""Du erklärst einen philosophischen Gedanken im Kontext Rudolf Steiners und verwandter Quellen.

Schreibstil: {style}

Aufgabe:
- Erkläre den Gedanken in 250–280 deutschen Wörtern (Zielband hart einhalten).
- Beziehe dich auf das wörtliche Zitat und den bereitgestellten Kontext.
- Keine Meta-Phrasen („Dieser Chunk…“, „Im Folgenden…“).
- Keine Aufzählung von Quellen-IDs; Inhaltlich bleiben.
- Nur Fließtext, kein Markdown, keine Überschrift."""

    user = f"""Gedanke:
{text}

Theme: {theme}
Alignment (zu Steiner): {alignment}

Zitat aus dem Quelltext:
«{quote}»

Kontext aus der Wissensbasis:
{context or "(kein Kontext)"}

Schreibe die Erklärung (250–280 Wörter)."""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def count_words(text: str) -> int:
    return len([w for w in (text or "").split() if w.strip()])
