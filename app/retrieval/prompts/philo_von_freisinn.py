"""Prompts for the philo-von-freisinn agent."""
from __future__ import annotations

from typing import Iterable, Mapping

from app.retrieval.models import RetrievedSnippet


def build_concept_explain_prompt(
    concept: str,
    primary: Iterable[RetrievedSnippet],
    expanded: Iterable[RetrievedSnippet],
) -> list[Mapping[str, str]]:
    context_lines: list[str] = []
    primary_list = list(primary)
    expanded_list = list(expanded)

    if primary_list:
        context_lines.append("Primäre Treffer:")
        for hit in primary_list:
            context_lines.append(f"- {hit.text}")
    if expanded_list:
        context_lines.append("\nZusätzliche Informationen:")
        for hit in expanded_list:
            context_lines.append(f"- {hit.text}")
    context_block = "\n".join(context_lines)

    system = "\n".join(
        [
            'Du bist „Philo-von-Freisinn“:',
            "- Philosophischer Assistent, Fokus auf individuelle Freiheit, logisch präzise.",
            "- Ton: klar, zugänglich, lebendig und humorvoll, aber ohne Ironie oder Personalisierungen.",
            "- Keine Fremdworte, die Steiner nicht nutzt.",
            "Aufgabe: Erkläre einen Begriff im Kontext der Sammlung Rudolf Steiner (Primär- und Sekundärliteratur plus Augmentierungen wie summaries, concepts).",
        ]
    )
    user = "\n".join(
        [
            f'Erkläre den Begriff: "{concept}".',
            "Schreibe für eine 16-jährige Leserin.",
            "Nutze nur den gelieferten Kontext; erfinde nichts.",
            "Keine Zitate, sondern erklärend zusammenfassen.",
            "",
            "Kontext:",
            context_block,
        ]
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

