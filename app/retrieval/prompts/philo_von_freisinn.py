"""Prompts for the philo-von-freisinn agent."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping

from app.retrieval.models import RetrievedSnippet


@lru_cache(maxsize=1)
def load_system_prompt() -> str:
    """Load system prompt from the assistants directory (cached)."""
    prompt_path = (
        Path(__file__)
        .resolve()
        .parents[2]
        / "assistants"
        / "philo-von-freisinn"
        / "assistant"
        / "prompts"
        / "instruction.md"
    )
    if not prompt_path.is_file():
        raise FileNotFoundError(f"System prompt file not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8").strip()


@lru_cache(maxsize=1)
def load_concept_explain_user_template() -> str:
    """Load concept-explain user prompt template (cached)."""
    prompt_path = (
        Path(__file__)
        .resolve()
        .parents[2]
        / "assistants"
        / "philo-von-freisinn"
        / "assistant"
        / "prompts"
        / "concept-explain-user.prompt"
    )
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Concept explain prompt file not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8").strip()


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

    system = load_system_prompt()
    template = load_concept_explain_user_template()
    user = template.format(concept=concept, context=context_block)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

