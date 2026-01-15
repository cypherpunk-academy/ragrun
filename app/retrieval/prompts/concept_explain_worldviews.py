"""Prompt builders for concept explain worldviews graph."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Mapping


def _resolve_assistants_root() -> Path:
    """Resolve assistants root within this repo."""
    return Path(__file__).resolve().parents[2] / "assistants"


@lru_cache(maxsize=1)
def _load_philo_system_prompt() -> str:
    """Load the system prompt from assistants (cached)."""
    assistants_root = _resolve_assistants_root()
    prompt_path = (
        assistants_root
        / "philo-von-freisinn"
        / "assistant"
        / "prompts"
        / "instruction.prompt"
    )
    if not prompt_path.is_file():
        raise FileNotFoundError(f"System prompt file not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8").strip()


@lru_cache(maxsize=1)
def _load_philo_concept_explain_template() -> str:
    """Load the concept-explain user prompt template from assistants (cached)."""
    assistants_root = _resolve_assistants_root()
    prompt_path = (
        assistants_root
        / "philo-von-freisinn"
        / "assistant"
        / "prompts"
        / "concept-explain-user.prompt"
    )
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Concept explain prompt file not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8").strip()


def build_philo_explain_prompt(*, concept: str, context: str) -> List[Mapping[str, str]]:
    """Prompt for the reasoning step that produces the base concept explanation."""
    system = _load_philo_system_prompt()
    template = _load_philo_concept_explain_template()
    user = template.format(concept=concept, context=context)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_worldview_what_prompt(
    *,
    concept_explanation: str,
    worldview_description: str,
    context: str,
) -> List[Mapping[str, str]]:
    """Prompt for the 'what' step per worldview."""
    system = (
        "Beschreibe den Begriff aus Sicht dieser Weltanschauung. "
        "Nutze nur den Kontext. Wenn unzureichend, gib 'Unzureichender Kontext' zurück."
    )
    user = (
        f"Weltanschauung:\n{worldview_description}\n\n"
        f"Basiserklärung:\n{concept_explanation}\n\n"
        f"Kontext (primär):\n{context}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_worldview_how_prompt(
    *,
    concept_explanation: str,
    worldview_description: str,
    main_points: str,
    context: str,
) -> List[Mapping[str, str]]:
    """Prompt for the 'how' step per worldview."""
    system = (
        "Erläutere, wie die Weltanschauung den Begriff praktisch/interpretiert. "
        "Nutze nur den Kontext. Markiere 'Unzureichender Kontext', wenn nötig."
    )
    user = (
        f"Weltanschauung:\n{worldview_description}\n\n"
        f"Basiserklärung:\n{concept_explanation}\n\n"
        f"Kerngedanken:\n{main_points}\n\n"
        f"Kontext (primär+sekundär):\n{context}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
