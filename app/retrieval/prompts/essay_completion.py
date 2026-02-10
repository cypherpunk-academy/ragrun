"""Prompt builders for essay completion and verification."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Mapping

from app.config import settings


def _resolve_assistants_root() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    configured = Path(settings.assistants_root)
    return configured if configured.is_absolute() else (repo_root / configured)


@lru_cache(maxsize=128)
def load_essay_system_prompt(assistant: str) -> str:
    prompt_path = (
        _resolve_assistants_root()
        / assistant
        / "prompts"
        / "essays"
        / "instruction.prompt"
    )
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Essay system prompt not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8").strip()


def _resolve_sigrid_essay_prompts_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    configured = Path(settings.assistants_root)
    assistants_root = configured if configured.is_absolute() else (repo_root / configured)
    return assistants_root / "sigrid-von-gleich" / "prompts" / "essays"


def _load_soul_mood_instruction(mood_index: int, mood_name: str) -> str:
    """Load instruction.md for a soul mood."""
    repo_root = Path(__file__).resolve().parents[3]
    configured = Path(settings.assistants_root)
    assistants_root = configured if configured.is_absolute() else (repo_root / configured)
    
    mood_dir = assistants_root / "sigrid-von-gleich" / "soul-moods" / f"{mood_index}_{mood_name}"
    instruction_path = mood_dir / "instruction.md"
    
    if not instruction_path.is_file():
        raise FileNotFoundError(f"Soul mood instruction not found: {instruction_path}")
    return instruction_path.read_text(encoding="utf-8").strip()


def _render_template(template: str, *, vars: Mapping[str, str]) -> str:
    out = template
    for key, value in vars.items():
        out = out.replace(f"{{{key}}}", value)
    return out


def build_completion_prompt(
    *,
    assistant: str,
    essay_title: str,
    mood_index: int,
    mood_name: str,
    mood_planet: str,
    mood_description: str,
    current_text: str,
    primary_books_list: str,
) -> list[Mapping[str, str]]:
    system = load_essay_system_prompt(assistant)
    user = f"""
Du bist ein Essayist. Schreibe den Abschnitt {mood_index} ({mood_name}, {mood_planet}) fuer den Essay:
"{essay_title}"

Seelenstimmung (Zusammenfassung):
{mood_description}

Primare Buecher:
{primary_books_list or "- (keine angegeben)"}

Vorhandener Text (falls vorhanden):
{current_text or "(leer)"}

Anforderungen:
- 200-300 Tokens (keine Stichpunkte)
- Keine Meta-Erklaerungen
- Keine Ueberschriften im Text
- Klare, lesbare Prosa
""".strip()
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_header_prompt(
    *,
    assistant: str,
    essay_title: str,
    mood_index: int,
    mood_name: str,
    text: str,
) -> list[Mapping[str, str]]:
    # Load system prompt from soul mood instruction.md
    system = _load_soul_mood_instruction(mood_index, mood_name)
    
    # Load user prompt from essay_header.prompt template
    prompt_path = _resolve_sigrid_essay_prompts_dir() / "essay_header.prompt"
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Header prompt not found: {prompt_path}")
    
    template = prompt_path.read_text(encoding="utf-8").strip()
    user = _render_template(
        template,
        vars={
            "essay_title": essay_title,
            "mood_name": mood_name,
            "text": text,
        },
    ).strip()
    
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_rewrite_from_draft_prompt(
    *,
    draft_text: str,
    primary_context: str,
    secondary_context: str,
    part: str,
    style: str,
    topic: str,
    essay_parts: str,
    mood_index: int,
    mood_name: str,
) -> list[Mapping[str, str]]:
    """Build rewrite prompt using draft as foundation with primary and secondary contexts.
    
    Uses essay_rewrite_from_draft.prompt template with:
    - {background} = draft_text (draft becomes the Grundgedanke)
    - {primary_context} = primary books chunks
    - {secondary_context} = secondary books chunks
    - {essay-parts-section} = previous parts for parts 2-7, empty for part 1
    """
    # System prompt from soul mood instruction
    system = _load_soul_mood_instruction(mood_index, mood_name)
    
    # Load the dedicated rewrite template
    prompt_path = _resolve_sigrid_essay_prompts_dir() / "essay_rewrite_from_draft.prompt"
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Rewrite prompt template not found: {prompt_path}")
    
    template = prompt_path.read_text(encoding="utf-8").strip()
    
    # Build essay-parts section (empty for part 1, populated for parts 2-7)
    essay_parts_section = ""
    if essay_parts.strip():
        essay_parts_section = f"Führe den Essay-Beginn fort:\n{essay_parts.strip()}\n"
    
    # Render template with all variables
    user = _render_template(
        template,
        vars={
            "part": part,
            "style": style.strip(),
            "topic": topic.strip(),
            "background": draft_text.strip(),
            "essay-parts-section": essay_parts_section,
            "primary_context": primary_context.strip(),
            "secondary_context": secondary_context.strip(),
        },
    ).strip()
    
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_tune_part_prompt(
    *,
    current_text: str,
    modifications: str,
    primary_context: str,
    secondary_context: str,
    part: str,
    style: str,
    essay_parts: str,
    mood_index: int,
    mood_name: str,
) -> list[Mapping[str, str]]:
    """Build tune part prompt with modification instructions.
    
    Uses essay_tune_part.prompt template with:
    - {current_text} = existing part text
    - {modifications} = user modification instructions
    - {primary_context} = primary books chunks
    - {secondary_context} = secondary books chunks
    - {essay-parts-section} = previous parts for context
    """
    # System prompt from soul mood instruction
    system = _load_soul_mood_instruction(mood_index, mood_name)
    
    # Load the tune part template
    prompt_path = _resolve_sigrid_essay_prompts_dir() / "essay_tune_part.prompt"
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Tune part prompt template not found: {prompt_path}")
    
    template = prompt_path.read_text(encoding="utf-8").strip()
    
    # Build essay-parts section
    essay_parts_section = ""
    if essay_parts.strip():
        essay_parts_section = f"{essay_parts.strip()}\n"
    
    # Render template with all variables
    user = _render_template(
        template,
        vars={
            "part": part,
            "style": style.strip(),
            "current_text": current_text.strip(),
            "modifications": modifications.strip(),
            "essay-parts-section": essay_parts_section,
            "primary_context": primary_context.strip(),
            "secondary_context": secondary_context.strip(),
        },
    ).strip()
    
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_rewrite_prompt(
    *,
    assistant: str,
    draft_text: str,
    verification_report: str,
    context: str,
    part: str,
    style: str,
    background: str,
    primary_books_context: str,
) -> list[Mapping[str, str]]:
    """DEPRECATED: Old rewrite prompt - kept for backward compatibility."""
    system = load_essay_system_prompt("sigrid-von-gleich")
    
    prompt_path = _resolve_sigrid_essay_prompts_dir() / "essay_rewrite.prompt"
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Rewrite prompt not found: {prompt_path}")
    
    template = prompt_path.read_text(encoding="utf-8").strip()
    user = _render_template(
        template,
        vars={
            "part": part,
            "style": style.strip(),
            "draft_text": draft_text.strip(),
            "verification_report": verification_report.strip(),
            "background": background.strip(),
            "primary_books_context": primary_books_context.strip(),
        },
    ).strip()
    
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
