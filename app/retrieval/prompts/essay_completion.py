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


def build_verification_prompt(
    *,
    assistant: str,
    draft_text: str,
    context: str,
) -> list[Mapping[str, str]]:
    system = load_essay_system_prompt(assistant)
    user = f"""
Pruefe den folgenden Abschnitt auf Stimmigkeit mit dem Kontext. Gib einen kurzen Bericht:
a) Staerken (in Bezug auf die Buecher)
b) Unklare oder zu starke Behauptungen
c) Verbesserungsvorschlaege
Bewertung: X/10

Abschnitt:
{draft_text}

Kontext (Primary Books):
{context}
""".strip()
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_header_prompt(
    *,
    assistant: str,
    essay_title: str,
    mood_name: str,
    text: str,
) -> list[Mapping[str, str]]:
    system = load_essay_system_prompt(assistant)
    user = f"""
Erstelle einen prägnanten Header (maximal 100 Zeichen) für folgenden Essay-Abschnitt:

Essay-Titel: {essay_title}
Seelenstimmung: {mood_name}

Text:
{text}

Anforderungen:
- Maximal 100 Zeichen
- Keine Meta-Erklärungen
- Eine Zeile
- Fasst den Kern des Texts zusammen
""".strip()
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_length_adjustment_prompt(
    *,
    soul_mood_instruction: str,
    text: str,
    target_tokens: int,
    reason: str,
) -> list[Mapping[str, str]]:
    system = soul_mood_instruction
    user = f"""
Passe die Länge des folgenden Texts an auf {target_tokens} Tokens ({reason}).

Aktueller Text:
{text}

Anforderungen:
- Ziel: {target_tokens} Tokens (±10)
- Inhalt und Stil beibehalten
- Klare, lesbare Prosa
- Keine Meta-Erklärungen
""".strip()
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_authenticity_check_prompt(
    *,
    draft_text: str,
    primary_books_context: str,
    primary_books_list: str,
    soul_mood_description: str,
    mood_name: str,
) -> list[Mapping[str, str]]:
    system = "Du bist ein präziser Prüfer für Authentizität von Steiner-basierten Texten."
    
    prompt_path = _resolve_sigrid_essay_prompts_dir() / "authenticity_check.prompt"
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Authenticity check prompt not found: {prompt_path}")
    
    template = prompt_path.read_text(encoding="utf-8").strip()
    user = _render_template(
        template,
        vars={
            "mood_name": mood_name,
            "soul_mood_description": soul_mood_description,
            "draft_text": draft_text,
            "primary_books_context": primary_books_context,
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
    system = load_essay_system_prompt(assistant)
    
    prompt_path = _resolve_sigrid_essay_prompts_dir() / "rewrite.prompt"
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
