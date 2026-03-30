"""Prompt loaders for typology explain (package: sibling *.prompt files)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Mapping

from app.retrieval.prompts.philo_von_freisinn import load_system_prompt


def _dir() -> Path:
    return Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def _load(name: str) -> str:
    path = _dir() / name
    if not path.is_file():
        raise FileNotFoundError(f"Typology prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def build_typology_extract_messages(
    *,
    name: str,
    aliases: List[str],
    chunks: str,
    known_members: List[str] | None = None,
) -> List[Mapping[str, str]]:
    """LLM messages: extract typology members + markdown explanation as JSON."""

    aliases_clean = [a.strip() for a in aliases if isinstance(a, str) and a.strip()]
    if not aliases_clean:
        aliases_block = "(keine zusätzlichen Aliasse)"
    else:
        aliases_block = "\n".join(f"- {a}" for a in aliases_clean)

    members_clean = [m.strip() for m in (known_members or []) if isinstance(m, str) and m.strip()]
    if members_clean:
        known_members_block = "\n".join(f"- {m}" for m in members_clean)
    else:
        known_members_block = "(keine bekannten Mitglieder angegeben – aus Quellen ableiten)"

    system = f"{load_system_prompt()}\n\n{_load('system.prompt')}"
    user = (
        _load("user.prompt")
        .replace("{name}", name)
        .replace("{aliases_block}", aliases_block)
        .replace("{known_members_block}", known_members_block)
        .replace("{chunks}", chunks)
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_typology_verify_messages(
    *,
    name: str,
    members: List[str],
    chunks: str,
) -> List[Mapping[str, str]]:
    """Optional verification: which members are unsupported by chunks."""

    members_clean = [m.strip() for m in members if isinstance(m, str) and m.strip()]
    members_block = "\n".join(f"- {m}" for m in members_clean) if members_clean else "(leer)"

    system = load_system_prompt()
    user = (
        _load("verify.prompt")
        .replace("{name}", name)
        .replace("{members_block}", members_block)
        .replace("{chunks}", chunks)
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
