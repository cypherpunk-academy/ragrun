"""Prompt builders for authentic concept retrieval/explanation (Steiner-first).

Prompt texts live under `ragrun/ragkeep/assistants/philo-von-freisinn/prompts`
and are loaded from disk (cached) for clarity and easy iteration.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Mapping

from app.config import settings
from app.retrieval.prompts.philo_von_freisinn import load_system_prompt


def _resolve_prompts_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    configured = Path(settings.assistants_root)
    assistants_root = configured if configured.is_absolute() else (repo_root / configured)
    return (
        assistants_root
        / "philo-von-freisinn"
        / "prompts"
        / "concepts"
    )


def _load_prompt_file(name: str) -> str:
    path = _resolve_prompts_dir() / name
    if not path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


@lru_cache(maxsize=1)
def _load_prior_user_template() -> str:
    return _load_prompt_file("authentic-steiner-prior.user.prompt")


@lru_cache(maxsize=1)
def _load_verify_system_template() -> str:
    return _load_prompt_file("authentic-steiner-verify.system.prompt")


@lru_cache(maxsize=1)
def _load_verify_user_template() -> str:
    return _load_prompt_file("authentic-steiner-verify.user.prompt")


@lru_cache(maxsize=1)
def _load_verify_query_user_template() -> str:
    return _load_prompt_file("authentic-steiner-verify-query.user.prompt")


@lru_cache(maxsize=1)
def _load_lexicon_user_template() -> str:
    return _load_prompt_file("authentic-steiner-lexicon.user.prompt")


def build_steiner_prior_prompt(*, concept: str, primary_books_list: str = "") -> List[Mapping[str, str]]:
    """Step 1.1: Ask model for Steiner-style meaning without retrieval."""
    system = load_system_prompt()
    user = _load_prior_user_template().format(
        concept=concept,
        primary_books_list=primary_books_list,
        min_words=settings.ace_chunk_min_words,
        target_words=settings.ace_chunk_target_words,
        max_words=settings.ace_chunk_max_words,
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_steiner_verify_prompt(*, steiner_prior_text: str, context: str) -> List[Mapping[str, str]]:
    """Step 1.2: Verify prior text against retrieved Steiner context."""
    system = _load_verify_system_template()
    user = _load_verify_user_template().format(
        steiner_prior_text=steiner_prior_text,
        context=context,
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_steiner_verify_query_prompt(*, b_section: str, primary_books_list: str = "") -> List[Mapping[str, str]]:
    """Generate a short RAG/Qdrant query from section (b) (unsupported claims)."""
    system = load_system_prompt()
    user = _load_verify_query_user_template().format(
        b_section=(b_section or "").strip(),
        primary_books_list=(primary_books_list or "").strip(),
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_steiner_lexicon_prompt(
    *, concept: str, context: str, verification_report: str | None
) -> List[Mapping[str, str]]:
    """Step 1.3: Build the final lexicon entry grounded in context + verification."""
    system = load_system_prompt()
    user = _load_lexicon_user_template().format(
        concept=concept,
        context=context,
        verification_report=(verification_report or "").strip(),
        min_words=settings.ace_chunk_min_words,
        target_words=settings.ace_chunk_target_words,
        max_words=settings.ace_chunk_max_words,
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

