"""Prompt loading and rendering for the sigrid-von-gleich worldview translations.

This module loads per-worldview prompt files from:
  ragkeep/assistants/sigrid-von-gleich/worldviews/<WV>/prompts/

Required files per requested worldview:
  - concept-explain-what.prompt
  - concept-explain-how.prompt

Worldview description is sourced from (first existing):
  - instructions.prompt
  - instructions.md
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Mapping

from app.config import settings


def _resolve_assistants_root() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    configured = Path(settings.assistants_root)
    return configured if configured.is_absolute() else (repo_root / configured)


def _worldview_prompts_dir(worldview: str) -> Path:
    return (
        _resolve_assistants_root()
        / "sigrid-von-gleich"
        / "worldviews"
        / worldview
        / "prompts"
    )


def _read_text(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return path.read_text(encoding="utf-8").strip()


@lru_cache(maxsize=256)
def _load_prompt_file(worldview: str, filename: str) -> str:
    return _read_text(_worldview_prompts_dir(worldview) / filename)


@lru_cache(maxsize=256)
def _load_worldview_description(worldview: str) -> str:
    prompts_dir = _worldview_prompts_dir(worldview)
    for name in ("instructions.prompt", "instructions.md"):
        p = prompts_dir / name
        if p.is_file():
            return _read_text(p)
    raise FileNotFoundError(str(prompts_dir / "instructions.(prompt|md)"))


def ensure_worldview_prompts_exist(*, worldview: str) -> None:
    """Fail-fast validation for a requested worldview."""
    prompts_dir = _worldview_prompts_dir(worldview)
    required = [
        prompts_dir / "concept-explain-what.prompt",
        prompts_dir / "concept-explain-how.prompt",
    ]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise ValueError(
            "Missing sigrid-von-gleich worldview prompt(s): " + ", ".join(missing)
        )


def render_worldview_what(*, worldview: str, concept_explanation: str, context1_k5: str) -> str:
    """Render the per-worldview WHAT prompt into a user message content string."""
    ensure_worldview_prompts_exist(worldview=worldview)
    template = _load_prompt_file(worldview, "concept-explain-what.prompt")
    desc = _load_worldview_description(worldview)
    return (
        template.replace("{{CONCEPT_EXPLANATION}}", (concept_explanation or "").strip())
        .replace("{{CONTEXT1_K5}}", (context1_k5 or "").strip())
        .replace("{{WORLDVIEW_DESCRIPTION}}", (desc or "").strip())
        .strip()
    )


def render_worldview_how(
    *,
    worldview: str,
    concept_explanation: str,
    context2_k10: str,
    main_points: str,
) -> str:
    """Render the per-worldview HOW prompt into a user message content string."""
    ensure_worldview_prompts_exist(worldview=worldview)
    template = _load_prompt_file(worldview, "concept-explain-how.prompt")
    desc = _load_worldview_description(worldview)
    return (
        template.replace("{{CONCEPT_EXPLANATION}}", (concept_explanation or "").strip())
        .replace("{{CONTEXT2_K10}}", (context2_k10 or "").strip())
        .replace("{{MAIN_POINTS}}", (main_points or "").strip())
        .replace("{{WORLDVIEW_DESCRIPTION}}", (desc or "").strip())
        .strip()
    )


def build_chat_messages(*, user_content: str) -> list[Mapping[str, str]]:
    # Sigrid prompt files already contain the task framing. Keep system neutral.
    return [
        {"role": "system", "content": "Du bist ein hilfreicher Assistent."},
        {"role": "user", "content": user_content},
    ]

