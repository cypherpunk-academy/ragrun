from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.config import settings
from app.retrieval.graphs.translate_to_worldview import run_translate_to_worldview_graph
from app.retrieval.prompts import sigrid_von_gleich_worldviews as sigrid_prompts
from app.retrieval.api.translate_to_worldview import _validate_worldviews


class _Dummy:
    """Placeholder client object for parameters that won't be used in fail-fast paths."""


@pytest.mark.asyncio
async def test_translate_to_worldview_fails_fast_on_missing_prompt_files(tmp_path, monkeypatch):
    # Point assistants_root to a temp directory so the test is hermetic.
    assistants_root = tmp_path / "assistants"
    monkeypatch.setattr(settings, "assistants_root", str(assistants_root))

    # Clear prompt caches (they are keyed by worldview/filename, not root).
    sigrid_prompts._load_prompt_file.cache_clear()  # type: ignore[attr-defined]
    sigrid_prompts._load_worldview_description.cache_clear()  # type: ignore[attr-defined]

    # Create one complete worldview, and one incomplete.
    # Complete: Mathematismus (what/how + instructions)
    math_prompts = (
        assistants_root / "sigrid-von-gleich" / "worldviews" / "Mathematismus" / "prompts"
    )
    math_prompts.mkdir(parents=True)
    (math_prompts / "concept-explain-what.prompt").write_text("WHAT {{CONCEPT_EXPLANATION}}", encoding="utf-8")
    (math_prompts / "concept-explain-how.prompt").write_text("HOW {{MAIN_POINTS}}", encoding="utf-8")
    (math_prompts / "instructions.prompt").write_text("DESC", encoding="utf-8")

    # Incomplete: Idealismus (missing what/how)
    ideal_prompts = (
        assistants_root / "sigrid-von-gleich" / "worldviews" / "Idealismus" / "prompts"
    )
    ideal_prompts.mkdir(parents=True)
    (ideal_prompts / "instructions.md").write_text("DESC", encoding="utf-8")

    with pytest.raises(ValueError) as exc:
        await run_translate_to_worldview_graph(
            text="Base text",
            worldviews=["Mathematismus", "Idealismus"],
            collection="test",
            embedding_client=_Dummy(),  # type: ignore[arg-type]
            qdrant_client=_Dummy(),  # type: ignore[arg-type]
            chat_client=_Dummy(),  # type: ignore[arg-type]
            concept="c",
        )

    assert "Missing sigrid-von-gleich worldview prompt(s)" in str(exc.value)


def test_translate_to_worldview_validates_allowed_worldviews():
    with pytest.raises(HTTPException) as exc:
        _validate_worldviews(["NotAWorldview"])
    assert exc.value.status_code == 400
    assert "Unknown worldview" in str(exc.value.detail)

