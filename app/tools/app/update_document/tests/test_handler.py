"""Tests für update_document (Contract §4.3)."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import yaml

from app.tools.app.update_document.handler import run
from app.tools.limits import MAX_DOCUMENT_CHARS
from app.tools.types import ToolContext

CASES_DIR = Path(__file__).parent / "cases"


def _load_case(name: str) -> dict:
    return yaml.safe_load((CASES_DIR / name).read_text(encoding="utf-8"))


def _ctx(linked_document_id: str) -> ToolContext:
    return ToolContext(user_id="user-1", linked_document_id=linked_document_id)


@pytest.mark.parametrize("case_file", ["update_paragraph.yaml", "update_section.yaml"])
def test_update_document_cases(case_file: str) -> None:
    case = _load_case(case_file)
    result = asyncio.run(run(_ctx(case["linked_document_id"]), case["args"]))
    assert result.result_key == "suggested_document_update"
    assert result.payload == case["expected_payload"]


def test_update_document_delete_paragraph_needs_no_content() -> None:
    result = asyncio.run(
        run(_ctx("doc-1"), {"operation": "delete_paragraph", "paragraph_id": "ch1.p2"})
    )
    assert result.payload["operation"] == "delete_paragraph"
    assert result.payload["content"] is None


def test_update_document_rejects_unknown_operation() -> None:
    with pytest.raises(ValueError):
        asyncio.run(run(_ctx("doc-1"), {"operation": "replace_all"}))


def test_update_document_rejects_missing_required_field() -> None:
    with pytest.raises(ValueError):
        asyncio.run(run(_ctx("doc-1"), {"operation": "update_paragraph", "content": "x"}))


def test_update_document_rejects_content_over_limit() -> None:
    with pytest.raises(ValueError):
        asyncio.run(
            run(
                _ctx("doc-1"),
                {
                    "operation": "update_paragraph",
                    "paragraph_id": "ch1.p2",
                    "content": "a" * (MAX_DOCUMENT_CHARS + 1),
                },
            )
        )
