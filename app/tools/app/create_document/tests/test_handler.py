"""Tests für create_document (Contract §4.1)."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import yaml

from app.tools.app.create_document.handler import run
from app.tools.limits import MAX_DOCUMENT_CHARS
from app.tools.types import ToolContext

CASES_DIR = Path(__file__).parent / "cases"


def _ctx() -> ToolContext:
    return ToolContext(user_id="user-1")


def test_create_document_from_case() -> None:
    case = yaml.safe_load((CASES_DIR / "create_document.yaml").read_text(encoding="utf-8"))
    result = asyncio.run(run(_ctx(), case["args"]))
    assert result.result_key == "suggested_document"
    assert result.payload["title"] == case["expected_payload"]["title"]
    assert result.payload["summary_for_chat"] == case["expected_payload"]["summary_for_chat"]
    assert result.payload["content"] == case["args"]["content"]


def test_create_document_rejects_empty_title() -> None:
    with pytest.raises(ValueError):
        asyncio.run(run(_ctx(), {"title": "  ", "content": "# X\n\nfoo"}))


def test_create_document_rejects_empty_content() -> None:
    with pytest.raises(ValueError):
        asyncio.run(run(_ctx(), {"title": "X", "content": "   "}))


def test_create_document_rejects_over_limit() -> None:
    with pytest.raises(ValueError):
        asyncio.run(run(_ctx(), {"title": "X", "content": "a" * (MAX_DOCUMENT_CHARS + 1)}))
