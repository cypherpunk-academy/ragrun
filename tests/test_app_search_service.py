"""Tests for app search type mapping."""
from __future__ import annotations

from app.services.app_search_service import _resolve_chunk_types


def test_resolve_chunk_types_default() -> None:
    types = _resolve_chunk_types(None)
    assert "book" in types
    assert "quote" in types


def test_resolve_chunk_types_app_aliases() -> None:
    types = _resolve_chunk_types(["text", "quote"])
    assert "book" in types
    assert "talk" in types
    assert "quote" in types
    assert "quote_explanation" in types
