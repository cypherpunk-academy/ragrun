"""Unit tests for embed-chunks SQL condition builder."""
from __future__ import annotations

from app.ingestion.repositories.rag_chunks_repository import build_embed_chunks_condition


def test_build_embed_chunks_condition_empty_chunk_types_returns_none():
    assert (
        build_embed_chunks_condition(
            "assistant-collection",
            chunk_types=[],
        )
        is None
    )


def test_build_embed_chunks_condition_assistant_only_shared_whitelist_empty():
    condition = build_embed_chunks_condition(
        "assistant-collection",
        shared_source_ids=[],
    )
    assert condition is not None
