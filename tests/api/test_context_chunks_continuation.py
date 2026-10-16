"""Regression: paragraph context continuation uses consecutive source_index only."""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from app.api import rag as rag_router


def _row(chunk_id: str, source_index: int, segment_id: str = "seg-iv") -> tuple:
    return (
        chunk_id,
        f"text-{source_index}",
        segment_id,
        "Chapter IV",
        [],
        source_index,
    )


@pytest.fixture
def mock_engine_for_context_chunks(monkeypatch):
    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=None)

    start_rows: list = []
    candidate_rows: list = []

    def make_result(rows):
        mr = MagicMock()
        mr.fetchall = MagicMock(return_value=rows)
        return mr

    def execute_side_effect(_sql, params):
        if "idx_hi" in params:
            return make_result(candidate_rows)
        return make_result(start_rows)

    mock_conn.execute = MagicMock(side_effect=execute_side_effect)
    mock_engine.connect = MagicMock(return_value=mock_conn)

    monkeypatch.setattr(rag_router, "get_engine", lambda: mock_engine)

    def set_rows(start: list, candidates: list) -> None:
        start_rows.clear()
        candidate_rows.clear()
        start_rows.extend(start)
        candidate_rows.extend(candidates)

    return set_rows


def test_context_chunks_stops_at_first_source_index_gap(mock_engine_for_context_chunks):
    """After max_idx 78, indices 82/86 must not appear (chapter IV, paragraph 23 style)."""
    set_rows = mock_engine_for_context_chunks
    set_rows(
        [_row("c75", 75)],
        [
            _row("c76", 76),
            _row("c77", 77),
            _row("c78", 78),
            _row("c82", 82),
            _row("c86", 86),
        ],
    )

    body = asyncio.run(
        rag_router.get_context_chunks(
            collection_name="test-col",
            source_id="book-1",
            segment_id="seg-iv",
            paragraph=23,
        )
    )
    assert body["fallback_used"] is False
    ids = [c["chunk_id"] for c in body["chunks"]]
    idxs = [c["source_index"] for c in body["chunks"]]
    assert ids == ["c75", "c76", "c77", "c78"]
    assert idxs == [75, 76, 77, 78]


def test_context_chunks_no_extra_when_max_already_last(mock_engine_for_context_chunks):
    set_rows = mock_engine_for_context_chunks
    set_rows([_row("c78", 78)], [_row("c82", 82)])
    body = asyncio.run(
        rag_router.get_context_chunks(
            collection_name="test-col",
            source_id="book-1",
            segment_id="seg-iv",
            paragraph=23,
        )
    )
    assert [c["chunk_id"] for c in body["chunks"]] == ["c78"]
