"""Unit tests for passage_read_service.read_passages."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.services.passage_read_service import read_passages


def test_read_passages_requires_address():
    engine = MagicMock()
    out = read_passages(engine)
    assert out.get("error")


@patch("app.services.passage_read_service._attach_citations")
@patch("app.services.passage_read_service._compute_nav")
@patch("app.services.passage_read_service._segment_bounds")
@patch("app.services.passage_read_service._fetch_range")
@patch("app.services.passage_read_service._fetch_one")
def test_read_passages_by_paragraph_id_merges_single_hit(
    mock_fetch_one,
    mock_fetch_range,
    mock_bounds,
    mock_nav,
    mock_attach,
):
    anchor = {
        "paragraph_id": "aaa",
        "source_id": "book-1",
        "segment_index": 2,
        "paragraph_number": 5,
        "text": "Hello",
        "segment_title": "Ch",
        "segment_slug": "ch-2",
        "source_title": "T",
        "source_author": "A",
    }
    mock_fetch_one.return_value = anchor
    mock_bounds.return_value = (1, 10)
    mock_fetch_range.return_value = [anchor]
    mock_nav.return_value = {"prev": None, "next": None}

    out = read_passages(MagicMock(), paragraph_id="aaa", before=0, after=0)
    assert out["text"] == "Hello"
    assert out["paragraph_id"] == "aaa"
    assert len(out["paragraphs"]) == 1
    assert mock_fetch_range.call_args[0][1:] == ("book-1", 2, 5, 5)


@patch("app.services.passage_read_service._attach_citations")
@patch("app.services.passage_read_service._compute_nav")
@patch("app.services.passage_read_service._segment_bounds")
@patch("app.services.passage_read_service._fetch_range")
def test_read_passages_by_source_range(
    mock_fetch_range,
    mock_bounds,
    mock_nav,
    mock_attach,
):
    para = {
        "paragraph_id": "bbb",
        "source_id": "src",
        "segment_index": 1,
        "paragraph_number": 3,
        "text": "Line",
        "segment_title": "S",
        "segment_slug": "s",
        "source_title": "T",
        "source_author": "A",
    }
    mock_fetch_range.return_value = [para]
    mock_bounds.return_value = (1, 8)
    mock_nav.return_value = {"prev": None, "next": None}

    out = read_passages(
        MagicMock(),
        source_id="src",
        segment_index=1,
        from_paragraph=3,
        to_paragraph=3,
    )
    assert out["paragraphs"][0]["paragraph_id"] == "bbb"
    assert "text" not in out or out.get("text") == "Line"


@patch("app.services.passage_read_service._fetch_one")
def test_read_passages_unknown_id(mock_fetch_one):
    mock_fetch_one.return_value = None
    out = read_passages(MagicMock(), paragraph_id="00000000-0000-0000-0000-000000000000")
    assert out["error"] == "paragraph not found"
