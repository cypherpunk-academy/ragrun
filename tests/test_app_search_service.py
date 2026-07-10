"""Tests for app search type mapping and quote navigation helpers."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.services.app_search_service import (
    _apply_parent_quote_hit,
    _apply_quote_nav_fields_from_meta,
    _chunk_ids_for_paragraph_lookup,
    _dedupe_results_by_chunk_id,
    _fix_quote_navigation,
    _is_quote_chunk_result,
    _quote_navigation_targets,
    _quote_text_needle,
    _resolve_chunk_types,
    _resolve_quote_explanations,
    _resolve_quote_paragraph_row,
    _resolve_quote_paragraph_row_by_text,
)


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


def test_is_quote_chunk_result_by_type() -> None:
    assert _is_quote_chunk_result({"chunk_type": "quote", "source_id": "book-1"})
    assert _is_quote_chunk_result({"chunk_type": "quote_explanation", "source_id": "book-1"})


def test_is_quote_chunk_result_by_source_suffix() -> None:
    assert _is_quote_chunk_result({"source_id": "book-1:quotes"})


def test_chunk_ids_for_paragraph_lookup_excludes_quotes() -> None:
    results = [
        {"chunk_id": "book-chunk", "chunk_type": "book", "source_id": "book-1"},
        {
            "chunk_id": "quote-chunk",
            "chunk_type": "quote",
            "source_id": "book-1:quotes",
        },
    ]
    assert _chunk_ids_for_paragraph_lookup(results) == ["book-chunk"]


def test_quote_navigation_targets_includes_prefilled_paragraph_id() -> None:
    """Generic lookup may set paragraph_id to chunk start; quotes must still be re-resolved."""
    quote = {
        "chunk_id": "quote-chunk",
        "source_id": "book-1:quotes",
        "paragraph_id": "book-1:5:1",
        "_quote_para": 27,
        "segment_id": "chapter-six",
    }
    assert _quote_navigation_targets([quote]) == [quote]


def test_quote_navigation_targets_includes_quotes_without_paragraph_meta() -> None:
    quote = {
        "chunk_id": "quote-chunk",
        "source_id": "book-1:quotes",
        "paragraph_id": None,
        "segment_id": "der-individualismus",
        "text": "Die Empfindungsinhalte sind mir gegeben, nicht aber ihre räumliche Aneinanderreihung",
    }
    assert _quote_navigation_targets([quote]) == [quote]


def test_quote_text_needle_requires_min_length() -> None:
    assert _quote_text_needle("short") is None
    needle = _quote_text_needle(
        "Die Empfindungsinhalte sind mir gegeben, nicht aber ihre räumliche Aneinanderreihung"
    )
    assert needle is not None
    assert needle.startswith("Die Empfindungsinhalte")


def test_resolve_quote_paragraph_row_without_segment_returns_none() -> None:
    conn = MagicMock()
    row = _resolve_quote_paragraph_row(
        conn,
        book_sid="book-1",
        para_num=91,
        segment_id=None,
    )
    assert row is None
    conn.execute.assert_not_called()


def test_quote_text_needle_strips_soft_hyphens() -> None:
    needle = _quote_text_needle(
        "In der Dreigliederung des sozialen Organismus in ein selbstständiges Geistesglied"
    )
    assert needle is not None
    assert "\u00ad" not in needle
    assert "Dreigliederung" in needle


def test_resolve_quote_paragraph_row_by_text_strips_soft_hyphens_in_sql() -> None:
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = (
        "book-1:41:23",
    )

    row = _resolve_quote_paragraph_row_by_text(
        conn,
        book_sid="book-1",
        segment_id="die-dreigliederung-des-sozialen-organismus-die-demokratie-und-der-sozialismus",
        quote_text=(
            "In der Dreigliederung des sozialen Organismus in ein selbstständiges "
            "Geistesglied, ein ebensolches Rechtsglied und Wirtschaftsglied liegt die "
            "Gesundung dieses Organismus."
        ),
    )

    assert row[0] == "book-1:41:23"
    sql = str(conn.execute.call_args[0][0])
    assert "chr(173)" in sql


def test_resolve_quote_paragraph_row_by_text() -> None:
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = ("book-1:7:91",)

    row = _resolve_quote_paragraph_row_by_text(
        conn,
        book_sid="book-1",
        segment_id="der-individualismus",
        quote_text="Die Empfindungsinhalte sind mir gegeben, nicht aber ihre räumliche Aneinanderreihung",
    )

    assert row[0] == "book-1:7:91"
    sql = str(conn.execute.call_args[0][0])
    assert "lower(:needle)" in sql
    assert "segment_id" in sql


def test_resolve_quote_paragraph_row_uses_segment_id() -> None:
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = (
        "book-1:5:27",
    )

    row = _resolve_quote_paragraph_row(
        conn,
        book_sid="book-1",
        para_num=27,
        segment_id="chapter-six",
    )

    assert row[0] == "book-1:5:27"
    sql = str(conn.execute.call_args[0][0])
    assert "segment_index" in sql
    assert "segment_id" in sql


@pytest.mark.asyncio
async def test_fix_quote_navigation_overrides_wrong_paragraph_id() -> None:
    engine = MagicMock()
    conn = MagicMock()
    engine.connect.return_value.__enter__.return_value = conn
    conn.execute.return_value.fetchone.return_value = ("book-1:5:27",)

    results = [
        {
            "chunk_id": "quote-chunk",
            "source_id": "book-1:quotes",
            "paragraph_id": "book-1:5:1",
            "_quote_para": 27,
            "segment_id": "chapter-six",
        }
    ]

    await _fix_quote_navigation(results, engine)

    assert results[0]["source_id"] == "book-1"
    assert results[0]["paragraph_id"] == "book-1:5:27"


@pytest.mark.asyncio
async def test_fix_quote_navigation_by_text_when_paragraph_meta_missing() -> None:
    engine = MagicMock()
    conn = MagicMock()
    engine.connect.return_value.__enter__.return_value = conn
    conn.execute.return_value.fetchone.return_value = ("book-1:7:91",)

    results = [
        {
            "chunk_id": "quote-chunk",
            "source_id": "book-1:quotes",
            "paragraph_id": None,
            "segment_id": "der-individualismus",
            "text": "Die Empfindungsinhalte sind mir gegeben, nicht aber ihre räumliche Aneinanderreihung",
        }
    ]

    await _fix_quote_navigation(results, engine)

    assert results[0]["source_id"] == "book-1"
    assert results[0]["paragraph_id"] == "book-1:7:91"
    sql = str(conn.execute.call_args[0][0])
    assert "lower(:needle)" in sql


def test_apply_quote_nav_fields_from_meta() -> None:
    hit: dict = {}
    meta = {
        "paragraph": 21,
        "paragraph_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "quote_span": {"start": 10, "end": 42},
        "quote_verified": True,
    }
    _apply_quote_nav_fields_from_meta(hit, meta)
    assert hit["paragraph_number"] == 21
    assert hit["paragraph_id"] == "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    assert hit["quote_span"] == {"start": 10, "end": 42}
    assert hit["quote_verified"] is True


def test_apply_parent_quote_hit_replaces_explanation_text() -> None:
    hit = {
        "chunk_id": "expl-chunk",
        "chunk_type": "quote_explanation",
        "text": "Das Gefühl ist eine Art Zwischenreich",
        "snippet": "Das Gefühl ist eine Art Zwischenreich",
        "quote_text": "Das Gefühl ist eine Art Zwischenreich",
        "score": 0.9,
        "_parent_id": "quote-chunk",
    }
    parent = {
        "chunk_id": "quote-chunk",
        "chunk_type": "quote",
        "source_id": "lecture:19190826a:quotes",
        "text": "– Gefühl ist sowohl noch nicht ganz gewordene Erkenntnis",
        "metadata": {
            "paragraph": 21,
            "segment_id": "funfter-vortrag-stuttgart-26-august-1919",
            "author": "Rudolf Steiner",
        },
    }

    _apply_parent_quote_hit(hit, parent)

    assert hit["chunk_id"] == "quote-chunk"
    assert hit["chunk_type"] == "quote"
    assert "Zwischenreich" not in hit["text"]
    assert hit["quote_text"].startswith("– Gefühl")
    assert hit["paragraph_number"] == 21


def test_dedupe_results_by_chunk_id_keeps_highest_score() -> None:
    results = [
        {"chunk_id": "same", "score": 0.5, "text": "low"},
        {"chunk_id": "other", "score": 0.8, "text": "x"},
        {"chunk_id": "same", "score": 0.9, "text": "high"},
    ]
    deduped = _dedupe_results_by_chunk_id(results)
    assert len(deduped) == 2
    same_hit = next(r for r in deduped if r["chunk_id"] == "same")
    assert same_hit["text"] == "high"


@pytest.mark.asyncio
async def test_resolve_quote_explanations_swaps_to_parent() -> None:
    engine = MagicMock()
    conn = MagicMock()
    engine.connect.return_value.__enter__.return_value = conn
    conn.execute.return_value.mappings.return_value.all.return_value = [
        {
            "chunk_id": "quote-chunk",
            "chunk_type": "quote",
            "source_id": "lecture:19190826a:quotes",
            "text": "– Gefühl ist sowohl noch nicht ganz gewordene Erkenntnis",
            "metadata": {"paragraph": 21, "segment_id": "vortrag-5"},
        }
    ]

    results = [
        {
            "chunk_id": "expl-chunk",
            "chunk_type": "quote_explanation",
            "source_id": "lecture:19190826a:quotes",
            "text": "Das Gefühl ist eine Art Zwischenreich",
            "snippet": "Das Gefühl ist eine Art Zwischenreich",
            "quote_text": "Das Gefühl ist eine Art Zwischenreich",
            "score": 0.95,
            "_parent_id": "quote-chunk",
        }
    ]

    await _resolve_quote_explanations(results, engine)

    assert len(results) == 1
    assert results[0]["chunk_type"] == "quote"
    assert results[0]["chunk_id"] == "quote-chunk"
    assert "Zwischenreich" not in results[0]["text"]


@pytest.mark.asyncio
async def test_resolve_quote_explanations_drops_missing_parent() -> None:
    engine = MagicMock()
    conn = MagicMock()
    engine.connect.return_value.__enter__.return_value = conn
    conn.execute.return_value.mappings.return_value.all.return_value = []

    results = [
        {
            "chunk_id": "expl-chunk",
            "chunk_type": "quote_explanation",
            "text": "Das Gefühl ist eine Art Zwischenreich",
            "score": 0.95,
            "_parent_id": "missing-quote",
        }
    ]

    await _resolve_quote_explanations(results, engine)

    assert results == []
