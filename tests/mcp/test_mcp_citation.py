"""Unit tests for MCP citation helpers."""
import asyncio
from unittest.mock import MagicMock

from app.mcp_server.citation import (
    fetch_citation,
    format_zitierform,
    note_display_title,
    resolve_segment_slug,
)
from app.mcp_server.tools import (
    _content_too_long,
    _validate_search_types,
    lecture_display_title,
)


def test_format_zitierform_chapter_paragraph():
    z = format_zitierform(
        "Die Philosophie der Freiheit",
        "Das Bewusstsein des Freien Willens",
        12,
    )
    assert z == (
        "Die Philosophie der Freiheit, "
        "Das Bewusstsein des Freien Willens, Absatz 12"
    )


def test_format_zitierform_chapter_only():
    z = format_zitierform("Die Philosophie der Freiheit", "Einleitung", None)
    assert z == "Die Philosophie der Freiheit, Einleitung"


def test_note_display_title_stored():
    title, source = note_display_title("Mein Titel", "# Heading\nBody", None)
    assert title == "Mein Titel"
    assert source == "stored"


def test_note_display_title_heading():
    title, source = note_display_title(
        None,
        "# Filo × Claude\n\nErster Absatz.",
        None,
    )
    assert title == "Filo × Claude"
    assert source == "heading"


def test_note_display_title_citation_fallback():
    citation = {"zitierform": "Band, Kapitel, Absatz 1"}
    title, source = note_display_title(None, "", citation)
    assert title == "Band, Kapitel, Absatz 1"
    assert source == "citation"


def test_resolve_segment_slug_by_slug():
    engine = MagicMock()
    conn = MagicMock()
    engine.connect.return_value.__enter__.return_value = conn
    result = MagicMock()
    result.mappings.return_value.first.return_value = {
        "segment_slug": "xiii-der-wert-des-lebens-pessimismus-und-optimismus"
    }
    conn.execute.return_value = result

    slug = resolve_segment_slug(
        engine,
        "4b8e4c2a-3f1b-4d2e-9c4a-8e4f4b2c3a1d",
        "xiii-der-wert-des-lebens-pessimismus-und-optimismus",
    )
    assert slug == "xiii-der-wert-des-lebens-pessimismus-und-optimismus"
    assert conn.execute.call_count == 1


def test_resolve_segment_slug_by_index():
    engine = MagicMock()
    conn = MagicMock()
    engine.connect.return_value.__enter__.return_value = conn

    slug_miss = MagicMock()
    slug_miss.mappings.return_value.first.return_value = None
    index_hit = MagicMock()
    index_hit.mappings.return_value.first.return_value = {
        "segment_slug": "xiii-der-wert-des-lebens-pessimismus-und-optimismus"
    }
    conn.execute.side_effect = [slug_miss, index_hit]

    slug = resolve_segment_slug(
        engine,
        "4b8e4c2a-3f1b-4d2e-9c4a-8e4f4b2c3a1d",
        "15",
    )
    assert slug == "xiii-der-wert-des-lebens-pessimismus-und-optimismus"
    assert conn.execute.call_count == 2


def test_validate_search_types_rejects_unknown():
    err = _validate_search_types(["unbekannt"])
    assert err is not None
    assert err["code"] == "invalid_types"


def test_validate_search_types_allows_known():
    assert _validate_search_types(["text", "chapter_summary"]) is None


def test_content_too_long_limit():
    assert _content_too_long("ok") is None
    err = _content_too_long("x" * (1_048_576 + 1))
    assert err is not None
    assert err["code"] == "content_too_long"


def test_fetch_citation_compares_paragraph_id_as_text():
    engine = MagicMock()
    conn = MagicMock()
    engine.connect.return_value.__enter__.return_value = conn
    result = MagicMock()
    result.mappings.return_value.first.return_value = None
    conn.execute.return_value = result

    assert fetch_citation(engine, "4716e5de-80df-45a3-b906-a5d829ef65b6") is None
    sql = str(conn.execute.call_args.args[0])
    assert "p.id::text = :pid" in sql
    assert "CAST(:pid AS uuid)" not in sql


def test_lecture_display_title_prefers_stored():
    assert (
        lecture_display_title(
            "Der menschliche und der kosmische Gedanke — Erster Vortrag",
            "Erster Vortrag",
            None,
            "Berlin",
            "20.01.1914",
            "Der menschliche und der kosmische Gedanke",
            "19140120b",
        )
        == "Der menschliche und der kosmische Gedanke — Erster Vortrag"
    )


def test_lecture_display_title_anlass_when_empty():
    title = lecture_display_title(
        None,
        "",
        "Mitgliederversammlung",
        "Dornach",
        "10.10.1920",
        None,
        "19201010a",
    )
    assert title == "Mitgliederversammlung (Dornach, 10.10.1920)"
    assert "GA" not in title


def test_lecture_display_title_never_embeds_ga():
    title = lecture_display_title(
        None,
        "Erster Vortrag",
        None,
        "Berlin",
        "20.01.1914",
        None,
        "19140120b",
    )
    assert title == "Erster Vortrag"
    assert "GA" not in title


def test_get_lecture_info_requires_source_id():
    from app.mcp_server.tools import get_lecture_info

    err = asyncio.run(get_lecture_info("  "))
    assert err["code"] == "empty_source_id"


def test_list_lectures_requires_ga():
    from app.mcp_server.tools import list_lectures

    err = asyncio.run(list_lectures(""))
    assert err["code"] == "empty_ga"


def test_resolve_chunk_types_ignores_unknown():
    from app.services.app_search_service import _resolve_chunk_types

    types = _resolve_chunk_types(["bogus_type", "chapter_summary"])
    assert types == ["chapter_summary"]
    assert "bogus_type" not in types
