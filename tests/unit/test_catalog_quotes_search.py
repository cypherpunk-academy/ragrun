"""Unit tests for catalog quotes search mode (search_corpus quotes=author|quoted)."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.app_search_service import (
    _authors_match,
    _extract_span_text,
    _mark_matches_quotes_kind,
    app_search_catalog_quotes,
)


def test_authors_match_normalized():
    assert _authors_match("Rudolf Steiner", "rudolf  steiner")
    assert not _authors_match("Goethe", "Rudolf Steiner")
    assert not _authors_match("", "Goethe")


def test_mark_matches_quotes_kind_author_vs_quoted():
    assert _mark_matches_quotes_kind("Rudolf Steiner", "Rudolf Steiner", "author")
    assert not _mark_matches_quotes_kind("Goethe", "Rudolf Steiner", "author")
    assert _mark_matches_quotes_kind("Goethe", "Rudolf Steiner", "quoted")
    assert not _mark_matches_quotes_kind("Rudolf Steiner", "Rudolf Steiner", "quoted")
    assert _mark_matches_quotes_kind("Goethe", None, "quoted")
    assert not _mark_matches_quotes_kind("Goethe", None, "author")


def test_extract_span_text():
    raw = "Hello quoted world here"
    assert _extract_span_text(raw, {"start": 6, "end": 13}) == "quoted"
    assert _extract_span_text(raw, {"start": -1, "end": 3}) == ""
    assert _extract_span_text(raw, {"start": 0, "end": 99}) == ""


@pytest.mark.asyncio
async def test_app_search_catalog_quotes_invalid_kind():
    out = await app_search_catalog_quotes(
        query="Freiheit",
        quotes_kind="bogus",
        engine=MagicMock(),
    )
    assert out["code"] == "invalid_quotes"


@pytest.mark.asyncio
async def test_app_search_catalog_quotes_empty_query():
    out = await app_search_catalog_quotes(
        query="  ",
        quotes_kind="author",
        engine=MagicMock(),
    )
    assert out["code"] == "empty_query"


@pytest.mark.asyncio
async def test_app_search_catalog_quotes_expands_marks_to_slim_hits():
    snip = MagicMock()
    snip.score = 0.9
    snip.payload = {"chunk_id": "c1", "metadata": {"chunk_id": "c1"}}

    async def fake_hybrid(**_kwargs):
        return [snip]

    async def fake_marks(chunk_ids, engine):
        return {
            "c1": {
                "book_author": "Rudolf Steiner",
                "quote_marks": [
                    {
                        "id": "q1",
                        "author": "Rudolf Steiner",
                        "paragraph_id": "p1",
                        "span": {"start": 0, "end": 8},
                    },
                    {
                        "id": "q2",
                        "author": "Goethe",
                        "paragraph_id": "p2",
                        "span": {"start": 0, "end": 6},
                    },
                ],
            }
        }

    async def fake_texts(paragraph_ids, engine):
        return {"p1": "Freiheit ist gut", "p2": "Goethe sagt"}

    with (
        patch(
            "app.services.app_search_service.hybrid_retrieve",
            new=AsyncMock(side_effect=fake_hybrid),
        ),
        patch(
            "app.services.app_search_service.load_assistant_rag_collection",
            return_value="philo",
        ),
        patch(
            "app.services.app_search_service.load_assistant_embedding_prefixes",
            return_value=("", ""),
        ),
        patch(
            "app.services.app_search_service.get_embedding_client",
            return_value=MagicMock(),
        ),
        patch(
            "app.services.app_search_service.get_qdrant_client",
            return_value=MagicMock(),
        ),
        patch(
            "app.services.app_search_service._load_quote_marks_and_book_authors",
            new=AsyncMock(side_effect=fake_marks),
        ),
        patch(
            "app.services.app_search_service._load_paragraph_texts",
            new=AsyncMock(side_effect=fake_texts),
        ),
        patch(
            "app.mcp_server.citation.passage_return_url",
            return_value="https://example.com/passage/p1",
        ),
    ):
        out = await app_search_catalog_quotes(
            query="Freiheit",
            quotes_kind="author",
            limit=10,
            engine=MagicMock(),
        )

    assert out["authors"] is None
    assert len(out["hits"]) == 1
    hit = out["hits"][0]
    assert hit == {
        "author": "Rudolf Steiner",
        "text": "Freiheit",
        "return_url": "https://example.com/passage/p1",
    }
    assert "score" not in hit
    assert "wegweiser" not in out


@pytest.mark.asyncio
async def test_app_search_catalog_quotes_quoted_plus_author_filter():
    snip = MagicMock()
    snip.score = 0.8
    snip.payload = {"chunk_id": "c1", "metadata": {"chunk_id": "c1"}}

    async def fake_marks(chunk_ids, engine):
        return {
            "c1": {
                "book_author": "Rudolf Steiner",
                "quote_marks": [
                    {
                        "id": "q1",
                        "author": "Goethe",
                        "paragraph_id": "p1",
                        "span": {"start": 0, "end": 6},
                    },
                    {
                        "id": "q2",
                        "author": "Schiller",
                        "paragraph_id": "p2",
                        "span": {"start": 0, "end": 8},
                    },
                ],
            }
        }

    with (
        patch(
            "app.services.app_search_service.hybrid_retrieve",
            new=AsyncMock(return_value=[snip]),
        ),
        patch(
            "app.services.app_search_service.load_assistant_rag_collection",
            return_value="philo",
        ),
        patch(
            "app.services.app_search_service.load_assistant_embedding_prefixes",
            return_value=("", ""),
        ),
        patch(
            "app.services.app_search_service.get_embedding_client",
            return_value=MagicMock(),
        ),
        patch(
            "app.services.app_search_service.get_qdrant_client",
            return_value=MagicMock(),
        ),
        patch(
            "app.services.app_search_service._load_quote_marks_and_book_authors",
            new=AsyncMock(side_effect=fake_marks),
        ),
        patch(
            "app.services.app_search_service._load_paragraph_texts",
            new=AsyncMock(return_value={"p1": "Goethe text", "p2": "Schiller"}),
        ),
        patch(
            "app.mcp_server.citation.passage_return_url",
            side_effect=lambda pid: f"https://example.com/passage/{pid}",
        ),
    ):
        out = await app_search_catalog_quotes(
            query="Natur",
            quotes_kind="quoted",
            author="Goethe",
            limit=10,
            engine=MagicMock(),
        )

    assert len(out["hits"]) == 1
    assert out["hits"][0]["author"] == "Goethe"
    assert out["hits"][0]["text"] == "Goethe"
    assert out["hits"][0]["return_url"].endswith("/passage/p1")


@pytest.mark.asyncio
async def test_app_search_catalog_quotes_empty_returns_authors():
    with (
        patch(
            "app.services.app_search_service.hybrid_retrieve",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "app.services.app_search_service.load_assistant_rag_collection",
            return_value="philo",
        ),
        patch(
            "app.services.app_search_service.load_assistant_embedding_prefixes",
            return_value=("", ""),
        ),
        patch(
            "app.services.app_search_service.get_embedding_client",
            return_value=MagicMock(),
        ),
        patch(
            "app.services.app_search_service.get_qdrant_client",
            return_value=MagicMock(),
        ),
        patch(
            "app.services.app_search_service.list_catalog_quote_authors",
            new=AsyncMock(return_value=["Goethe", "Schiller"]),
        ),
    ):
        out = await app_search_catalog_quotes(
            query="xyz-no-match",
            quotes_kind="quoted",
            author="Unbekannt",
            limit=5,
            engine=MagicMock(),
        )

    assert out["hits"] == []
    assert out["authors"] == ["Goethe", "Schiller"]


@pytest.mark.asyncio
async def test_search_corpus_quotes_mode_delegates():
    from app.mcp_server import tools as tools_mod

    fake = {"hits": [{"author": "A", "text": "t", "return_url": "u"}], "authors": None}
    with patch.object(
        tools_mod,
        "app_search_catalog_quotes",
        new=AsyncMock(return_value=fake),
    ) as mocked:
        out = await tools_mod.search_corpus(
            query="Freiheit",
            quotes="author",
            limit=3,
        )
    assert out == fake
    mocked.assert_awaited_once()
    kwargs = mocked.await_args.kwargs
    assert kwargs["quotes_kind"] == "author"
    assert kwargs["limit"] == 3


@pytest.mark.asyncio
async def test_search_corpus_rejects_bad_quotes():
    from app.mcp_server import tools as tools_mod

    out = await tools_mod.search_corpus(query="x", quotes="nope")
    assert out["code"] == "invalid_quotes"
