"""Unit tests for corpus search policy (E10, MCP split helpers)."""
from app.mcp_server.tools import (
    _expand_and_rank_wegweiser,
    _sort_wegweiser_raw,
    _split_hits_and_wegweiser,
)
from app.services.corpus_search_policy import apply_quote_hit_policy


def test_apply_quote_hit_policy_drops_unverified():
    rows = [
        {"chunk_type": "quote", "quote_verified": False, "paragraph_id": "p1", "score": 1.0},
        {"chunk_type": "book", "paragraph_id": "p2", "score": 0.5},
    ]
    out = apply_quote_hit_policy(rows)
    assert len(out) == 1
    assert out[0]["chunk_type"] == "book"


def test_apply_quote_hit_policy_dedupes_paragraph_prefers_book():
    rows = [
        {"chunk_type": "quote", "quote_verified": True, "paragraph_id": "p1", "score": 0.9},
        {"chunk_type": "book", "paragraph_id": "p1", "score": 0.8},
    ]
    out = apply_quote_hit_policy(rows)
    assert len(out) == 1
    assert out[0]["chunk_type"] == "book"


def test_split_hits_and_wegweiser():
    raw = [
        {"chunk_type": "book", "score": 0.5, "chunk_id": "b1"},
        {"chunk_type": "typology", "score": 0.9, "chunk_id": "t1"},
        {"chunk_type": "begriff", "score": 0.7, "chunk_id": "c1"},
        {"chunk_type": "curated_text", "score": 0.95, "chunk_id": "cur1"},
    ]
    hits, weg = _split_hits_and_wegweiser(raw, limit_hits=5, limit_wegweiser=5)
    assert [h["chunk_id"] for h in hits] == ["b1"]
    assert {w["chunk_id"] for w in weg} == {"t1", "c1"}  # raw rows before wegweiser mapping
    # curated_text is neither hit nor wegweiser (expanded via references separately)
    assert "cur1" not in {h["chunk_id"] for h in hits}
    assert "cur1" not in {w["chunk_id"] for w in weg}


def test_curated_text_not_original_hit():
    from app.services.corpus_search_policy import is_augment_hit, is_original_hit

    assert is_original_hit("curated_text") is False
    assert is_augment_hit("curated_text") is False
    assert is_original_hit("book") is True


def test_wegweiser_sort_prefers_member_with_references():
    rows = [
        {
            "chunk_id": "g1",
            "chunk_type": "typology",
            "typology_role": "group",
            "score": 0.9,
        },
        {
            "chunk_id": "m1",
            "chunk_type": "typology",
            "typology_role": "member",
            "score": 0.01,
            "references": [{"chunk_id": "b1"}],
        },
    ]
    ordered = _sort_wegweiser_raw(rows)
    assert ordered[0]["chunk_id"] == "m1"


def test_wegweiser_sort_member_belegt_before_member_redaktionell():
    rows = [
        {
            "chunk_id": "m0",
            "chunk_type": "typology",
            "typology_role": "member",
            "score": 0.5,
        },
        {
            "chunk_id": "m1",
            "chunk_type": "typology",
            "typology_role": "member",
            "score": 0.01,
            "references": [{"chunk_id": "b1"}],
        },
    ]
    ordered = _sort_wegweiser_raw(rows)
    assert ordered[0]["chunk_id"] == "m1"


def test_normalize_chunk_references_accepts_begruendung():
    from app.mcp_server.tools import _normalize_chunk_references

    out = _normalize_chunk_references(
        [{"chunk_id": "c1", "begruendung": "Steiner sagt …", "relevance": 0.8}]
    )
    assert out[0]["description"] == "Steiner sagt …"
    assert out[0]["relevance"] == 0.8


def test_normalize_chunk_references_keeps_ga_angabe_fields():
    from app.mcp_server.tools import _is_ga_angabe_ref, _normalize_chunk_references

    out = _normalize_chunk_references(
        [
            {
                "chunk_id": "ga1",
                "chunk_type": "ga_text",
                "zitierbar": False,
                "snippet": "GA Wortlaut …",
                "ga": "4",
                "page_start": 12,
                "relevance": 0.6,
            }
        ]
    )
    assert out[0]["chunk_type"] == "ga_text"
    assert out[0]["zitierbar"] is False
    assert out[0]["snippet"] == "GA Wortlaut …"
    assert out[0]["ga"] == "4"
    assert out[0]["page_start"] == 12
    assert _is_ga_angabe_ref(out[0]) is True


def test_get_ga_chunk_by_payload_chunk_id(monkeypatch):
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    from app.mcp_server import tools as tools_mod

    client = MagicMock()
    client.scroll_points = AsyncMock(
        return_value=[
            {
                "id": "point-uuid",
                "payload": {
                    "chunk_id": "ga-chunk-1",
                    "chunk_type": "ga_text",
                    "text": "Volltext des GA-Chunks zum Nachladen.",
                    "ga": "224",
                    "page_start": 110,
                    "page_end": 111,
                    "lecture_id": "19230711c",
                    "vortragstitel": "Testvortrag",
                },
            }
        ]
    )
    monkeypatch.setattr(
        "app.core.providers.get_qdrant_client",
        lambda: client,
    )

    out = asyncio.run(tools_mod.get_ga_chunk("ga-chunk-1"))
    assert out.get("code") is None
    assert out["chunk_id"] == "ga-chunk-1"
    assert out["zitierbar"] is False
    assert "Volltext" in out["text"]
    assert out["ga"] == "224"
    client.scroll_points.assert_awaited()
    filt = client.scroll_points.await_args.kwargs["filter_"]
    assert filt["must"][0]["match"]["value"] == "ga-chunk-1"


def test_get_ga_chunk_empty_and_missing(monkeypatch):
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    from app.mcp_server import tools as tools_mod

    assert asyncio.run(tools_mod.get_ga_chunk(""))["code"] == "empty_chunk_id"

    client = MagicMock()
    client.scroll_points = AsyncMock(return_value=[])
    monkeypatch.setattr("app.core.providers.get_qdrant_client", lambda: client)
    assert asyncio.run(tools_mod.get_ga_chunk("missing"))["code"] == "not_found"


def test_hits_from_stored_references_ga_angabe_not_citable(monkeypatch):
    import asyncio
    from unittest.mock import MagicMock

    from app.mcp_server import tools as tools_mod

    async def fake_enrich(items, raw, engine):
        for item in items:
            item["citation"] = {"zitierform": "Buch, S. 1"}

    async def fake_lookup(_ids, _engine):
        return {"book1": "para-1"}

    monkeypatch.setattr(tools_mod, "_enrich_corpus_hits", fake_enrich)
    monkeypatch.setattr(tools_mod, "_lookup_paragraph_ids", fake_lookup)

    engine = MagicMock()
    conn = MagicMock()
    engine.connect.return_value.__enter__.return_value = conn
    conn.execute.return_value.mappings.return_value.all.return_value = [
        {
            "chunk_id": "book1",
            "source_id": "src-book",
            "chunk_type": "book",
            "text": "Korpus-Text",
            "metadata": {"source_title": "Philosophie der Freiheit", "author": "Rudolf Steiner"},
        }
    ]

    hits = asyncio.run(
        tools_mod._hits_from_stored_references(
            engine,
            stored=[
                {
                    "chunk_id": "book1",
                    "chunk_type": "book",
                    "relevance": 0.9,
                    "description": "korpus",
                },
                {
                    "chunk_id": "ga1",
                    "chunk_type": "ga_text",
                    "zitierbar": False,
                    "snippet": "Angabe-Text",
                    "ga": "257",
                    "relevance": 0.5,
                },
            ],
            via_base={"kind": "curated_text", "ref": "cur1", "weg": "references"},
            limit=5,
        )
    )
    assert len(hits) == 2
    book = next(h for h in hits if h["chunk_type"] == "book")
    ga = next(h for h in hits if h["chunk_type"] == "ga_text")
    assert book.get("citation") == {"zitierform": "Buch, S. 1"}
    assert ga["zitierbar"] is False
    assert ga.get("citation") is None
    assert ga["snippet"].startswith("Angabe-Text")
    assert ga["via"]["kind"] == "curated_text"


def test_expand_and_rank_injects_members_for_group(monkeypatch):
    import asyncio

    from app.mcp_server import tools as tools_mod

    group = {
        "chunk_id": "g1",
        "chunk_type": "typology",
        "typology_role": "group",
        "group_id": "feld-6",
        "health": "gesund",
        "typology_slug": "soziale-gesundheit",
        "score": 0.9,
    }

    def fake_fetch(engine, **kwargs):
        assert kwargs["group_id"] == "feld-6"
        return [
            {
                "chunk_id": "m1",
                "chunk_type": "typology",
                "text": "member text",
                "metadata": {
                    "typology_role": "member",
                    "group_id": "feld-6",
                    "health": "gesund",
                    "typology_slug": "soziale-gesundheit",
                    "segment_title": "Unterhalt",
                    "references": [{"chunk_id": "b1", "description": "x"}],
                },
                "refs": [{"chunk_id": "b1", "description": "x"}],
            }
        ]

    monkeypatch.setattr(tools_mod, "_fetch_typology_members_for_group", fake_fetch)

    async def _run():
        return await _expand_and_rank_wegweiser([group], engine=None, limit=5)

    out = asyncio.run(_run())
    ids = [r["chunk_id"] for r in out]
    assert "m1" in ids
    assert ids[0] == "m1"
    assert out[0].get("references")
