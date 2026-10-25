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
    ]
    hits, weg = _split_hits_and_wegweiser(raw, limit_hits=5, limit_wegweiser=5)
    assert [h["chunk_id"] for h in hits] == ["b1"]
    assert {w["chunk_id"] for w in weg} == {"t1", "c1"}  # raw rows before wegweiser mapping


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
