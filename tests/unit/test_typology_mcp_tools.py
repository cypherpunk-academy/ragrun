"""Unit tests for MCP typology list/search helpers (no live Qdrant)."""
from __future__ import annotations

import pytest

from app.mcp_server.typology_tools import (
    _build_focus,
    _index_typologies,
    _resolve_typology_slug,
    _typology_catalog,
    typology_fields_for_search_result,
)


def _row(
    chunk_id: str,
    slug: str,
    role: str,
    *,
    segment_title: str = "",
    group_id: str | None = None,
    health: str | None = None,
    segment_index: int | None = None,
    counterpart: str | None = None,
    source_id: str = "src-1",
) -> dict:
    meta = {
        "typology_slug": slug,
        "typology_role": role,
        "segment_title": segment_title or chunk_id,
        "source_id": source_id,
    }
    if group_id:
        meta["group_id"] = group_id
    if health:
        meta["health"] = health
    if segment_index is not None:
        meta["segment_index"] = segment_index
    if counterpart:
        meta["counterpart_chunk_id"] = counterpart
    return {"chunk_id": chunk_id, "source_id": source_id, "text": "x", "meta": meta}


def test_typology_catalog_includes_all_overviews():
    rows = [
        _row("ov1", "12-weltanschauungen", "overview", segment_title="12 Weltanschauungen"),
        _row("m1", "12-weltanschauungen", "member", segment_title="Materialismus"),
        _row("ov2", "soziale-gesundheit", "overview", segment_title="Soziale Gesundheit"),
        _row(
            "g1",
            "soziale-gesundheit",
            "group",
            group_id="feld-1",
            health="gesund",
        ),
    ]
    by_slug = _index_typologies(rows)
    catalog = _typology_catalog(by_slug)
    assert len(catalog) == 2
    slugs = {e["typology_slug"] for e in catalog}
    assert slugs == {"12-weltanschauungen", "soziale-gesundheit"}
    sg = next(e for e in catalog if e["typology_slug"] == "soziale-gesundheit")
    assert sg["has_health"] is True
    assert sg["group_count"] == 1


def test_resolve_typology_by_slug_and_title():
    rows = [
        _row("ov", "12-weltanschauungen", "overview", segment_title="12 Weltanschauungen"),
        _row("m1", "12-weltanschauungen", "member", segment_title="Materialismus"),
    ]
    by_slug = _index_typologies(rows)
    assert _resolve_typology_slug(name="12-weltanschauungen", chunk_id=None, by_slug=by_slug) == (
        "12-weltanschauungen"
    )
    assert _resolve_typology_slug(name="12 Weltanschauungen", chunk_id=None, by_slug=by_slug) == (
        "12-weltanschauungen"
    )
    assert _resolve_typology_slug(name=None, chunk_id="m1", by_slug=by_slug) == "12-weltanschauungen"


def test_focus_weltanschauung_siblings():
    rows = [
        _row("ov", "12-weltanschauungen", "overview", segment_title="12 Weltanschauungen"),
        _row("a", "12-weltanschauungen", "member", segment_title="A"),
        _row("b", "12-weltanschauungen", "member", segment_title="B"),
    ]
    by_id = {r["chunk_id"]: r for r in rows}
    focus = _build_focus(focus_id="a", chunks=rows, by_id=by_id)
    assert focus["same_group"]
    assert len(focus["same_group"]) == 1
    assert focus["same_group"][0]["chunk_id"] == "b"


def test_focus_counterpart_and_group():
    rows = [
        _row("g-g", "soziale-gesundheit", "group", group_id="feld-1", health="gesund", counterpart="g-k"),
        _row("g-k", "soziale-gesundheit", "group", group_id="feld-1", health="krank", counterpart="g-g"),
        _row(
            "m-g",
            "soziale-gesundheit",
            "member",
            group_id="feld-1",
            health="gesund",
            segment_index=0,
            counterpart="m-k",
        ),
        _row(
            "m-k",
            "soziale-gesundheit",
            "member",
            group_id="feld-1",
            health="krank",
            segment_index=0,
            counterpart="m-g",
        ),
    ]
    by_id = {r["chunk_id"]: r for r in rows}
    focus = _build_focus(focus_id="m-g", chunks=rows, by_id=by_id)
    assert focus["counterpart"]["chunk_id"] == "m-k"
    assert any(g["chunk_id"] == "g-k" for g in focus["other_groups"])


def test_hit_from_snippet_uses_full_text_override():
    from app.mcp_server.typology_tools import _hit_from_snippet
    from app.retrieval.models import RetrievedSnippet

    snip = RetrievedSnippet(
        score=0.9,
        text="short from qdrant",
        payload={"chunk_id": "abc", "typology_role": "member", "segment_title": "X"},
    )
    hit = _hit_from_snippet(snip, text_override="full text from rag_chunks table")
    assert hit["text"] == "full text from rag_chunks table"
    assert "snippet" not in hit


def test_hit_from_snippet_strips_soft_hyphens():
    from app.mcp_server.typology_tools import _hit_from_snippet
    from app.retrieval.models import RetrievedSnippet

    snip = RetrievedSnippet(
        score=0.9,
        text="Wieder\u00aderkennung",
        payload={"chunk_id": "abc", "typology_role": "member", "segment_title": "X"},
    )
    hit = _hit_from_snippet(snip)
    assert hit["text"] == "Wiedererkennung"
    assert "\u00ad" not in hit["text"]


def test_typology_fields_for_search_result():
    meta = {
        "chunk_type": "typology",
        "typology_role": "member",
        "group_title": "Geist -> Recht",
        "health": "gesund",
        "glieder": ["Sanguinisch", "Cholerisch", " ", "Phlegmatisch"],
    }
    fields = typology_fields_for_search_result(meta)
    assert fields["typology_role"] == "member"
    assert fields["health"] == "gesund"
    assert fields["glieder"] == ["Sanguinisch", "Cholerisch", "Phlegmatisch"]
    assert typology_fields_for_search_result({"chunk_type": "book"}) == {}


def test_build_wegweiser_includes_glieder():
    from app.mcp_server.tools import _build_wegweiser

    raw = {
        "chunk_id": "ov-sinne",
        "chunk_type": "typology",
        "score": 0.5,
        "segment_title": "12 Sinne",
        "typology_role": "overview",
        "glieder": ["Ich-Sinn", "Gedankensinn"],
    }
    meta = {
        "typology_slug": "12-sinne",
        "typology_role": "overview",
        "glieder": ["Ich-Sinn", "Gedankensinn"],
    }
    entry = _build_wegweiser(raw, meta)
    assert entry["glieder"] == ["Ich-Sinn", "Gedankensinn"]
    assert entry["herkunft"] == "augment"
