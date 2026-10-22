"""Unit tests for GA ranking. No Qdrant."""
from __future__ import annotations

import asyncio
import logging

import pytest

from app.retrieval.models import RetrievedSnippet
from app.retrieval.utils.ga_ranking import (
    find_title_lectures,
    fold_text,
    normalize_rrf,
    query_tokens,
    rank_ga_pool,
    select_title_lectures,
    title_score,
    token_matches,
)
from app.retrieval.utils.retrievers import _rrf_fuse, dense_retrieve


def _snip(
    chunk_id: str,
    *,
    score: float = 0.0,
    **meta: object,
) -> RetrievedSnippet:
    body: dict[str, object] = {"chunk_id": chunk_id, "text": "t" * 80, **meta}
    return RetrievedSnippet(text=str(body["text"]), score=score, payload={"payload": body})


def test_query_tokens_fold_stopwords_and_short_tokens() -> None:
    assert query_tokens("Karma Gesetze") == ["karma", "gesetze"]
    assert query_tokens("Der und die") == []
    assert query_tokens("für die Freiheit") == ["freiheit"]
    assert fold_text("Gemüt") == "gemuet"


def test_prefix_match_karma_and_gesetz() -> None:
    assert token_matches("karma", "karmabildung")
    assert token_matches("karmabildung", "karma")
    assert not token_matches("gesetze", "gesetzmaessigkeiten")
    assert token_matches("gesetz", "gesetzmaessigkeiten")
    assert token_matches("gesetze", "gesetze")


def test_title_score_fraction_exact_and_real_karmabildung_title() -> None:
    title = (
        "Gesetze der Karmabildung. Jakob Frohschammer und das Urbild "
        "der Strader-Gestalt aus den Mysteriendramen"
    )
    assert title_score("Karma Gesetze", title) == pytest.approx(1.0)
    assert title_score("Karma Gesetze", "Die verschiedenen Gesetzmäßigkeiten") == pytest.approx(0.0)
    assert title_score("Gesetz", "Gesetzmäßigkeiten der Natur") == pytest.approx(1.0)
    assert title_score("Der und die", "Der und die") == pytest.approx(1.0)
    assert title_score("Gemuet", "Gemüt und Welt") == pytest.approx(1.0)
    assert title_score("Karma Gesetze", "Nur Karma") == pytest.approx(0.5)


def test_rrf_norm_of_rank_one_in_both_lists() -> None:
    snippet = _snip("c1")
    fused = _rrf_fuse([snippet], [snippet], k_fused=5)
    assert fused[0].score == pytest.approx(2 / 61)
    assert normalize_rrf(fused[0].score) == pytest.approx(1.0)
    assert normalize_rrf(1 / 61) == pytest.approx(0.5)


def test_grouping_fallbacks_and_lecture_hits() -> None:
    strong = _snip("a", score=2 / 61, lecture_id="L1", segment_id="other", ga="236")
    weak = _snip("b", score=1 / 61, lecture_id="L1", ga="236")
    ranked = rank_ga_pool(
        [strong, weak],
        query="zzz",
        title_weight=0.0,
        group_by_lecture=True,
        max_per_ga=0,
        limit=10,
    )
    assert [hit.meta["chunk_id"] for hit in ranked] == ["a"]
    assert ranked[0].lecture_hits == 2

    same_segment = rank_ga_pool(
        [
            _snip("s1", score=2 / 61, segment_id="4:11", ga="4"),
            _snip("s2", score=1 / 61, segment_id="4:11", ga="4"),
        ],
        query="zzz",
        title_weight=0.0,
        group_by_lecture=True,
        max_per_ga=0,
        limit=10,
    )
    assert [hit.meta["chunk_id"] for hit in same_segment] == ["s1"]
    assert same_segment[0].lecture_hits == 2

    separate = rank_ga_pool(
        [
            _snip("c1", score=2 / 61, ga="4"),
            _snip("c2", score=1 / 61, ga="4"),
        ],
        query="zzz",
        title_weight=0.0,
        group_by_lecture=True,
        max_per_ga=0,
        limit=10,
    )
    assert [hit.meta["chunk_id"] for hit in separate] == ["c1", "c2"]


def test_max_per_ga_on_off_and_disabled_by_caller_filter() -> None:
    fused = [
        _snip(f"c{i}", score=(5 - i) / 61, lecture_id=f"L{i}", ga="236")
        for i in range(4)
    ]
    capped = rank_ga_pool(
        fused,
        query="zzz",
        title_weight=0.0,
        group_by_lecture=True,
        max_per_ga=3,
        apply_ga_cap=True,
        limit=10,
    )
    assert [hit.meta["chunk_id"] for hit in capped] == ["c0", "c1", "c2"]

    uncapped = rank_ga_pool(
        fused,
        query="zzz",
        title_weight=0.0,
        group_by_lecture=True,
        max_per_ga=0,
        apply_ga_cap=True,
        limit=10,
    )
    assert len(uncapped) == 4

    filtered = rank_ga_pool(
        fused,
        query="zzz",
        title_weight=0.0,
        group_by_lecture=True,
        max_per_ga=1,
        apply_ga_cap=False,
        limit=10,
    )
    assert len(filtered) == 4


def test_tie_keeps_input_order() -> None:
    ranked = rank_ga_pool(
        [
            _snip("a", score=1 / 61, lecture_id="L1", ga="1"),
            _snip("b", score=1 / 61, lecture_id="L2", ga="2"),
        ],
        query="zzz",
        title_weight=0.0,
        group_by_lecture=True,
        max_per_ga=0,
        limit=10,
    )
    assert [hit.meta["chunk_id"] for hit in ranked] == ["a", "b"]


def test_title_signal_can_outrank_weak_rrf() -> None:
    weak = _snip("early", score=0.2 * (2 / 61), lecture_id="L1", ga="96", vortragstitel="Allgemeines")
    title_hit = _snip(
        "late",
        score=0.9,
        lecture_id="19240511c",
        ga="236",
        vortragstitel="Gesetze der Karmabildung",
    )
    ranked = rank_ga_pool(
        [weak],
        [title_hit],
        query="Karma Gesetze",
        title_weight=0.35,
        group_by_lecture=True,
        max_per_ga=0,
        limit=10,
    )
    assert ranked[0].meta["chunk_id"] == "late"
    assert ranked[0].rrf_norm == pytest.approx(0.0)
    assert ranked[0].title_score == pytest.approx(1.0)
    assert ranked[0].score == pytest.approx(0.35)


def test_select_title_lectures_skips_pool_and_other_volumes() -> None:
    rows = [
        {"id": "19240511c", "ga": "236", "vortragstitel": "Gesetze der Karmabildung"},
        {"id": "19041020", "ga": "53", "vortragstitel": "Reinkarnation und Karma"},
        {"id": "in-pool", "ga": "236", "vortragstitel": "Gesetze des Karma"},
        {"id": "early", "ga": "96", "vortragstitel": "Gesetzmäßigkeiten des Karma"},
    ]
    picked = select_title_lectures(
        rows,
        "Karma Gesetze",
        exclude={"in-pool"},
        gas=["236"],
        limit=10,
    )
    assert [row["id"] for row in picked] == ["19240511c"]


def test_title_path_failure_returns_empty(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    def boom() -> list[dict[str, str]]:
        raise RuntimeError("catalog down")

    monkeypatch.setattr("app.retrieval.utils.ga_ranking._load_catalog_titles", boom)
    with caplog.at_level(logging.WARNING):
        rows = asyncio.run(
            find_title_lectures("Karma Gesetze", exclude=set(), gas=None)
        )
    assert rows == []
    assert "title path" in caplog.text.lower()


def test_dense_retrieve_reuses_query_vector() -> None:
    class Emb:
        async def embed_texts(self, texts: list[str]) -> object:
            raise AssertionError("embed called")

    class Qdrant:
        def __init__(self) -> None:
            self.vectors: list[list[float]] = []

        async def search_points(self, collection: str, vector: list[float], **kwargs: object) -> list[dict]:
            self.vectors.append(list(vector))
            return [
                {
                    "score": 0.9,
                    "payload": {"text": "x" * 400, "chunk_id": "c1"},
                }
            ]

    qdrant = Qdrant()
    hits = asyncio.run(
        dense_retrieve(
            query="karma",
            k=1,
            worldview=None,
            book_types=["ga_text"],
            collection="rudolf-steiner-ga",
            embedding_client=Emb(),  # type: ignore[arg-type]
            qdrant_client=qdrant,  # type: ignore[arg-type]
            query_vector=[0.1, 0.2],
        )
    )
    assert qdrant.vectors == [[0.1, 0.2]]
    assert len(hits) == 1
