"""Tests für chunk_references_service (Referenzen nachträglich ermitteln)."""
from __future__ import annotations

import pytest

from app.retrieval.models import RetrievedSnippet
from app.retrieval.services import chunk_references_service as svc


def _snippet(chunk_id: str, score: float = 0.9) -> RetrievedSnippet:
    return RetrievedSnippet(
        text=f"Text {chunk_id}",
        score=score,
        payload={"payload": {"chunk_id": chunk_id}},
    )


@pytest.fixture(autouse=True)
def _no_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(svc, "load_assistant_embedding_prefixes", lambda _a: ("passage: ", "query: "))


@pytest.mark.asyncio
async def test_references_use_books_and_lectures_and_skip_excluded(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    async def fake_dense_retrieve(**kwargs):
        calls.append(list(kwargs["book_types"]))
        if kwargs["book_types"] == ["primary"]:
            return [_snippet("book-1"), _snippet("self")]
        return [_snippet("talk-1"), _snippet("book-1")]  # Duplikat

    seen: dict[str, list[str]] = {}

    async def fake_eval(*, generated_text, retrieved_chunks, llm, max_chunks):
        seen["ids"] = [s.payload["payload"]["chunk_id"] for s in retrieved_chunks]
        return [{"chunk_id": "book-1", "description": "x", "relevance": 0.8}]

    monkeypatch.setattr(svc, "dense_retrieve", fake_dense_retrieve)
    monkeypatch.setattr(svc, "evaluate_chunk_relevance", fake_eval)

    refs = await svc.compute_chunk_references(
        text="Eine Begriffserklärung.",
        collection="c",
        assistant="a",
        embedding_client=object(),  # type: ignore[arg-type]
        qdrant_client=object(),  # type: ignore[arg-type]
        chat_client=object(),  # type: ignore[arg-type]
        exclude_chunk_ids=["self"],
    )

    assert calls == [["primary"], ["talk", "talk_summary"]]
    assert seen["ids"] == ["book-1", "talk-1"]
    assert refs == [{"chunk_id": "book-1", "description": "x", "relevance": 0.8}]


@pytest.mark.asyncio
async def test_no_hits_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_dense_retrieve(**_kwargs):
        return []

    monkeypatch.setattr(svc, "dense_retrieve", fake_dense_retrieve)

    refs = await svc.compute_chunk_references(
        text="Text",
        collection="c",
        assistant="a",
        embedding_client=object(),  # type: ignore[arg-type]
        qdrant_client=object(),  # type: ignore[arg-type]
        chat_client=object(),  # type: ignore[arg-type]
    )
    assert refs == []


@pytest.mark.asyncio
async def test_empty_text_raises() -> None:
    with pytest.raises(ValueError):
        await svc.compute_chunk_references(
            text="   ",
            collection="c",
            assistant="a",
            embedding_client=object(),  # type: ignore[arg-type]
            qdrant_client=object(),  # type: ignore[arg-type]
            chat_client=object(),  # type: ignore[arg-type]
        )
