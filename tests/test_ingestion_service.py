"""Unit tests for the ingestion service."""
from __future__ import annotations

from datetime import datetime
from typing import Iterable, List, Sequence
from uuid import uuid5, NAMESPACE_DNS

import pytest

from app.services.embedding_client import EmbeddingBatchResult
from app.services.ingestion_service import IngestionService
from app.services.mirror_repository import ChunkMirrorRepository
from app.shared.models import ChunkRecord


def _chunk_payload(chunk_id: str, content_hash: str) -> dict[str, object]:
    now = datetime.utcnow().isoformat()
    return {
        "id": chunk_id,
        "text": f"text for {chunk_id}",
        "metadata": {
            "author": "Author",
            "source_id": "source-1",
            "source_title": "Title",
            "source_index": 1,
            "segment_id": "seg",
            "segment_title": "Segment",
            "segment_index": 0,
            "parent_id": None,
            "chunk_id": chunk_id,
            "chunk_type": "book",
            "worldviews": ["Idealismus"],
            "importance": 5,
            "text": f"text for {chunk_id}",
            "content_hash": content_hash,
            "created_at": now,
            "updated_at": now,
            "source_type": "book",
            "language": "de",
            "tags": ["tag"],
        },
    }


def _chunk_record(chunk_id: str, content_hash: str) -> ChunkRecord:
    return ChunkRecord.from_dict(_chunk_payload(chunk_id, content_hash))


class FakeEmbeddingClient:
    def __init__(self, dimension: int = 4) -> None:
        self.dimension = dimension
        self.calls: list[dict[str, object]] = []

    async def embed_texts(
        self,
        texts: Sequence[str],
        *,
        model_name: str | None = None,
        batch_size: int | None = None,
    ) -> EmbeddingBatchResult:
        self.calls.append({"texts": list(texts), "model": model_name, "batch_size": batch_size})
        embeddings = [
            [float(idx + 1) for _ in range(self.dimension)] for idx, _ in enumerate(texts)
        ]
        return EmbeddingBatchResult(
            embeddings=embeddings,
            dimensions=self.dimension,
            model_name=model_name or "default-model",
        )


class FakeQdrantClient:
    def __init__(self) -> None:
        self.ensure_calls: list[dict[str, object]] = []
        self.upserts: list[dict[str, object]] = []
        self.deletes: list[dict[str, object]] = []
        self.retrieves: list[dict[str, object]] = []
        self.payload_updates: list[dict[str, object]] = []
        self.scrolls: list[dict[str, object]] = []

    async def ensure_collection(self, name: str, *, vector_size: int) -> None:
        self.ensure_calls.append({"name": name, "vector_size": vector_size})

    async def upsert_points(
        self,
        collection: str,
        points: Iterable[dict[str, object]],
        *,
        wait: bool = True,
    ) -> None:
        self.upserts.append({"collection": collection, "points": list(points), "wait": wait})

    async def delete_points(
        self,
        collection: str,
        point_ids: Iterable[str],
        *,
        wait: bool = True,
    ) -> None:
        self.deletes.append({"collection": collection, "ids": list(point_ids), "wait": wait})

    async def retrieve_points(
        self,
        collection: str,
        point_ids: Sequence[str],
        *,
        with_vectors: bool = False,
        with_payload: bool = True,
    ) -> List[dict[str, object]]:
        self.retrieves.append(
            {
                "collection": collection,
                "ids": list(point_ids),
                "with_vectors": with_vectors,
                "with_payload": with_payload,
            }
        )
        # Default: nothing exists yet
        return []

    async def set_payload(
        self,
        collection: str,
        updates: Sequence[dict[str, object]],
        *,
        wait: bool = True,
    ) -> None:
        self.payload_updates.append({"collection": collection, "updates": list(updates), "wait": wait})

    async def scroll_all_points(
        self,
        collection: str,
        *,
        filter_: dict[str, object] | None = None,
        limit: int = 256,
        with_payload: bool = True,
        with_vectors: bool = False,
        max_pages: int = 10_000,
    ) -> List[dict[str, object]]:
        self.scrolls.append(
            {
                "collection": collection,
                "filter": filter_,
                "limit": limit,
                "with_payload": with_payload,
                "with_vectors": with_vectors,
                "max_pages": max_pages,
            }
        )
        # Default: no existing points for cleanup
        return []


class FakeMirror(ChunkMirrorRepository):
    def __init__(self) -> None:
        self.upserts: list[dict[str, object]] = []
        self.deletes: list[dict[str, object]] = []

    async def upsert_chunks(self, collection: str, chunks: Iterable[ChunkRecord]) -> None:
        self.upserts.append({"collection": collection, "chunks": list(chunks)})

    async def delete_chunks(self, collection: str, chunk_ids: Iterable[str]) -> None:
        self.deletes.append({"collection": collection, "chunk_ids": list(chunk_ids)})


class FakeTelemetry:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def record_ingestion_run(self, **payload: object) -> None:
        self.calls.append(payload)


def _service(
    dimension: int = 8,
) -> tuple[IngestionService, FakeEmbeddingClient, FakeQdrantClient, FakeMirror, FakeTelemetry]:
    embedding_client = FakeEmbeddingClient(dimension)
    qdrant_client = FakeQdrantClient()
    mirror = FakeMirror()
    telemetry = FakeTelemetry()
    service = IngestionService(
        embedding_client=embedding_client,
        qdrant_client=qdrant_client,
        mirror_repository=mirror,
        telemetry_client=telemetry,
        default_batch_size=2,
    )
    return service, embedding_client, qdrant_client, mirror, telemetry


@pytest.mark.asyncio
async def test_upload_chunks_dedupes_and_upserts():
    service, embedding_client, qdrant_client, mirror, telemetry = _service()
    chunk_a = _chunk_record("chunk-1", "hash-1")
    chunk_dup = _chunk_record("chunk-1", "hash-1")
    chunk_b = _chunk_record("chunk-2", "hash-2")

    result = await service.upload_chunks(
        collection="books",
        chunks=[chunk_a, chunk_dup, chunk_b],
        embedding_model="custom-model",
        batch_size=2,
    )

    assert result.ingested == 2
    assert result.requested == 3
    assert result.duplicates == 1
    assert result.embedding_model == "custom-model"
    assert result.vector_size == embedding_client.dimension

    assert qdrant_client.ensure_calls == [{"name": "books", "vector_size": embedding_client.dimension}]
    assert len(qdrant_client.upserts) == 1
    assert len(qdrant_client.upserts[0]["points"]) == 2
    assert mirror.upserts and len(mirror.upserts[0]["chunks"]) == 2
    assert telemetry.calls and telemetry.calls[0]["collection"] == "books"


@pytest.mark.asyncio
async def test_upload_raises_when_no_chunks_provided():
    service, *_ = _service()

    with pytest.raises(ValueError):
        await service.upload_chunks(collection="books", chunks=[])


@pytest.mark.asyncio
async def test_delete_chunks_invokes_backends():
    service, _, qdrant_client, mirror, _ = _service()

    chunk_ids = ["a", "b", "c"]
    result = await service.delete_chunks(
        collection="books",
        chunk_ids=chunk_ids,
    )

    # Qdrant receives UUIDs (converted from chunk_ids)
    expected_uuids = [str(uuid5(NAMESPACE_DNS, cid)) for cid in chunk_ids]

    assert result.deleted == 3
    assert qdrant_client.deletes[0]["ids"] == expected_uuids
    assert mirror.deletes[0]["chunk_ids"] == chunk_ids


def test_chunk_record_rejects_legacy_worldview_field():
    payload = _chunk_payload("legacy-worldview", "hash-legacy")
    payload["metadata"].pop("worldviews", None)
    payload["metadata"]["worldview"] = "Idealismus"

    with pytest.raises(ValueError):
        ChunkRecord.from_dict(payload)


def test_chunk_record_requires_worldviews_list_of_strings():
    payload = _chunk_payload("bad-worldviews", "hash-bad")
    payload["metadata"]["worldviews"] = "Idealismus"  # not a list

    with pytest.raises(ValueError):
        ChunkRecord.from_dict(payload)
