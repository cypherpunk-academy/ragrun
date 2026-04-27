"""API tests for ragprep-compatible RAG endpoints."""
from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from app.shared.models import ChunkRecord

import pytest
from fastapi.testclient import TestClient

from app.api import rag as rag_router
from app.main import app
from app.services.ingestion_service import DeleteResult, UploadResult


def _sample_chunk_jsonl(chunk_id: str, content_hash: str) -> str:
    """Generate a single JSONL line for testing."""
    now = datetime.utcnow().isoformat()
    chunk = {
        "id": chunk_id,
        "text": f"Sample text for {chunk_id}",
        "metadata": {
            "chunk_id": chunk_id,
            "source_id": "test-source",
            "content_hash": content_hash,
            "chunk_type": "book",
            "language": "en",
            "created_at": now,
            "updated_at": now,
        },
    }
    return json.dumps(chunk)


class StubIngestionService:
    """Stub ingestion service for testing."""

    def __init__(self) -> None:
        self.upload_calls = []
        self.delete_calls = []

    async def upload_chunks(self, **kwargs):
        self.upload_calls.append(kwargs)
        requested = len(kwargs["chunks"])
        return UploadResult(
            ingestion_id="ing-123",
            collection=kwargs["collection"],
            requested=requested,
            ingested=requested,
            duplicates=0,
            embedding_model=kwargs.get("embedding_model") or "default-model",
            vector_size=768,
            unchanged=0,
            changed=requested,
            new=0,
            stale_deleted=0,
        )

    async def delete_chunks(self, **kwargs):
        self.delete_calls.append(kwargs)
        return DeleteResult(
            collection=kwargs["collection"],
            requested=len(kwargs["chunk_ids"]),
            deleted=len(kwargs["chunk_ids"]),
        )


class StubRagChunksRepository:
    def __init__(self) -> None:
        self.upsert_calls: list[dict] = []
        self.list_records: list[ChunkRecord] = []
        self.mark_embedded_calls: list[tuple[str, list[str]]] = []
        self.last_embed_query: tuple[str, object] | None = None
        self.deprecate_orphans_calls: list[tuple[str, dict[str, list[str]]]] = []

    async def upsert_chunks(self, rag_partition: str, chunks, *, default_scope=None):
        self.upsert_calls.append(
            {"rag_partition": rag_partition, "chunks": chunks, "default_scope": default_scope}
        )

    async def deprecate_orphans_for_sources(
        self, rag_partition: str, active_by_source: dict[str, list[str]]
    ) -> dict[str, int]:
        self.deprecate_orphans_calls.append((rag_partition, dict(active_by_source)))
        return {sid: 0 for sid in active_by_source}

    async def list_chunk_records_for_embed(
        self, assistant_rag_collection: str, *, shared_source_ids=None
    ):
        self.last_embed_query = (assistant_rag_collection, shared_source_ids)
        return list(self.list_records)

    async def mark_embedded_for_embed_run(self, assistant_rag_collection: str, chunk_ids):
        self.mark_embedded_calls.append((assistant_rag_collection, list(chunk_ids)))

    async def delete_chunks(self, collection: str, chunk_ids):
        return None


@pytest.fixture
def client_with_stub(monkeypatch):
    """Test client with overridden ingestion service."""
    stub = StubIngestionService()
    rag_stub = StubRagChunksRepository()

    # Mock get_engine to avoid DB connections
    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=None)
    mock_result = MagicMock()
    mock_result.fetchall = MagicMock(return_value=[])
    mock_result.scalar = MagicMock(return_value=0)
    mock_conn.execute = MagicMock(return_value=mock_result)
    mock_engine.connect = MagicMock(return_value=mock_conn)

    def mock_get_engine():
        return mock_engine

    # Patch at the module level where it's imported
    monkeypatch.setattr("app.api.rag.get_engine", mock_get_engine)
    monkeypatch.setattr("app.api.rag.get_rag_chunks_repository", lambda: rag_stub)

    app.dependency_overrides[rag_router.get_ingestion_service] = lambda: stub
    client = TestClient(app)

    yield client, stub, rag_stub

    # Cleanup
    app.dependency_overrides.pop(rag_router.get_ingestion_service, None)


def test_store_endpoint_persists_chunks(client_with_stub):
    """Verify store-chunks accepts JSONL and upserts rag_chunks."""
    client, _stub, rag_stub = client_with_stub

    lines = [
        _sample_chunk_jsonl("test-001", "hash1"),
        _sample_chunk_jsonl("test-002", "hash2"),
    ]
    jsonl_content = "\n".join(lines)

    payload = {
        "chunks_jsonl_content": jsonl_content,
        "collection_name": "test-collection",
    }

    response = client.post("/api/v1/rag/store-chunks", json=payload)

    assert response.status_code == 202
    body = response.json()
    assert body["collection"] == "test-collection"
    assert body["stored"] == 2
    assert body["deprecated"] == 0
    assert body["deprecated_by_source"] == {"test-source": 0}
    assert rag_stub.upsert_calls[0]["rag_partition"] == "test-collection"
    assert len(rag_stub.upsert_calls[0]["chunks"]) == 2
    assert rag_stub.deprecate_orphans_calls[0][0] == "test-collection"
    assert rag_stub.deprecate_orphans_calls[0][1] == {
        "test-source": ["test-001", "test-002"],
    }


def test_embed_endpoint_runs_ingestion(client_with_stub):
    """Verify embed-chunks loads rag_chunks and calls ingestion + mark_embedded."""
    client, stub, rag_stub = client_with_stub
    rag_stub.list_records = [
        ChunkRecord.from_dict(json.loads(_sample_chunk_jsonl("test-001", "hash1"))),
    ]

    payload = {
        "collection_name": "test-collection",
        "skip_cleanup": True,
    }

    response = client.post("/api/v1/rag/embed-chunks", json=payload)
    assert response.status_code == 202
    assert stub.upload_calls[0]["collection"] == "test-collection"
    assert stub.upload_calls[0].get("skip_cleanup") is True
    assert rag_stub.mark_embedded_calls[0][0] == "test-collection"
    assert rag_stub.mark_embedded_calls[0][1] == ["test-001"]
    assert rag_stub.last_embed_query == ("test-collection", None)


def test_embed_endpoint_passes_shared_source_ids_whitelist(client_with_stub):
    """embed-chunks forwards shared_source_ids to the repository."""
    client, stub, rag_stub = client_with_stub
    rag_stub.list_records = [
        ChunkRecord.from_dict(json.loads(_sample_chunk_jsonl("test-001", "hash1"))),
    ]

    payload = {
        "collection_name": "test-collection",
        "skip_cleanup": True,
        "shared_source_ids": ["book-a", "lecture:xyz"],
    }

    response = client.post("/api/v1/rag/embed-chunks", json=payload)
    assert response.status_code == 202
    assert rag_stub.last_embed_query == ("test-collection", ["book-a", "lecture:xyz"])


def test_store_endpoint_validates_jsonl(client_with_stub):
    """Verify store-chunks rejects malformed JSONL."""
    client, _, _ = client_with_stub

    payload = {
        "chunks_jsonl_content": "not valid json\n{also bad",
        "collection_name": "test-collection",
    }

    response = client.post("/api/v1/rag/store-chunks", json=payload)
    assert response.status_code == 400
    assert "Invalid JSONL" in response.json()["detail"]


def test_delete_endpoint_requires_filter_or_all(client_with_stub):
    """Verify delete-chunks requires either filter or all=true."""
    client, _, _ = client_with_stub
    
    payload = {
        "collection_name": "test-collection",
        # Missing both 'all' and 'filter'
    }

    response = client.post("/api/v1/rag/delete-chunks", json=payload)
    assert response.status_code == 400
    assert "all=true" in response.json()["detail"] or "filter" in response.json()["detail"]


def test_delete_endpoint_dry_run(client_with_stub):
    """Verify delete-chunks dry_run returns matched count."""
    client, _, _ = client_with_stub
    
    payload = {
        "collection_name": "test-collection",
        "filter": {"book_id": "test-book"},
        "dry_run": True,
    }

    response = client.post("/api/v1/rag/delete-chunks", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["dry_run"] is True
    assert "matched" in data
    assert data["deleted"] == 0


def test_list_chunks_returns_inventory(monkeypatch):
    """Verify list-chunks inventories from Qdrant scroll (independent of mirror)."""

    class FakeQdrant:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        async def scroll_points_page(
            self,
            _collection: str,
            *,
            filter_,
            limit: int,
            offset,
            with_payload: bool,
            with_vectors: bool,
        ):
            assert filter_ == {"must": [{"key": "source_id", "match": {"value": "test-source"}}]}
            assert with_payload is True
            assert with_vectors is False
            assert limit >= 1
            # Single page
            return (
                [
                    {
                        "id": "ignored",
                        "payload": {
                            "chunk_id": "c1",
                            "content_hash": "h1",
                            "updated_at": "2025-01-01T00:00:00Z",
                            "chunk_type": "book",
                            "source_id": "test-source",
                        },
                    }
                ],
                None,
            )

    monkeypatch.setattr("app.api.rag.QdrantClient", FakeQdrant)

    client = TestClient(app)
    res = client.post(
        "/api/v1/rag/list-chunks",
        json={"collection_name": "test-collection", "source_id": "test-source", "limit": 10},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["collection"] == "test-collection"
    assert body["source_id"] == "test-source"
    assert body["chunks"][0]["chunk_id"] == "c1"
    assert body["chunks"][0]["content_hash"] == "h1"
    assert body["chunks"][0]["chunk_type"] == "book"


def test_delete_chunk_ids_dry_run_and_limit(client_with_stub):
    """Verify delete-chunk-ids enforces limit and supports dry_run."""
    client, stub, _ = client_with_stub

    # limit enforcement happens even on dry_run
    res = client.post(
        "/api/v1/rag/delete-chunk-ids",
        json={
            "collection_name": "test-collection",
            "chunk_ids": ["a", "b"],
            "dry_run": True,
            "limit": 1,
        },
    )
    assert res.status_code == 400

    res2 = client.post(
        "/api/v1/rag/delete-chunk-ids",
        json={
            "collection_name": "test-collection",
            "chunk_ids": ["a", "b"],
            "dry_run": True,
            "limit": 3,
        },
    )
    assert res2.status_code == 200
    body = res2.json()
    assert body["dry_run"] is True
    assert body["matched"] == 2
    assert body["deleted"] == 0
    assert stub.delete_calls == []
