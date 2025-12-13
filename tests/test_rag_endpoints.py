"""API tests for ragprep-compatible RAG endpoints."""
from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

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


@pytest.fixture
def client_with_stub(monkeypatch):
    """Test client with overridden ingestion service."""
    stub = StubIngestionService()
    
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
    
    app.dependency_overrides[rag_router.get_ingestion_service] = lambda: stub
    client = TestClient(app)
    
    yield client, stub
    
    # Cleanup
    app.dependency_overrides.pop(rag_router.get_ingestion_service, None)


def test_upload_endpoint_returns_stats(client_with_stub):
    """Verify upload-chunks endpoint accepts JSONL and returns ingestion stats."""
    client, stub = client_with_stub
    
    # Build JSONL content with 2 chunks
    lines = [
        _sample_chunk_jsonl("test-001", "hash1"),
        _sample_chunk_jsonl("test-002", "hash2"),
    ]
    jsonl_content = "\n".join(lines)

    payload = {
        "chunks_jsonl_content": jsonl_content,
        "collection_name": "test-collection",
    }

    response = client.post("/api/v1/rag/upload-chunks", json=payload)
    
    assert response.status_code == 202
    body = response.json()
    assert body["collection"] == "test-collection"
    assert body["ingested"] == 2
    assert stub.upload_calls[0]["collection"] == "test-collection"

def test_upload_endpoint_passes_skip_cleanup(client_with_stub):
    """Verify upload-chunks forwards skip_cleanup to the ingestion service."""
    client, stub = client_with_stub

    payload = {
        "chunks_jsonl_content": _sample_chunk_jsonl("test-001", "hash1"),
        "collection_name": "test-collection",
        "skip_cleanup": True,
    }

    response = client.post("/api/v1/rag/upload-chunks", json=payload)
    assert response.status_code == 202
    assert stub.upload_calls[0].get("skip_cleanup") is True


def test_upload_endpoint_validates_jsonl(client_with_stub):
    """Verify upload-chunks rejects malformed JSONL."""
    client, _ = client_with_stub
    
    payload = {
        "chunks_jsonl_content": "not valid json\n{also bad",
        "collection_name": "test-collection",
    }

    response = client.post("/api/v1/rag/upload-chunks", json=payload)
    assert response.status_code == 400
    assert "Invalid JSONL" in response.json()["detail"]


def test_delete_endpoint_requires_filter_or_all(client_with_stub):
    """Verify delete-chunks requires either filter or all=true."""
    client, _ = client_with_stub
    
    payload = {
        "collection_name": "test-collection",
        # Missing both 'all' and 'filter'
    }

    response = client.post("/api/v1/rag/delete-chunks", json=payload)
    assert response.status_code == 400
    assert "all=true" in response.json()["detail"] or "filter" in response.json()["detail"]


def test_delete_endpoint_dry_run(client_with_stub):
    """Verify delete-chunks dry_run returns matched count."""
    client, _ = client_with_stub
    
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
    client, stub = client_with_stub

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
