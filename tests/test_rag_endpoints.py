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
        return UploadResult(
            ingestion_id="ing-123",
            collection=kwargs["collection"],
            requested=len(kwargs["chunks"]),
            ingested=len(kwargs["chunks"]),
            duplicates=0,
            embedding_model=kwargs.get("embedding_model") or "default-model",
            vector_size=768,
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
