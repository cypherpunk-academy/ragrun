"""API tests for the /rag endpoints."""
from __future__ import annotations

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from app.api import rag as rag_router
from app.main import app
from app.services.ingestion_service import DeleteResult, UploadResult


class StubIngestionService:
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


def _chunk_payload(chunk_id: str) -> dict:
    now = datetime.utcnow().isoformat()
    return {
        "id": chunk_id,
        "text": f"text {chunk_id}",
        "metadata": {
            "author": "Author",
            "source_id": "source",
            "source_title": "Title",
            "source_index": 0,
            "segment_id": "seg",
            "segment_title": "Segment",
            "segment_index": 0,
            "parent_id": None,
            "chunk_id": chunk_id,
            "chunk_type": "book",
            "worldview": "Idealismus",
            "importance": 5,
            "text": f"text {chunk_id}",
            "content_hash": f"hash-{chunk_id}",
            "created_at": now,
            "updated_at": now,
            "source_type": "book",
            "language": "de",
            "tags": ["tag"],
        },
    }


@pytest.fixture
def client_with_stub():
    stub = StubIngestionService()
    app.dependency_overrides[rag_router.get_ingestion_service] = lambda: stub
    client = TestClient(app)
    yield client, stub
    app.dependency_overrides.pop(rag_router.get_ingestion_service, None)


def test_upload_endpoint_returns_stats(client_with_stub):
    client, stub = client_with_stub
    payload = {
        "collection": "books",
        "embedding_model": "custom",
        "chunks": [_chunk_payload("chunk-1")],
    }

    response = client.post("/rag/upload", json=payload)

    assert response.status_code == 202
    body = response.json()
    assert body["collection"] == "books"
    assert body["ingested"] == 1
    assert body["embedding_model"] == "custom"
    assert stub.upload_calls and stub.upload_calls[0]["collection"] == "books"


def test_delete_endpoint_deletes_chunks(client_with_stub):
    client, stub = client_with_stub

    response = client.request(
        "DELETE",
        "/rag/delete",
        json={"collection": "books", "chunk_ids": ["a", "b"], "cascade": False},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["deleted"] == 2
    assert stub.delete_calls and stub.delete_calls[0]["chunk_ids"] == ["a", "b"]

