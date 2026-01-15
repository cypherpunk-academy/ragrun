"""Integration tests for the SQLAlchemy-based mirror repository."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select

from app.db.tables import chunks_table, metadata
from app.services.mirror_repository import ChunkMirrorRepository
from ragrun.models import ChunkRecord


def _chunk_payload(chunk_id: str) -> dict[str, object]:
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
            "worldviews": ["Idealismus"],
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


@pytest.mark.asyncio
async def test_repository_upsert_and_delete(tmp_path: Path):
    db_path = tmp_path / "mirror.db"
    engine = create_engine(f"sqlite:///{db_path}", future=True)
    metadata.create_all(engine)

    repo = ChunkMirrorRepository(engine)
    chunk = ChunkRecord.from_dict(_chunk_payload("chunk-1"))

    await repo.upsert_chunks("books", [chunk])

    with engine.connect() as connection:
        rows = connection.execute(select(chunks_table)).all()
        assert len(rows) == 1
        assert rows[0].chunk_id == "chunk-1"

    await repo.delete_chunks("books", ["chunk-1"])

    with engine.connect() as connection:
        remaining = connection.execute(select(chunks_table)).all()
        assert remaining == []

