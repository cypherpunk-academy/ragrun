"""Upsert keeps DB references when text/meaning unchanged; clears on rewrite."""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

from app.config import normalize_postgres_dsn, settings
from app.ingestion.repositories.rag_chunks_repository import (
    _metadata_meaning,
    RagChunksRepository,
)
from app.db.tables import rag_chunks_table
from app.shared.models import ChunkRecord


def test_metadata_meaning_drops_volatile_keys():
    from sqlalchemy.dialects import postgresql

    expr = _metadata_meaning(rag_chunks_table.c.metadata)
    compiled = expr.compile(dialect=postgresql.dialect())
    sql = str(compiled)
    assert "rag_chunks.metadata -" in sql
    assert "TEXT[]" in sql
    # Bind holds the ignored key list (not inlined into SQL text).
    params = list(compiled.params.values())
    assert params and set(params[0]) >= {
        "created_at",
        "updated_at",
        "source_index",
        "references",
        "content_length",
    }


def _try_engine():
    dsn = normalize_postgres_dsn(settings.postgres_dsn)
    engine = create_engine(dsn, pool_pre_ping=True)
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except OperationalError as exc:
        pytest.skip(f"Postgres unreachable ({dsn}): {exc}")
    return engine


def _meta(
    *,
    chunk_id: str,
    content_hash: str,
    text: str,
    segment_title: str,
    source_index: int,
    updated_at: datetime,
    references: list | None = None,
) -> dict:
    payload = {
        "source_id": "test-upsert-refs-source",
        "chunk_id": chunk_id,
        "chunk_type": "typology",
        "language": "de",
        "importance": 5,
        "content_hash": content_hash,
        "created_at": updated_at.isoformat(),
        "updated_at": updated_at.isoformat(),
        "source_index": source_index,
        "source_type": "typology",
        "segment_title": segment_title,
        "typology_slug": "test-upsert-refs",
        "typology_role": "member",
        "text": text,
    }
    if references is not None:
        payload["references"] = references
    return payload


@pytest.mark.asyncio
async def test_upsert_preserves_and_clears_references():
    engine = _try_engine()
    partition = f"__test_upsert_refs_{uuid4().hex[:8]}"
    chunk_id = str(uuid4())
    repo = RagChunksRepository(engine)
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    t1 = datetime(2024, 6, 1, tzinfo=timezone.utc)
    refs = [{"chunk_id": "target-1", "relation": "erlaeutert", "relevance": 0.9}]

    try:
        # Seed with references (augment-style payload).
        await repo.upsert_chunks(
            partition,
            [
                ChunkRecord.from_dict(
                    {
                        "text": "aspect body",
                        "metadata": _meta(
                            chunk_id=chunk_id,
                            content_hash="hash-a",
                            text="aspect body",
                            segment_title="Entfaltung",
                            source_index=0,
                            updated_at=t0,
                            references=refs,
                        ),
                    }
                )
            ],
        )

        with engine.connect() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT "references" AS refs, embedded_at
                    FROM rag_chunks
                    WHERE rag_partition = :p AND chunk_id = :c
                    """
                ),
                {"p": partition, "c": chunk_id},
            ).mappings().one()
        assert row["refs"] == refs

        # Same text+meaning, new timestamps/source_index, no references → keep refs.
        await repo.upsert_chunks(
            partition,
            [
                ChunkRecord.from_dict(
                    {
                        "text": "aspect body",
                        "metadata": _meta(
                            chunk_id=chunk_id,
                            content_hash="hash-a",
                            text="aspect body",
                            segment_title="Entfaltung",
                            source_index=99,
                            updated_at=t1,
                        ),
                    }
                )
            ],
        )
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT "references" AS refs, updated_at
                    FROM rag_chunks
                    WHERE rag_partition = :p AND chunk_id = :c
                    """
                ),
                {"p": partition, "c": chunk_id},
            ).mappings().one()
        assert row["refs"] == refs
        assert row["updated_at"] == t0

        # Text change without references → clear refs.
        await repo.upsert_chunks(
            partition,
            [
                ChunkRecord.from_dict(
                    {
                        "text": "aspect body rewritten",
                        "metadata": _meta(
                            chunk_id=chunk_id,
                            content_hash="hash-b",
                            text="aspect body rewritten",
                            segment_title="Entfaltung",
                            source_index=99,
                            updated_at=t1,
                        ),
                    }
                )
            ],
        )
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT "references" AS refs, text, content_hash
                    FROM rag_chunks
                    WHERE rag_partition = :p AND chunk_id = :c
                    """
                ),
                {"p": partition, "c": chunk_id},
            ).mappings().one()
        assert row["refs"] is None
        assert row["text"] == "aspect body rewritten"
        assert row["content_hash"] == "hash-b"

        # Explicit references payload writes even when text unchanged.
        await repo.upsert_chunks(
            partition,
            [
                ChunkRecord.from_dict(
                    {
                        "text": "aspect body rewritten",
                        "metadata": _meta(
                            chunk_id=chunk_id,
                            content_hash="hash-b",
                            text="aspect body rewritten",
                            segment_title="Entfaltung",
                            source_index=1,
                            updated_at=t1,
                            references=refs,
                        ),
                    }
                )
            ],
        )
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT "references" AS refs
                    FROM rag_chunks
                    WHERE rag_partition = :p AND chunk_id = :c
                    """
                ),
                {"p": partition, "c": chunk_id},
            ).mappings().one()
        assert row["refs"] == refs
    finally:
        with engine.begin() as conn:
            conn.execute(
                text("DELETE FROM rag_chunks WHERE rag_partition = :p"),
                {"p": partition},
            )
        engine.dispose()
