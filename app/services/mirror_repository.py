"""Postgres mirror for chunk metadata."""
from __future__ import annotations

import asyncio
from typing import Iterable, List

from sqlalchemy import delete, insert
from sqlalchemy.engine import Engine

from ragrun.models import ChunkRecord

from app.db.tables import chunks_table


class ChunkMirrorRepository:
    """Persists chunk metadata into the relational mirror (Postgres by default)."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    async def upsert_chunks(self, collection: str, chunks: Iterable[ChunkRecord]) -> None:
        """Insert or update the provided chunks for the collection."""

        chunk_list = list(chunks)
        if not chunk_list:
            return

        rows: List[dict] = []
        for chunk in chunk_list:
            metadata = chunk.metadata
            rows.append(
                {
                    "collection": collection,
                    "chunk_id": metadata.chunk_id,
                    "source_id": metadata.source_id,
                    "chunk_type": metadata.chunk_type,
                    "language": metadata.language,
                    "worldview": metadata.worldview,
                    "importance": metadata.importance,
                    "content_hash": metadata.content_hash,
                    "text": chunk.text,
                    "created_at": metadata.created_at,
                    "updated_at": metadata.updated_at,
                    "metadata": metadata.model_dump(mode="json"),
                }
            )

        chunk_ids = [row["chunk_id"] for row in rows]

        def _write() -> None:
            with self.engine.begin() as connection:
                connection.execute(
                    delete(chunks_table).where(
                        chunks_table.c.collection == collection,
                        chunks_table.c.chunk_id.in_(chunk_ids),
                    )
                )
                connection.execute(insert(chunks_table), rows)

        await asyncio.to_thread(_write)

    async def delete_chunks(self, collection: str, chunk_ids: Iterable[str]) -> None:
        """Delete mirrored chunks (noop when list is empty)."""

        ids = list(chunk_ids)
        if not ids:
            return

        def _delete() -> None:
            with self.engine.begin() as connection:
                connection.execute(
                    delete(chunks_table).where(
                        chunks_table.c.collection == collection,
                        chunks_table.c.chunk_id.in_(ids),
                    )
                )

        await asyncio.to_thread(_delete)


