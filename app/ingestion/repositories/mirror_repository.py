"""Postgres mirror for Qdrant chunk payloads (vector_chunks table)."""
from __future__ import annotations

import asyncio
from typing import Iterable, List

from sqlalchemy import delete, insert, select
from sqlalchemy.engine import Engine

from app.shared.models import ChunkRecord

from app.db.tables import vector_chunks_table


class VectorChunksRepository:
    """Persists chunk metadata into vector_chunks (relational mirror of Qdrant)."""

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
                    "worldviews": metadata.worldviews or [],
                    "importance": metadata.importance,
                    "content_hash": metadata.content_hash,
                    "text": chunk.text,
                    "created_at": metadata.created_at,
                    "updated_at": metadata.updated_at,
                    "metadata": metadata.model_dump(mode="json"),
                    "references": metadata.references,
                }
            )

        # Deduplicate by (collection, chunk_id), keeping the row with latest updated_at.
        seen: dict[str, dict] = {}
        for row in rows:
            cid = row["chunk_id"]
            existing = seen.get(cid)
            if existing is None:
                seen[cid] = row
            else:
                u_new = row.get("updated_at")
                u_old = existing.get("updated_at")
                if u_new is not None and (u_old is None or u_new > u_old):
                    seen[cid] = row
        rows = list(seen.values())

        chunk_ids = [row["chunk_id"] for row in rows]

        def _write() -> None:
            with self.engine.begin() as connection:
                connection.execute(
                    delete(vector_chunks_table).where(
                        vector_chunks_table.c.collection == collection,
                        vector_chunks_table.c.chunk_id.in_(chunk_ids),
                    )
                )
                connection.execute(insert(vector_chunks_table), rows)

        await asyncio.to_thread(_write)

    async def delete_chunks(self, collection: str, chunk_ids: Iterable[str]) -> None:
        """Delete mirrored chunks (noop when list is empty)."""

        ids = list(chunk_ids)
        if not ids:
            return

        def _delete() -> None:
            with self.engine.begin() as connection:
                connection.execute(
                    delete(vector_chunks_table).where(
                        vector_chunks_table.c.collection == collection,
                        vector_chunks_table.c.chunk_id.in_(ids),
                    )
                )

        await asyncio.to_thread(_delete)

    async def list_chunk_ids_by_source(self, collection: str, source_id: str) -> List[str]:
        """Return chunk_ids for a given source_id in a collection."""

        if not source_id:
            return []

        def _select() -> List[str]:
            with self.engine.begin() as connection:
                rows = connection.execute(
                    select(vector_chunks_table.c.chunk_id).where(
                        vector_chunks_table.c.collection == collection,
                        vector_chunks_table.c.source_id == source_id,
                    )
                ).fetchall()
                return [r[0] for r in rows]

        return await asyncio.to_thread(_select)


# Backward-compatible alias
ChunkMirrorRepository = VectorChunksRepository
