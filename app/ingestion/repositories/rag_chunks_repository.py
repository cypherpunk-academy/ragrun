"""Primary chunk store (rag_chunks table, DB-first)."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Mapping, Sequence

from sqlalchemy import and_, case, delete, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Engine

from app.db.tables import rag_chunks_table
from app.shared.models import ChunkRecord
from app.shared.rag_partition import RAG_PARTITION_SHARED

# One INSERT … ON CONFLICT per batch (not per row) — critical for remote Postgres/Supabase.
_UPSERT_BATCH_SIZE = 200


def _scope_for_chunk(chunk: ChunkRecord, default_scope: str | None) -> str | None:
    st = chunk.metadata.source_type
    if isinstance(st, str) and st.strip():
        return st.strip()
    if default_scope and default_scope.strip():
        return default_scope.strip()
    return None


def _row_to_chunk_record(row: Mapping[str, object]) -> ChunkRecord:
    meta = row["metadata"]
    if not isinstance(meta, dict):
        raise ValueError("rag_chunks.metadata must be a JSON object")
    return ChunkRecord.from_dict({"text": str(row.get("text") or ""), "metadata": meta})


class RagChunksRepository:
    """CRUD for rag_chunks (source of truth before embedding)."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    async def upsert_chunks(
        self,
        rag_partition: str,
        chunks: Iterable[ChunkRecord],
        *,
        default_scope: str | None = None,
    ) -> None:
        """Insert or update chunks. Resets embedded_at when content_hash changes."""

        chunk_list = list(chunks)
        if not chunk_list:
            return

        rows: List[dict] = []
        for chunk in chunk_list:
            metadata = chunk.metadata
            scope = _scope_for_chunk(chunk, default_scope)
            rows.append(
                {
                    "rag_partition": rag_partition,
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
                    "scope": scope,
                }
            )

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

        def _upsert_batch(connection, batch: List[dict]) -> None:
            if not batch:
                return
            values = [{**row, "embedded_at": None} for row in batch]
            stmt = pg_insert(rag_chunks_table).values(values)
            excluded = stmt.excluded
            stmt = stmt.on_conflict_do_update(
                index_elements=[
                    rag_chunks_table.c.rag_partition,
                    rag_chunks_table.c.chunk_id,
                ],
                set_={
                    "source_id": excluded.source_id,
                    "chunk_type": excluded.chunk_type,
                    "language": excluded.language,
                    "worldviews": excluded.worldviews,
                    "importance": excluded.importance,
                    "content_hash": excluded.content_hash,
                    "text": excluded.text,
                    "created_at": excluded.created_at,
                    "updated_at": excluded.updated_at,
                    "metadata": excluded.metadata,
                    "references": excluded.references,
                    "scope": excluded.scope,
                    "deprecated_at": None,
                    "embedded_at": case(
                        (
                            rag_chunks_table.c.content_hash != excluded.content_hash,
                            None,
                        ),
                        else_=rag_chunks_table.c.embedded_at,
                    ),
                },
                # Re-runs with unchanged summaries: skip row rewrite (major Supabase win).
                where=or_(
                    rag_chunks_table.c.content_hash != excluded.content_hash,
                    rag_chunks_table.c.deprecated_at.is_not(None),
                ),
            )
            connection.execute(stmt)

        def _write() -> None:
            with self.engine.begin() as connection:
                for offset in range(0, len(rows), _UPSERT_BATCH_SIZE):
                    _upsert_batch(connection, rows[offset : offset + _UPSERT_BATCH_SIZE])

        await asyncio.to_thread(_write)

    async def deprecate_orphans_for_sources(
        self,
        rag_partition: str,
        active_by_source: Dict[str, List[str]],
        *,
        chunk_types_by_source: Dict[str, List[str]] | None = None,
    ) -> Dict[str, int]:
        """Set deprecated_at=now() for non-active rows per (rag_partition, source_id).

        For each ``source_id`` in ``active_by_source``, rows in ``rag_partition`` with that
        ``source_id``, optional ``chunk_type`` filter, and ``chunk_id`` not in the active
        list are marked deprecated. Upserted batch rows stay active (``deprecated_at``
        cleared on conflict update when needed).

        Returns: number of rows marked deprecated per ``source_id`` (0 if none or ``active_ids`` empty).
        """

        if not active_by_source:
            return {}
        now = datetime.now(timezone.utc)

        def _write() -> Dict[str, int]:
            out: Dict[str, int] = {}
            with self.engine.begin() as connection:
                for source_id, active_ids in active_by_source.items():
                    if not active_ids:
                        out[source_id] = 0
                        continue
                    conditions = [
                        rag_chunks_table.c.rag_partition == rag_partition,
                        rag_chunks_table.c.source_id == source_id,
                        rag_chunks_table.c.chunk_id.not_in(active_ids),
                        rag_chunks_table.c.deprecated_at.is_(None),
                    ]
                    if chunk_types_by_source:
                        types = chunk_types_by_source.get(source_id)
                        if types:
                            conditions.append(rag_chunks_table.c.chunk_type.in_(types))
                    result = connection.execute(
                        update(rag_chunks_table)
                        .where(and_(*conditions))
                        .values(deprecated_at=now)
                    )
                    out[source_id] = int(result.rowcount or 0)
            return out

        return await asyncio.to_thread(_write)

    async def list_chunk_records_for_embed(
        self,
        assistant_rag_collection: str,
        *,
        shared_source_ids: list[str] | None = None,
        source_ids: list[str] | None = None,
        chunk_types: list[str] | None = None,
        only_unembedded: bool = False,
        max_chunks: int | None = None,
    ) -> List[ChunkRecord]:
        """Chunks to embed into Qdrant collection ``assistant_rag_collection``.

        Loads the assistant partition in full plus a subset of ``__shared__`` rows:
        - ``shared_source_ids is None``: all shared rows (legacy / open query).
        - ``shared_source_ids == []``: no shared rows.
        - otherwise: shared rows whose ``source_id`` is in the whitelist.

        Optional additional filters:
        - ``source_ids``: when non-empty, only rows whose ``source_id`` is in this list
          (applies to BOTH partitions). Use for per-source-id iteration from the client.
        - ``chunk_types``: when non-empty, only rows whose ``chunk_type`` is in this list.
          When ``[]``, returns no rows (caller uses this to mean “no filter” only if omitted).
        """

        asst = rag_chunks_table.c.rag_partition == assistant_rag_collection
        shared_col = rag_chunks_table.c.rag_partition == RAG_PARTITION_SHARED

        if shared_source_ids is not None and len(shared_source_ids) == 0:
            partition_condition = asst
        elif shared_source_ids is None:
            partition_condition = or_(asst, shared_col)
        else:
            partition_condition = or_(
                asst,
                and_(shared_col, rag_chunks_table.c.source_id.in_(shared_source_ids)),
            )

        if source_ids is not None and len(source_ids) > 0:
            condition = and_(
                partition_condition,
                rag_chunks_table.c.source_id.in_(source_ids),
            )
        else:
            condition = partition_condition

        if chunk_types is not None:
            if len(chunk_types) == 0:
                return []
            condition = and_(condition, rag_chunks_table.c.chunk_type.in_(chunk_types))

        active = rag_chunks_table.c.deprecated_at.is_(None)
        if only_unembedded:
            condition = and_(condition, rag_chunks_table.c.embedded_at.is_(None))

        def _select() -> List[ChunkRecord]:
            with self.engine.begin() as connection:
                stmt = (
                    select(rag_chunks_table)
                    .where(and_(condition, active))
                    .order_by(rag_chunks_table.c.updated_at.asc(), rag_chunks_table.c.chunk_id.asc())
                )
                if max_chunks is not None and max_chunks > 0:
                    stmt = stmt.limit(max_chunks)
                result = connection.execute(stmt)
                out: List[ChunkRecord] = []
                for row in result.mappings():
                    out.append(_row_to_chunk_record(row))
                return out

        return await asyncio.to_thread(_select)

    async def mark_embedded_for_embed_run(
        self,
        assistant_rag_collection: str,
        chunk_ids: Sequence[str],
    ) -> None:
        """Set embedded_at for rows in both __shared__ and the assistant partition."""

        ids = list(chunk_ids)
        if not ids:
            return
        now = datetime.now(timezone.utc)
        parts = or_(
            rag_chunks_table.c.rag_partition == RAG_PARTITION_SHARED,
            rag_chunks_table.c.rag_partition == assistant_rag_collection,
        )

        def _upd() -> None:
            with self.engine.begin() as connection:
                connection.execute(
                    update(rag_chunks_table)
                    .where(
                        parts,
                        rag_chunks_table.c.chunk_id.in_(ids),
                    )
                    .values(embedded_at=now)
                )

        await asyncio.to_thread(_upd)

    async def delete_chunks(self, rag_partition: str, chunk_ids: Iterable[str]) -> None:
        """Delete rows from rag_chunks for a single partition (e.g. assistant Qdrant name).

        Does not delete ``__shared__`` rows unless ``rag_partition`` is ``__shared__``.
        """

        ids = list(chunk_ids)
        if not ids:
            return

        def _delete() -> None:
            with self.engine.begin() as connection:
                connection.execute(
                    delete(rag_chunks_table).where(
                        rag_chunks_table.c.rag_partition == rag_partition,
                        rag_chunks_table.c.chunk_id.in_(ids),
                    )
                )

        await asyncio.to_thread(_delete)
