"""Primary chunk store (rag_chunks table, DB-first)."""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Mapping, Sequence

logger = logging.getLogger(__name__)

# Serialize embed DB reads: one open Supavisor connection at a time.
_embed_db_lock = asyncio.Lock()

from sqlalchemy import and_, case, delete, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Engine

from app.db.session import fetch_mappings_autocommit, fetch_one_autocommit

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


def build_embed_chunks_condition(
    assistant_rag_collection: str,
    *,
    shared_source_ids: list[str] | None = None,
    source_ids: list[str] | None = None,
    chunk_types: list[str] | None = None,
    only_unembedded: bool = False,
) -> object | None:
    """SQLAlchemy condition for embed-chunks queries, or None when no rows match."""

    asst = rag_chunks_table.c.rag_partition == assistant_rag_collection
    shared_col = rag_chunks_table.c.rag_partition == RAG_PARTITION_SHARED

    if source_ids is not None and len(source_ids) > 0:
        sid_in = rag_chunks_table.c.source_id.in_(source_ids)
        if shared_source_ids is not None and len(shared_source_ids) == 0:
            partition_condition = and_(asst, sid_in)
        else:
            partition_condition = and_(sid_in, or_(asst, shared_col))
        condition = partition_condition
    elif shared_source_ids is not None and len(shared_source_ids) == 0:
        condition = asst
    elif shared_source_ids is None:
        condition = or_(asst, shared_col)
    else:
        condition = or_(
            asst,
            and_(shared_col, rag_chunks_table.c.source_id.in_(shared_source_ids)),
        )

    if chunk_types is not None:
        if len(chunk_types) == 0:
            return None
        condition = and_(condition, rag_chunks_table.c.chunk_type.in_(chunk_types))

    active = rag_chunks_table.c.deprecated_at.is_(None)
    if only_unembedded:
        condition = and_(condition, rag_chunks_table.c.embedded_at.is_(None))

    return and_(condition, active)


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
                # Skip no-op re-runs when text unchanged; still apply metadata-only repairs.
                where=or_(
                    rag_chunks_table.c.content_hash != excluded.content_hash,
                    rag_chunks_table.c.deprecated_at.is_not(None),
                    rag_chunks_table.c.metadata.is_distinct_from(excluded.metadata),
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
    ) -> tuple[Dict[str, int], List[str]]:
        """Set deprecated_at=now() for non-active rows per (rag_partition, source_id).

        For each ``source_id`` in ``active_by_source``, rows in ``rag_partition`` with that
        ``source_id``, optional ``chunk_type`` filter, and ``chunk_id`` not in the active
        list are marked deprecated. Upserted batch rows stay active (``deprecated_at``
        cleared on conflict update when needed).

        Returns: (per ``source_id`` deprecation counts, all newly deprecated ``chunk_id``s).
        """

        if not active_by_source:
            return {}, []
        now = datetime.now(timezone.utc)

        def _write() -> tuple[Dict[str, int], List[str]]:
            out: Dict[str, int] = {}
            deprecated_ids: List[str] = []
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
                        .returning(rag_chunks_table.c.chunk_id)
                    )
                    ids = [str(row[0]) for row in result.fetchall()]
                    deprecated_ids.extend(ids)
                    out[source_id] = len(ids)
            return out, deprecated_ids

        return await asyncio.to_thread(_write)

    async def list_active_chunk_ids_for_embed(
        self,
        assistant_rag_collection: str,
        source_type_keys: set[tuple[str, str]],
    ) -> dict[tuple[str, str], set[str]]:
        """Active rag_chunks chunk_ids per (source_id, chunk_type) for embed cleanup."""

        if not source_type_keys:
            return {}

        partitions = (assistant_rag_collection, RAG_PARTITION_SHARED)
        keys_list = list(source_type_keys)

        def _select() -> dict[tuple[str, str], set[str]]:
            out: dict[tuple[str, str], set[str]] = {key: set() for key in source_type_keys}
            for offset in range(0, len(keys_list), 100):
                batch = keys_list[offset : offset + 100]
                key_conditions = [
                    and_(
                        rag_chunks_table.c.source_id == source_id,
                        rag_chunks_table.c.chunk_type == chunk_type,
                    )
                    for source_id, chunk_type in batch
                ]
                stmt = (
                    select(
                        rag_chunks_table.c.source_id,
                        rag_chunks_table.c.chunk_type,
                        rag_chunks_table.c.chunk_id,
                    )
                    .where(
                        rag_chunks_table.c.deprecated_at.is_(None),
                        rag_chunks_table.c.rag_partition.in_(partitions),
                        or_(*key_conditions),
                    )
                )
                rows = fetch_mappings_autocommit(self.engine, stmt)
                for row in rows:
                    key = (str(row["source_id"]), str(row["chunk_type"]))
                    if key in out:
                        out[key].add(str(row["chunk_id"]))
            return out

        async with _embed_db_lock:
            return await asyncio.to_thread(_select)

    async def list_embed_cleanup_scope(
        self,
        assistant_rag_collection: str,
        *,
        shared_source_ids: list[str] | None = None,
        source_ids: list[str] | None = None,
        chunk_types: list[str] | None = None,
    ) -> set[tuple[str, str]]:
        """Distinct active (source_id, chunk_type) keys matching embed-chunks filters."""

        condition = build_embed_chunks_condition(
            assistant_rag_collection,
            shared_source_ids=shared_source_ids,
            source_ids=source_ids,
            chunk_types=chunk_types,
            only_unembedded=False,
        )
        if condition is None:
            return set()

        def _select() -> set[tuple[str, str]]:
            rows = fetch_mappings_autocommit(
                self.engine,
                select(
                    rag_chunks_table.c.source_id,
                    rag_chunks_table.c.chunk_type,
                )
                .where(condition)
                .distinct(),
            )
            return {(str(row["source_id"]), str(row["chunk_type"])) for row in rows}

        async with _embed_db_lock:
            return await asyncio.to_thread(_select)

    async def deprecate_chunk_ids(
        self,
        rag_partition: str,
        chunk_ids: List[str],
    ) -> int:
        """Set deprecated_at=now() for the given chunk_ids in rag_partition (active rows only)."""

        if not chunk_ids:
            return 0
        now = datetime.now(timezone.utc)

        def _write() -> int:
            with self.engine.begin() as connection:
                result = connection.execute(
                    update(rag_chunks_table)
                    .where(
                        and_(
                            rag_chunks_table.c.rag_partition == rag_partition,
                            rag_chunks_table.c.chunk_id.in_(chunk_ids),
                            rag_chunks_table.c.deprecated_at.is_(None),
                        )
                    )
                    .values(deprecated_at=now)
                )
                return int(result.rowcount or 0)

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

        condition = build_embed_chunks_condition(
            assistant_rag_collection,
            shared_source_ids=shared_source_ids,
            source_ids=source_ids,
            chunk_types=chunk_types,
            only_unembedded=only_unembedded,
        )
        if condition is None:
            return []

        t0 = time.perf_counter()
        logger.info(
            "list_chunk_records_for_embed: querying Supabase — collection=%s source_ids=%s",
            assistant_rag_collection,
            source_ids,
        )

        def _select() -> List[ChunkRecord]:
            stmt = (
                select(rag_chunks_table)
                .where(condition)
                .order_by(rag_chunks_table.c.updated_at.asc(), rag_chunks_table.c.chunk_id.asc())
            )
            if max_chunks is not None and max_chunks > 0:
                stmt = stmt.limit(max_chunks)
            rows = fetch_mappings_autocommit(self.engine, stmt)
            return [_row_to_chunk_record(row) for row in rows]

        # Do NOT wrap asyncio.to_thread in asyncio.wait_for: cancelling the await
        # abandons the worker thread, which keeps its Postgres transaction open
        # (idle in transaction) and leaks Supavisor connections.
        async with _embed_db_lock:
            chunks = await asyncio.to_thread(_select)
        logger.info(
            "list_chunk_records_for_embed: done in %.2fs — %d chunks returned",
            time.perf_counter() - t0,
            len(chunks),
        )
        return chunks

    async def stats_for_embed(
        self,
        assistant_rag_collection: str,
        *,
        shared_source_ids: list[str] | None = None,
        source_ids: list[str] | None = None,
        chunk_types: list[str] | None = None,
        only_unembedded: bool = False,
        max_chunks: int | None = None,
    ) -> tuple[int, float]:
        """Return (chunk_count, text_kb) for embed-chunks filters without loading rows."""

        condition = build_embed_chunks_condition(
            assistant_rag_collection,
            shared_source_ids=shared_source_ids,
            source_ids=source_ids,
            chunk_types=chunk_types,
            only_unembedded=only_unembedded,
        )
        if condition is None:
            return 0, 0.0

        def _aggregate() -> tuple[int, float]:
            base = (
                select(
                    rag_chunks_table.c.text,
                    rag_chunks_table.c.updated_at,
                    rag_chunks_table.c.chunk_id,
                )
                .where(condition)
                .order_by(rag_chunks_table.c.updated_at.asc(), rag_chunks_table.c.chunk_id.asc())
            )
            if max_chunks is not None and max_chunks > 0:
                base = base.limit(max_chunks)
            limited = base.subquery("embed_stats_rows")
            row = fetch_one_autocommit(
                self.engine,
                select(
                    func.count(),
                    func.coalesce(func.sum(func.octet_length(limited.c.text)), 0),
                ).select_from(limited),
            )
            count = int(row[0] or 0)
            text_bytes = int(row[1] or 0)
            return count, text_bytes / 1024.0

        async with _embed_db_lock:
            return await asyncio.to_thread(_aggregate)

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
