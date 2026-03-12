"""Dual-table event recording: event_metadata (permanent) + event_content (7-day retention)."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence
from uuid import UUID

from sqlalchemy import insert, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker

from app.db.async_session import get_async_sessionmaker
from app.db.tables import event_content_table, event_metadata_table

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _to_json_safe(value: Any) -> Any:
    """Convert value to JSON-serializable form."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class EventRecorder:
    """Best-effort dual-write to event_metadata and event_content.

    Failures are logged but never raised to callers.
    """

    def __init__(self, session_factory: sessionmaker[AsyncSession] | None = None) -> None:
        self.session_factory = session_factory or get_async_sessionmaker()

    async def record_metadata_only(
        self,
        *,
        endpoint: str,
        collection: str | None = None,
        graph_event_id: str | UUID | None = None,
        graph_name: str | None = None,
        step: str | None = None,
        worldview: str | None = None,
        retrieval_mode: str | None = None,
        sufficiency: str | None = None,
        error_count: int | None = None,
        metadata: Mapping[str, Any] | None = None,
        chunk_ids: Sequence[str] | None = None,
    ) -> None:
        """Record metadata only (no content). For upload-chunks, delete-chunks."""
        row = {
            "endpoint": endpoint,
            "collection": collection,
            "graph_event_id": str(graph_event_id) if graph_event_id else None,
            "graph_name": graph_name,
            "step": step,
            "worldview": worldview,
            "retrieval_mode": retrieval_mode,
            "sufficiency": sufficiency,
            "error_count": error_count,
            "metadata": dict(metadata) if metadata else None,
            "chunk_ids": list(chunk_ids) if chunk_ids else None,
            "created_at": _now(),
        }
        await self._insert_metadata(row)

    async def record_event(
        self,
        *,
        endpoint: str,
        graph_event_id: str | UUID,
        graph_name: str,
        step: str,
        concept: str | None = None,
        worldview: str | None = None,
        query_text: str | None = None,
        prompt_messages: Mapping[str, object] | Sequence[object] | None = None,
        context_refs: Sequence[str] | Mapping[str, object] | None = None,
        context_source: Sequence[object] | Mapping[str, object] | None = None,
        context_text: str | None = None,
        response_text: str | None = None,
        retrieval_mode: str | None = None,
        sufficiency: str | None = None,
        errors: Sequence[str] | None = None,
        metadata: Mapping[str, object] | None = None,
        collection: str | None = None,
        chunk_ids: Sequence[str] | None = None,
    ) -> None:
        """Record metadata + content (for graph steps, concept-explain, quote-explain)."""
        error_count = len(errors) if errors else None
        meta_row = {
            "endpoint": endpoint,
            "graph_event_id": str(graph_event_id),
            "graph_name": graph_name,
            "step": step,
            "collection": collection,
            "worldview": worldview,
            "retrieval_mode": retrieval_mode,
            "sufficiency": sufficiency,
            "error_count": error_count,
            "metadata": dict(metadata) if metadata else None,
            "chunk_ids": list(chunk_ids) if chunk_ids else None,
            "created_at": _now(),
        }
        meta_id = await self._insert_metadata(meta_row)
        if meta_id is None:
            return

        content_row = {
            "event_metadata_id": meta_id,
            "thread_id": str(graph_event_id),
            "concept": concept,
            "query_text": query_text,
            "prompt_messages": (
                list(prompt_messages)
                if isinstance(prompt_messages, Sequence) and not isinstance(prompt_messages, (str, bytes))
                else prompt_messages
            ),
            "context_refs": (
                list(context_refs)
                if isinstance(context_refs, Sequence) and not isinstance(context_refs, (str, bytes))
                else context_refs
            ),
            "context_source": (
                list(context_source)
                if isinstance(context_source, Sequence) and not isinstance(context_source, (str, bytes))
                else context_source
            ),
            "context_text": context_text,
            "response_text": response_text,
            "errors": list(errors) if errors else None,
            "created_at": _now(),
        }
        await self._insert_content(content_row)

    async def _insert_metadata(self, row: Mapping[str, Any]) -> int | None:
        try:
            async with self.session_factory() as session:
                result = await session.execute(
                    insert(event_metadata_table).values(**row).returning(event_metadata_table.c.id)
                )
                meta_id = result.scalar_one()
                await session.commit()
                return meta_id
        except Exception:
            logger.exception("Failed to insert event_metadata endpoint=%s", row.get("endpoint"))
            return None

    async def _insert_content(self, row: Mapping[str, Any]) -> None:
        try:
            async with self.session_factory() as session:
                await session.execute(insert(event_content_table), [row])
                await session.commit()
        except Exception:
            logger.exception("Failed to insert event_content event_metadata_id=%s", row.get("event_metadata_id"))


def enqueue_record_metadata_only(recorder: EventRecorder, **kwargs: Any) -> None:
    """Fire-and-forget metadata-only recording."""

    async def _run() -> None:
        await recorder.record_metadata_only(**kwargs)

    asyncio.create_task(_run())


def enqueue_record_event(recorder: EventRecorder, **kwargs: Any) -> None:
    """Fire-and-forget metadata+content recording."""

    async def _run() -> None:
        await recorder.record_event(**kwargs)

    asyncio.create_task(_run())
