"""Persistence helper for retrieval logging."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Iterable, Mapping, Any

from sqlalchemy import insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker

from app.db.async_session import get_async_sessionmaker
from app.db.tables import retrieval_chunks_table, retrieval_events_table
from app.retrieval.models import RetrievedSnippet

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _extract_metadata(snippet: RetrievedSnippet) -> Mapping[str, Any]:
    payload = snippet.payload or {}
    nested = payload.get("payload") or payload
    return nested.get("metadata") or {}


class RetrievalLoggingRepository:
    """Async repository to store retrieval events and their chunks."""

    def __init__(self, session_factory: sessionmaker[AsyncSession] | None = None) -> None:
        self.session_factory = session_factory or get_async_sessionmaker()

    async def log_concept_explain(
        self,
        *,
        concept: str,
        branch: str,
        collection: str,
        answer: str,
        primary: Iterable[RetrievedSnippet],
        expanded: Iterable[RetrievedSnippet],
    ) -> None:
        """Persist the retrieval event and associated chunks."""

        try:
            async with self.session_factory() as session:
                async with session.begin():
                    event_id = await self._insert_event(
                        session,
                        concept=concept,
                        branch=branch,
                        collection=collection,
                        answer=answer,
                    )
                    await self._insert_chunks(session, event_id, "primary", primary)
                    await self._insert_chunks(session, event_id, "expanded", expanded)
        except Exception:
            logger.exception("Failed to log retrieval event for concept=%s", concept)

    async def _insert_event(
        self,
        session: AsyncSession,
        *,
        concept: str,
        branch: str,
        collection: str,
        answer: str,
    ) -> int:
        stmt = (
            insert(retrieval_events_table)
            .returning(retrieval_events_table.c.id)
        )
        result = await session.execute(
            stmt,
            {
                "concept": concept,
                "branch": branch,
                "collection": collection,
                "answer": answer,
                "created_at": _now(),
            },
        )
        return int(result.scalar_one())

    async def _insert_chunks(
        self,
        session: AsyncSession,
        event_id: int,
        kind: str,
        snippets: Iterable[RetrievedSnippet],
    ) -> None:
        rows = []
        now = _now()
        for snippet in snippets:
            metadata = _extract_metadata(snippet)
            rows.append(
                {
                    "event_id": event_id,
                    "kind": kind,
                    "text": snippet.text,
                    "score": snippet.score,
                    "source_id": metadata.get("source_id"),
                    "chunk_type": metadata.get("chunk_type"),
                    "metadata": metadata,
                    "created_at": now,
                }
            )

        if rows:
            await session.execute(insert(retrieval_chunks_table), rows)


def enqueue_log_concept_explain(
    *,
    repository: RetrievalLoggingRepository,
    concept: str,
    branch: str,
    collection: str,
    answer: str,
    primary: Iterable[RetrievedSnippet],
    expanded: Iterable[RetrievedSnippet],
) -> None:
    """Fire-and-forget helper to avoid blocking the main retrieval path."""

    async def _runner() -> None:
        await repository.log_concept_explain(
            concept=concept,
            branch=branch,
            collection=collection,
            answer=answer,
            primary=primary,
            expanded=expanded,
        )

    asyncio.create_task(_runner())


