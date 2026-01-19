"""Best-effort persistence for graph events (retrieval + LLM steps)."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Mapping, Sequence
from uuid import UUID

from sqlalchemy.exc import DBAPIError
from sqlalchemy import insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker

from app.db.async_session import get_async_sessionmaker
from app.db.tables import retrieval_graph_events_table

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class GraphEventRecorder:
    """Async repository to store per-step graph events.

    Designed to be best-effort: failures are logged but never raised to callers.
    """

    def __init__(self, session_factory: sessionmaker[AsyncSession] | None = None) -> None:
        self.session_factory = session_factory or get_async_sessionmaker()

    async def record_event(
        self,
        *,
        graph_event_id: str | UUID,
        graph_name: str,
        step: str,
        concept: str,
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
    ) -> None:
        row = {
            "graph_event_id": str(graph_event_id),
            "graph_name": graph_name,
            "step": step,
            "concept": concept,
            "worldview": worldview,
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
            "retrieval_mode": retrieval_mode,
            "sufficiency": sufficiency,
            "errors": list(errors) if errors else None,
            "metadata": dict(metadata) if metadata else None,
            "created_at": _now(),
        }

        async def _insert(payload: Mapping[str, object]) -> None:
            async with self.session_factory() as session:
                async with session.begin():
                    await session.execute(insert(retrieval_graph_events_table), [payload])

        try:
            await _insert(row)
        except DBAPIError as exc:
            # Backward compatibility: if DB schema hasn't been migrated yet, retry without the new column.
            msg = str(exc).lower()
            if "context_source" in msg and "does not exist" in msg:
                try:
                    fallback = dict(row)
                    fallback.pop("context_source", None)
                    await _insert(fallback)
                    return
                except Exception:
                    logger.exception(
                        "Failed to record graph event step=%s graph=%s (retry without context_source)",
                        step,
                        graph_name,
                    )
                    return

            logger.exception("Failed to record graph event step=%s graph=%s", step, graph_name)
        except Exception:
            logger.exception("Failed to record graph event step=%s graph=%s", step, graph_name)
