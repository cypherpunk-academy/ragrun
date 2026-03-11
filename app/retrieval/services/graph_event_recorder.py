"""Best-effort persistence for graph events (retrieval + LLM steps).

Delegates to EventRecorder for dual-write to event_metadata + event_content.
"""
from __future__ import annotations

import logging
from typing import Mapping, Sequence
from uuid import UUID

from app.retrieval.services.event_recorder import EventRecorder

logger = logging.getLogger(__name__)


class GraphEventRecorder:
    """Async repository to store per-step graph events.

    Writes to event_metadata (permanent) + event_content (7-day retention).
    Designed to be best-effort: failures are logged but never raised to callers.
    """

    def __init__(self, event_recorder: EventRecorder | None = None) -> None:
        self._recorder = event_recorder or EventRecorder()

    async def record_event(
        self,
        *,
        graph_event_id: str | UUID,
        graph_name: str,
        step: str,
        concept: str,
        collection: str | None = None,
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
        chunk_ids: list[str] | None = None
        if context_refs:
            refs = list(context_refs) if isinstance(context_refs, Sequence) else list(context_refs.values())
            chunk_ids = [r for r in refs if isinstance(r, str)]

        await self._recorder.record_event(
            endpoint=f"graphs/{graph_name}",
            graph_event_id=graph_event_id,
            graph_name=graph_name,
            step=step,
            concept=concept,
            collection=collection,
            worldview=worldview,
            query_text=query_text,
            prompt_messages=prompt_messages,
            context_refs=context_refs,
            context_source=context_source,
            context_text=context_text,
            response_text=response_text,
            retrieval_mode=retrieval_mode,
            sufficiency=sufficiency,
            errors=errors,
            metadata=metadata,
            chunk_ids=chunk_ids,
        )
