"""Repository protocol boundaries for /app/* services (Postgres-backed today)."""
from __future__ import annotations

from typing import Any, Protocol, Sequence


class CatalogPort(Protocol):
    """Read-only catalogue: sources, segments, chunk text."""

    async def list_sources(self) -> list[dict[str, Any]]: ...

    async def list_segments(self, source_id: str) -> list[dict[str, Any]]: ...

    async def get_chunk_text(self, chunk_id: str, *, source_id: str | None = None) -> dict[str, Any] | None: ...


class TalksPort(Protocol):
    """App chat persistence in rag_talks / rag_turns."""

    async def create_talk_turn(
        self,
        *,
        user_id: str,
        user_name: str,
        collection: str,
        personality: str,
        title: str,
        user_message: str,
        assistant_message: str,
        talk_id: str | None = None,
        kontext_meta: dict[str, Any] | None = None,
        usage: dict[str, Any] | None = None,
    ) -> dict[str, str]: ...

    async def load_talk_turns(self, talk_id: str) -> Sequence[dict[str, Any]]: ...

    async def save_talk_summary(self, talk_id: str, summary: str) -> None: ...
