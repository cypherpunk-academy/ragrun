"""Fire-and-forget LLM usage recording into rag_usage table."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker

from app.db.async_session import get_async_sessionmaker
from app.db.tables import rag_usage_table

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class UsageRecorder:
    """Best-effort insert into rag_usage. Failures are logged but never raised."""

    def __init__(self, session_factory: sessionmaker[AsyncSession] | None = None) -> None:
        self.session_factory = session_factory or get_async_sessionmaker()

    async def record(
        self,
        *,
        account_id: str,
        model: str,
        provider: str,
        endpoint: str,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
        thread_id: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        row = {
            "account_id": account_id,
            "thread_id": thread_id,
            "endpoint": endpoint,
            "model": model,
            "provider": provider,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "extra": extra or None,
            "created_at": _now(),
        }
        try:
            async with self.session_factory() as session:
                await session.execute(insert(rag_usage_table).values(**row))
                await session.commit()
        except Exception:
            logger.exception(
                "Failed to record usage endpoint=%s account=%s", endpoint, account_id
            )


def enqueue_record_usage(recorder: UsageRecorder, **kwargs: Any) -> None:
    """Fire-and-forget usage recording."""

    async def _run() -> None:
        await recorder.record(**kwargs)

    asyncio.create_task(_run())
