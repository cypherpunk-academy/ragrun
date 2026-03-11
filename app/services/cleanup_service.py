"""Background cleanup service for event_content 7-day retention."""
from __future__ import annotations

import asyncio
import logging

from sqlalchemy import text

from app.db.async_session import get_async_sessionmaker

logger = logging.getLogger(__name__)

_CLEANUP_INTERVAL_SECONDS = 24 * 60 * 60  # 24 Stunden
_DELETE_SQL = text(
    "DELETE FROM event_content WHERE created_at < NOW() - INTERVAL '7 days'"
)


async def run_cleanup_once() -> int:
    """Delete event_content rows older than 7 days. Returns deleted row count."""
    session_factory = get_async_sessionmaker()
    async with session_factory() as session:
        async with session.begin():
            result = await session.execute(_DELETE_SQL)
            deleted: int = result.rowcount
    if deleted > 0:
        logger.info("event_content cleanup: deleted %d rows older than 7 days", deleted)
    return deleted


async def run_cleanup_loop() -> None:
    """Periodically delete stale event_content rows. Runs every 24 hours."""
    logger.info("event_content cleanup loop started (interval=%ds)", _CLEANUP_INTERVAL_SECONDS)
    while True:
        try:
            await run_cleanup_once()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("event_content cleanup failed – will retry in %ds", _CLEANUP_INTERVAL_SECONDS)
        await asyncio.sleep(_CLEANUP_INTERVAL_SECONDS)
