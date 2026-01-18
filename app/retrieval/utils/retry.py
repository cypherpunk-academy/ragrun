"""Async retry helpers for retrieval pipelines."""
from __future__ import annotations

import asyncio
import logging
import random
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")


async def retry_async(
    func: Callable[[], Awaitable[T]],
    *,
    retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 8.0,
    jitter: float = 0.2,
    retry_on: Callable[[Exception], bool] | None = None,
    logger: logging.Logger | None = None,
    operation: str | None = None,
) -> T:
    """Retry an async callable with exponential backoff and jitter."""
    attempt = 0
    while True:
        try:
            return await func()
        except Exception as exc:
            attempt += 1
            should_retry = retry_on(exc) if retry_on else True
            if attempt > retries or not should_retry:
                raise

            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            if jitter:
                delay = delay * (1.0 + random.uniform(-jitter, jitter))
            delay = max(0.0, delay)

            if logger:
                op = operation or "operation"
                logger.warning(
                    "Retrying %s after error (attempt %s/%s, delay=%.2fs): %s",
                    op,
                    attempt,
                    retries,
                    delay,
                    exc,
                )

            await asyncio.sleep(delay)
