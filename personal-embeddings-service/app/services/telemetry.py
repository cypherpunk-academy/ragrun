"""Lightweight telemetry hooks for LangFuse datasets."""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, Optional

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class TelemetryClient:
    """Sends embedding metrics to LangFuse when configured."""

    def __init__(self) -> None:
        self.host = settings.langfuse_host.rstrip("/") if settings.langfuse_host else None
        self.public_key = settings.langfuse_public_key
        self.secret_key = settings.langfuse_secret_key
        self.dataset = settings.langfuse_dataset
        self.enabled = bool(self.host and self.public_key and self.secret_key)
        self._endpoint = (
            f"{self.host}/api/public/ingestion/events" if self.host else None
        )

    async def record_embedding_batch(
        self,
        route: str,
        count: int,
        duration_seconds: float,
        model_name: str,
    ) -> None:
        if not self.enabled or not self._endpoint:
            return

        payload: Dict[str, Any] = {
            "traceId": str(uuid.uuid4()),
            "name": "embedding_batch",
            "timestamp": int(time.time() * 1000),
            "dataset": self.dataset,
            "metadata": {
                "route": route,
                "count": count,
                "duration_ms": round(duration_seconds * 1000, 3),
                "model": model_name,
            },
        }
        headers = {
            "Content-Type": "application/json",
            "X-Langfuse-Public-Key": self.public_key,
            "X-Langfuse-Secret-Key": self.secret_key,
        }

        try:
            async with httpx.AsyncClient(timeout=settings.telemetry_timeout_seconds) as client:
                await client.post(self._endpoint, json=payload, headers=headers)
        except Exception as exc:  # pragma: no cover - best effort telemetry
            logger.debug("LangFuse telemetry failed: %s", exc)


telemetry_client = TelemetryClient()
