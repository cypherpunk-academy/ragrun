"""Best-effort LangFuse telemetry helpers."""
from __future__ import annotations

import time
from typing import Any, Dict

import httpx

from app.config import settings


class IngestionTelemetryClient:
    """Publishes ingestion metrics to LangFuse when configured."""

    def __init__(self) -> None:
        self.host = (
            str(settings.langfuse_host).rstrip("/") if settings.langfuse_host else None
        )
        self.public_key = settings.langfuse_public_key
        self.secret_key = settings.langfuse_secret_key
        self.dataset = settings.langfuse_ingestion_dataset
        self.enabled = bool(self.host and self.public_key and self.secret_key and self.dataset)
        self._endpoint = (
            f"{self.host}/api/public/ingestion/events" if self.host else None
        )

    async def record_ingestion_run(
        self,
        *,
        ingestion_id: str,
        collection: str,
        count: int,
        duplicates: int,
        duration_seconds: float,
        embedding_model: str,
        vector_size: int,
    ) -> None:
        if not self.enabled or not self._endpoint:
            return

        payload: Dict[str, Any] = {
            "traceId": ingestion_id,
            "name": "rag_ingestion",
            "timestamp": int(time.time() * 1000),
            "dataset": self.dataset,
            "metadata": {
                "collection": collection,
                "count": count,
                "duplicates": duplicates,
                "duration_ms": round(duration_seconds * 1000, 3),
                "embedding_model": embedding_model,
                "vector_size": vector_size,
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
        except Exception:
            # Intentional swallow; telemetry must never break ingestion.
            return


telemetry_client = IngestionTelemetryClient()

