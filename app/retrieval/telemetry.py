"""Best-effort telemetry hooks for retrieval flows (LangFuse-ready)."""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

import httpx

from app.config import settings


class RetrievalTelemetry:
    """Publishes retrieval metrics to LangFuse when configured."""

    def __init__(self) -> None:
        self.host = str(settings.langfuse_host).rstrip("/") if settings.langfuse_host else None
        self.public_key = settings.langfuse_public_key
        self.secret_key = settings.langfuse_secret_key
        self.dataset = (
            getattr(settings, "langfuse_retrieval_dataset", None)
            or settings.langfuse_ingestion_dataset
        )
        self.enabled = bool(self.host and self.public_key and self.secret_key and self.dataset)
        self._endpoint = f"{self.host}/api/public/ingestion/events" if self.host else None

    async def record_retrieval(
        self,
        *,
        trace_id: Optional[str],
        agent: str,
        branch: str,
        concept: str,
        retrieved: int,
        expanded: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enabled or not self._endpoint:
            return

        payload: Dict[str, Any] = {
            "traceId": trace_id,
            "name": "rag_retrieval",
            "timestamp": int(time.time() * 1000),
            "dataset": self.dataset,
            "metadata": {
                "agent": agent,
                "branch": branch,
                "concept": concept,
                "retrieved": retrieved,
                "expanded": expanded,
                **(metadata or {}),
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
            # Telemetry must never break retrieval.
            return

    async def record_worldviews(
        self,
        *,
        trace_id: Optional[str],
        graph_id: Optional[str],
        concept: str,
        worldviews: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Generic event for concept-explain-worldviews runs."""

        if not self.enabled or not self._endpoint:
            return

        payload: Dict[str, Any] = {
            "traceId": trace_id,
            "name": "concept_explain_worldviews",
            "timestamp": int(time.time() * 1000),
            "dataset": self.dataset,
            "metadata": {
                "graph_id": graph_id,
                "concept": concept,
                "worldviews": worldviews,
                **(metadata or {}),
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
            return


retrieval_telemetry = RetrievalTelemetry()


