from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

logger = logging.getLogger(__name__)

COLLECTION = "ragscribe"
VECTOR_SIZE = 1
DUMMY_VECTOR = [0.0]

STATUS_EMOJI = {
    "ok": "✅",
    "queued": "📥",
    "processing": "🔄",
    "error": "❌",
}


def status_emoji(status: str) -> str:
    return STATUS_EMOJI.get(status, "❓")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class ScribeStore:
    def __init__(self) -> None:
        url = (os.environ.get("QDRANT_URL") or "http://127.0.0.1:6333").rstrip("/")
        api_key = (os.environ.get("QDRANT_API_KEY") or "").strip() or None
        # Server on Tuxai is Qdrant 1.11; client 1.12 is close enough.
        self._client = QdrantClient(url=url, api_key=api_key, prefer_grpc=False)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        names = {c.name for c in self._client.get_collections().collections}
        if COLLECTION not in names:
            logger.info("Creating Qdrant collection %s", COLLECTION)
            self._client.create_collection(
                collection_name=COLLECTION,
                vectors_config=qm.VectorParams(
                    size=VECTOR_SIZE,
                    distance=qm.Distance.COSINE,
                ),
            )
        for field in ("source", "status"):
            try:
                self._client.create_payload_index(
                    collection_name=COLLECTION,
                    field_name=field,
                    field_schema=qm.PayloadSchemaType.KEYWORD,
                )
            except Exception:
                # Index may already exist
                pass

    def mark_interrupted(self) -> int:
        """Mark leftover queued/processing points as error after restart."""
        points, _ = self._client.scroll(
            collection_name=COLLECTION,
            scroll_filter=qm.Filter(
                should=[
                    qm.FieldCondition(
                        key="status", match=qm.MatchValue(value="queued")
                    ),
                    qm.FieldCondition(
                        key="status", match=qm.MatchValue(value="processing")
                    ),
                ]
            ),
            limit=10_000,
            with_payload=True,
            with_vectors=False,
        )
        count = 0
        for p in points:
            pid = str(p.id)
            payload = dict(p.payload or {})
            payload["status"] = "error"
            payload["error"] = "interrupted"
            payload["pct"] = payload.get("pct") or 0
            self.upsert(pid, payload)
            count += 1
        if count:
            logger.warning("Marked %s interrupted jobs as error", count)
        return count

    def upsert(self, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = dict(payload)
        body.setdefault("created_at", _utc_now_iso())
        self._client.upsert(
            collection_name=COLLECTION,
            points=[
                qm.PointStruct(
                    id=job_id,
                    vector=DUMMY_VECTOR,
                    payload=body,
                )
            ],
        )
        return self.to_public(job_id, body)

    def get(self, job_id: str) -> Optional[dict[str, Any]]:
        try:
            UUID(job_id)
        except ValueError:
            return None
        points = self._client.retrieve(
            collection_name=COLLECTION,
            ids=[job_id],
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            return None
        return self.to_public(job_id, dict(points[0].payload or {}))

    def delete(self, job_id: str) -> bool:
        existing = self.get(job_id)
        if existing is None:
            return False
        self._client.delete(
            collection_name=COLLECTION,
            points_selector=qm.PointIdsList(points=[job_id]),
        )
        return True

    def list_jobs(
        self,
        *,
        source: Optional[str] = None,
        q: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 200))
        conditions: list[qm.FieldCondition] = []
        if source and source in {"youtube", "tiktok", "other"}:
            conditions.append(
                qm.FieldCondition(key="source", match=qm.MatchValue(value=source))
            )
        scroll_filter = qm.Filter(must=conditions) if conditions else None

        # Over-fetch when text-filtering; personal volume stays small.
        fetch_limit = 500 if (q or "").strip() else limit
        points, _ = self._client.scroll(
            collection_name=COLLECTION,
            scroll_filter=scroll_filter,
            limit=fetch_limit,
            with_payload=True,
            with_vectors=False,
        )

        items = [
            self.to_public(str(p.id), dict(p.payload or {})) for p in points
        ]
        items.sort(key=lambda x: x.get("created_at") or "", reverse=True)

        needle = (q or "").strip().casefold()
        if needle:
            items = [
                it
                for it in items
                if needle in (it.get("text") or "").casefold()
                or needle in (it.get("url") or "").casefold()
            ]
        return items[:limit]

    def patch(self, job_id: str, **fields: Any) -> dict[str, Any]:
        current = self.get(job_id)
        if current is None:
            raise KeyError(job_id)
        payload = {
            k: v
            for k, v in current.items()
            if k not in {"id", "status_emoji"}
        }
        payload.update(fields)
        return self.upsert(job_id, payload)

    @staticmethod
    def to_public(job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        status = str(payload.get("status") or "queued")
        return {
            "id": job_id,
            "url": payload.get("url") or "",
            "source": payload.get("source") or "other",
            "status": status,
            "status_emoji": status_emoji(status),
            "created_at": payload.get("created_at"),
            "duration_sec": payload.get("duration_sec"),
            "text": payload.get("text") or "",
            "error": payload.get("error"),
            "pct": payload.get("pct") if payload.get("pct") is not None else 0,
        }


_store: ScribeStore | None = None


def get_store() -> ScribeStore:
    global _store
    if _store is None:
        _store = ScribeStore()
    return _store
