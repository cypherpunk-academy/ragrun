"""Core ingestion service used by Phase 3 endpoints."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple
from uuid import UUID, uuid4, uuid5, NAMESPACE_DNS

from ragrun.models import ChunkRecord

from .embedding_client import EmbeddingClient
from .mirror_repository import ChunkMirrorRepository
from .qdrant_client import QdrantClient
from .telemetry import IngestionTelemetryClient


@dataclass(slots=True)
class UploadResult:
    """Structured response returned to the API layer."""

    ingestion_id: str
    collection: str
    requested: int
    ingested: int
    duplicates: int
    embedding_model: str
    vector_size: int
    unchanged: int
    changed: int
    new: int
    stale_deleted: int


@dataclass(slots=True)
class DeleteResult:
    """Structured delete response."""

    collection: str
    requested: int
    deleted: int


class IngestionService:
    """Coordinates validation, embedding, and Qdrant upserts."""

    _TAG_STRIP_RE = re.compile(r"</?\s*(q|i)\b[^>]*>", re.IGNORECASE)

    @staticmethod
    def _qdrant_filter_for_source(source_id: str) -> dict[str, object]:
        # Ingestion stores source_id as a flat payload field.
        return {"must": [{"key": "source_id", "match": {"value": source_id}}]}

    @staticmethod
    def _strip_markup(text: str) -> str:
        """Remove <q ...>...</q> and <i ...>...</i> tags, keep inner text."""

        return IngestionService._TAG_STRIP_RE.sub("", text or "")

    def __init__(
        self,
        *,
        embedding_client: EmbeddingClient,
        qdrant_client: QdrantClient,
        mirror_repository: ChunkMirrorRepository,
        telemetry_client: Optional[IngestionTelemetryClient] = None,
        default_batch_size: int = 64,
    ) -> None:
        self.embedding_client = embedding_client
        self.qdrant_client = qdrant_client
        self.mirror_repository = mirror_repository
        self.telemetry_client = telemetry_client
        self.default_batch_size = default_batch_size

    async def upload_chunks(
        self,
        *,
        collection: str,
        chunks: Sequence[ChunkRecord],
        embedding_model: str | None = None,
        batch_size: int | None = None,
        skip_cleanup: bool = False,
    ) -> UploadResult:
        """Validate, dedupe, embed, and upsert a batch of chunks.

        By default this performs a per-source_id cleanup of stale chunk_ids (sync-style).
        Set skip_cleanup=True when the caller will handle deletions explicitly (e.g. CLI sync).
        """

        if not chunks:
            raise ValueError("at least one chunk is required")

        unique_chunks, duplicate_count = self._dedupe_chunks(chunks)
        if not unique_chunks:
            raise ValueError("all chunks were filtered as duplicates")

        start_time = time.perf_counter()

        existing_payloads = await self._fetch_existing(collection, unique_chunks)
        unchanged, changed, new = self._classify_chunks(unique_chunks, existing_payloads)

        # Embed only changed + new
        to_embed = changed + new
        embedding_batch = None

        # Strip formatting tags (<q>, <i>) from embedding text to reduce noise,
        # while preserving the original text for storage and display.
        texts = [self._strip_markup(chunk.text) for chunk in to_embed]
        embeddings: Sequence[Sequence[float]] = []
        vector_size = 0
        if to_embed:
            embedding_batch = await self.embedding_client.embed_texts(
                texts,
                model_name=embedding_model,
                batch_size=batch_size or self.default_batch_size,
            )
            if len(embedding_batch.embeddings) != len(to_embed):
                raise RuntimeError(
                    "embedding count does not match chunk count "
                    f"({len(embedding_batch.embeddings)} != {len(to_embed)})"
                )
            embeddings = embedding_batch.embeddings
            vector_size = embedding_batch.dimensions

            await self.qdrant_client.ensure_collection(
                collection,
                vector_size=vector_size,
            )

            points = self._build_qdrant_points(to_embed, embeddings)
            await self.qdrant_client.upsert_points(collection, points)

        # For unchanged chunks, update payload without re-embedding
        if unchanged:
            payload_updates = []
            for chunk in unchanged:
                payload = chunk.metadata.model_dump(mode="json")
                payload["text"] = chunk.text
                payload["chunk_id"] = chunk.metadata.chunk_id
                payload["source_id"] = chunk.metadata.source_id
                payload["content_hash"] = chunk.metadata.content_hash
                payload_updates.append(
                    {
                        "id": str(uuid5(NAMESPACE_DNS, chunk.metadata.chunk_id)),
                        "payload": payload,
                    }
                )
            await self.qdrant_client.set_payload(collection, payload_updates)

        # Mirror: write all unique_chunks (so metadata/text stay aligned)
        await self.mirror_repository.upsert_chunks(collection, unique_chunks)

        # Cleanup stale chunk_ids for involved source_ids (sync-style), unless disabled
        stale_deleted = 0
        if not skip_cleanup:
            _, stale_deleted = await self._cleanup_stale(collection, unique_chunks)

        # Determine reporting values
        result_embedding_model = embedding_model or "skipped"
        if to_embed:
            # embedding_batch scoped above
            result_embedding_model = getattr(embedding_batch, "model_name", result_embedding_model)
        result_vector_size = vector_size

        result = UploadResult(
            ingestion_id=str(uuid4()),
            collection=collection,
            requested=len(chunks),
            # "ingested" = wirklich geschrieben/überschrieben in Qdrant (changed + new)
            ingested=len(changed) + len(new),
            duplicates=duplicate_count,
            embedding_model=result_embedding_model,
            vector_size=result_vector_size,
            unchanged=len(unchanged),
            changed=len(changed),
            new=len(new),
            stale_deleted=stale_deleted,
        )

        if self.telemetry_client:
            await self.telemetry_client.record_ingestion_run(
                ingestion_id=result.ingestion_id,
                collection=collection,
                count=result.ingested,
                duplicates=duplicate_count,
                duration_seconds=time.perf_counter() - start_time,
                embedding_model=result.embedding_model,
                vector_size=result.vector_size,
            )

        return result

    async def delete_chunks(
        self,
        *,
        collection: str,
        chunk_ids: Sequence[str],
    ) -> DeleteResult:
        """Delete chunk ids from Qdrant and Postgres."""

        if not chunk_ids:
            raise ValueError("chunk_ids must not be empty")

        # Convert chunk IDs to UUIDs for Qdrant
        point_uuids = [str(uuid5(NAMESPACE_DNS, cid)) for cid in chunk_ids]

        await self.qdrant_client.delete_points(collection, point_uuids)
        # Best-effort: mirror cleanup should not block Qdrant deletion.
        try:
            await self.mirror_repository.delete_chunks(collection, chunk_ids)
        except Exception:
            pass

        return DeleteResult(
            collection=collection,
            requested=len(chunk_ids),
            deleted=len(chunk_ids),
        )

    def _dedupe_chunks(self, chunks: Sequence[ChunkRecord]) -> tuple[List[ChunkRecord], int]:
        """Drop duplicates based on chunk_id + content_hash."""

        seen: set[tuple[str, str]] = set()
        deduped: List[ChunkRecord] = []
        duplicates = 0

        for chunk in chunks:
            key = (chunk.metadata.chunk_id, chunk.metadata.content_hash)
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            deduped.append(chunk)

        return deduped, duplicates

    async def _fetch_existing(
        self,
        collection: str,
        chunks: Sequence[ChunkRecord],
    ) -> dict[str, dict[str, object]]:
        """Retrieve existing points by chunk_id (returns payload keyed by chunk_id)."""

        if not chunks:
            return {}

        ids = [uuid5(NAMESPACE_DNS, c.metadata.chunk_id) for c in chunks]
        points = await self.qdrant_client.retrieve_points(
            collection,
            [str(pid) for pid in ids],
            with_vectors=False,
            with_payload=True,
        )
        existing: dict[str, dict[str, object]] = {}
        for point in points:
            payload = point.get("payload") or {}
            cid = payload.get("chunk_id")
            if isinstance(cid, str):
                existing[cid] = payload
        return existing

    def _classify_chunks(
        self,
        incoming: Sequence[ChunkRecord],
        existing_payloads: dict[str, dict[str, object]],
    ) -> Tuple[List[ChunkRecord], List[ChunkRecord], List[ChunkRecord]]:
        """Return (unchanged, changed, new) lists based on content_hash."""

        unchanged: List[ChunkRecord] = []
        changed: List[ChunkRecord] = []
        new: List[ChunkRecord] = []

        for chunk in incoming:
            cid = chunk.metadata.chunk_id
            existing = existing_payloads.get(cid)
            if existing is None:
                new.append(chunk)
                continue
            prev_hash = existing.get("content_hash")
            if isinstance(prev_hash, str) and prev_hash == chunk.metadata.content_hash:
                unchanged.append(chunk)
            else:
                changed.append(chunk)

        return unchanged, changed, new

    def _build_qdrant_points(
        self,
        chunks: Sequence[ChunkRecord],
        embeddings: Sequence[Sequence[float]],
    ) -> Iterable[dict[str, object]]:
        """Convert chunk records into Qdrant point payloads."""

        for chunk, vector in zip(chunks, embeddings):
            payload = chunk.metadata.model_dump(mode="json")
            payload["text"] = chunk.text
            payload["chunk_id"] = chunk.metadata.chunk_id
            payload["source_id"] = chunk.metadata.source_id
            payload["content_hash"] = chunk.metadata.content_hash
            
            # Qdrant requires UUID or unsigned int for point IDs
            # Use UUID v5 (deterministic) based on chunk_id
            point_uuid = uuid5(NAMESPACE_DNS, chunk.metadata.chunk_id)
            
            yield {
                "id": str(point_uuid),
                "vector": list(vector),
                "payload": payload,
            }

    async def _cleanup_stale(
        self,
        collection: str,
        chunks: Sequence[ChunkRecord],
    ) -> tuple[int, int]:
        """Delete chunk_ids of same source_ids that were not part of the upload."""

        if not chunks:
            return (0, 0)

        by_source: dict[str, set[str]] = {}
        for ch in chunks:
            by_source.setdefault(ch.metadata.source_id, set()).add(ch.metadata.chunk_id)

        total_stale = 0
        total_sources = 0

        for source_id, delivered_ids in by_source.items():
            total_sources += 1
            # Cleanup source-of-truth: Qdrant (mirror may be missing/out-of-date).
            existing_points = await self.qdrant_client.scroll_all_points(
                collection,
                filter_=self._qdrant_filter_for_source(source_id),
                limit=512,
                with_payload=True,
                with_vectors=False,
            )
            existing_ids: list[str] = []
            for item in existing_points:
                payload = item.get("payload") or {}
                cid = payload.get("chunk_id")
                if isinstance(cid, str) and cid:
                    existing_ids.append(cid)

            stale_ids = [cid for cid in existing_ids if cid not in delivered_ids]
            if not stale_ids:
                continue
            total_stale += len(stale_ids)
            # Delete from Qdrant (UUIDv5) and mirror
            point_uuids = [str(uuid5(NAMESPACE_DNS, cid)) for cid in stale_ids]
            await self.qdrant_client.delete_points(collection, point_uuids)
            # Best-effort: mirror cleanup should not block sync.
            try:
                await self.mirror_repository.delete_chunks(collection, stale_ids)
            except Exception:
                pass

        return (total_sources, total_stale)

