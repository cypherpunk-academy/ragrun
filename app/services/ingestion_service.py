"""Core ingestion service used by Phase 3 endpoints."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence
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


@dataclass(slots=True)
class DeleteResult:
    """Structured delete response."""

    collection: str
    requested: int
    deleted: int


class IngestionService:
    """Coordinates validation, embedding, and Qdrant upserts."""

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
    ) -> UploadResult:
        """Validate, dedupe, embed, and upsert a batch of chunks."""

        if not chunks:
            raise ValueError("at least one chunk is required")

        unique_chunks, duplicate_count = self._dedupe_chunks(chunks)
        if not unique_chunks:
            raise ValueError("all chunks were filtered as duplicates")

        start_time = time.perf_counter()

        texts = [chunk.text for chunk in unique_chunks]
        embedding_batch = await self.embedding_client.embed_texts(
            texts,
            model_name=embedding_model,
            batch_size=batch_size or self.default_batch_size,
        )

        if len(embedding_batch.embeddings) != len(unique_chunks):
            raise RuntimeError(
                "embedding count does not match chunk count "
                f"({len(embedding_batch.embeddings)} != {len(unique_chunks)})"
            )

        vector_size = embedding_batch.dimensions

        await self.qdrant_client.ensure_collection(
            collection,
            vector_size=vector_size,
        )

        points = self._build_qdrant_points(unique_chunks, embedding_batch.embeddings)
        await self.qdrant_client.upsert_points(collection, points)
        await self.mirror_repository.upsert_chunks(collection, unique_chunks)

        result = UploadResult(
            ingestion_id=str(uuid4()),
            collection=collection,
            requested=len(chunks),
            ingested=len(unique_chunks),
            duplicates=duplicate_count,
            embedding_model=embedding_batch.model_name,
            vector_size=vector_size,
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
        await self.mirror_repository.delete_chunks(collection, chunk_ids)

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

