"""Core ingestion service used by Phase 3 endpoints."""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple
from uuid import UUID, uuid4, uuid5, NAMESPACE_DNS

from app.debug_agent_log import agent_log
from app.shared.models import ChunkRecord
from app.infra.embedding_client import EmbeddingClient
from app.infra.qdrant_client import QdrantClient
from app.infra.sparse_embedder import SparseEmbedder
from app.ingestion.repositories import VectorChunksRepository
from app.core.telemetry import IngestionTelemetryClient

logger = logging.getLogger(__name__)


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
    payload_changed: int
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
    _SOFT_HYPHEN_RE = re.compile("\u00ad")

    @staticmethod
    def _qdrant_filter_for_source(source_id: str) -> dict[str, object]:
        # Ingestion stores source_id as a flat payload field.
        return {"must": [{"key": "source_id", "match": {"value": source_id}}]}

    @staticmethod
    def _qdrant_filter_for_source_and_type(
        source_id: str,
        chunk_type: str,
    ) -> dict[str, object]:
        return {
            "must": [
                {"key": "source_id", "match": {"value": source_id}},
                {"key": "chunk_type", "match": {"value": chunk_type}},
            ]
        }

    @staticmethod
    def _strip_markup(text: str) -> str:
        """Remove <q ...>...</q> and <i ...>...</i> tags, keep inner text."""

        return IngestionService._TAG_STRIP_RE.sub("", text or "")

    @classmethod
    def _prepare_embedding_text(cls, text: str) -> str:
        """Normalize chunk text for dense/sparse embedding (storage text unchanged)."""

        stripped = cls._strip_markup(text)
        return cls._SOFT_HYPHEN_RE.sub("", stripped)

    def __init__(
        self,
        *,
        embedding_client: EmbeddingClient,
        qdrant_client: QdrantClient,
        vector_chunks_repository: VectorChunksRepository,
        telemetry_client: Optional[IngestionTelemetryClient] = None,
        sparse_embedder: Optional[SparseEmbedder] = None,
        default_batch_size: int = 64,
    ) -> None:
        self.embedding_client = embedding_client
        self.qdrant_client = qdrant_client
        self.vector_chunks_repository = vector_chunks_repository
        self.telemetry_client = telemetry_client
        self.sparse_embedder = sparse_embedder
        self.default_batch_size = default_batch_size

    async def upload_chunks(
        self,
        *,
        collection: str,
        chunks: Sequence[ChunkRecord],
        embedding_model: str | None = None,
        batch_size: int | None = None,
        skip_cleanup: bool = False,
        cleanup_active_ids: Mapping[tuple[str, str], set[str]] | None = None,
        prefix_passage: str | None = None,
        shared_book_chunk_type_override: str | None = None,
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

        # Assistant-specific mapping for shared corpus base chunks:
        # keep rag_chunks neutral (`book`), but allow embed-time role override
        # for both shared books and shared lectures in Phase 1.
        if shared_book_chunk_type_override in ("book", "secondary_book"):
            for ch in unique_chunks:
                source_type = (ch.metadata.source_type or "").strip().lower()
                if source_type in ("book", "lecture") and ch.metadata.chunk_type == "book":
                    ch.metadata.chunk_type = shared_book_chunk_type_override

        start_time = time.perf_counter()
        logger.info(
            "upload_chunks: collection=%s requested=%d unique=%d duplicates=%d",
            collection, len(chunks), len(unique_chunks), duplicate_count,
        )

        t0 = time.perf_counter()
        existing_payloads = await self._fetch_existing(collection, unique_chunks)
        logger.info(
            "upload_chunks: _fetch_existing done in %.2fs — found %d existing points",
            time.perf_counter() - t0, len(existing_payloads),
        )

        unchanged, changed_embed, changed_payload_only, new = self._classify_chunks(
            unique_chunks, existing_payloads
        )
        logger.info(
            "upload_chunks: classify — unchanged=%d changed_embed=%d changed_payload_only=%d new=%d",
            len(unchanged), len(changed_embed), len(changed_payload_only), len(new),
        )
        if unique_chunks:
            sample = unique_chunks[0].metadata
            agent_log(
                location="ingestion_service.py:upload_chunks:classify",
                message="upload_chunks classified",
                data={
                    "source_id": sample.source_id,
                    "unchanged": len(unchanged),
                    "changed_embed": len(changed_embed),
                    "changed_payload_only": len(changed_payload_only),
                    "new": len(new),
                    "total": len(unique_chunks),
                },
                hypothesis_id="H5",
            )

        # Embed only changed + new
        to_embed = changed_embed + new
        embedding_batch = None

        # Strip formatting tags (<q>, <i>) and soft hyphens from embedding text to reduce
        # noise, while preserving the original text for storage and display.
        # Prepend passage prefix for instruction-tuned models (e.g. "passage: " for e5).
        from app.config import settings as _settings
        if prefix_passage is None:
            _passage_prefix: str = _settings.embedding_prefix_passage or ""
        else:
            _passage_prefix = prefix_passage
        texts = [
            (_passage_prefix + self._prepare_embedding_text(chunk.text))
            for chunk in to_embed
        ]
        embeddings: Sequence[Sequence[float]] = []
        vector_size = 0
        if to_embed:
            embed_batch_size = batch_size or self.default_batch_size
            t0 = time.perf_counter()
            logger.info("upload_chunks: embedding %d texts (batch_size=%d)…", len(texts), embed_batch_size)
            embedding_batch = await self.embedding_client.embed_texts(
                texts,
                model_name=embedding_model,
                batch_size=embed_batch_size,
            )
            logger.info("upload_chunks: embedding done in %.2fs", time.perf_counter() - t0)
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
                sparse_vector_name=(
                    SparseEmbedder.VECTOR_NAME if self.sparse_embedder is not None else None
                ),
            )
            await self.qdrant_client.ensure_text_index(collection, field_name="text")
            sparse_enabled = False
            if self.sparse_embedder is not None:
                sparse_enabled = await self.qdrant_client.ensure_sparse_config(collection)

            sparse_vectors = None
            if self.sparse_embedder is not None and sparse_enabled:
                sparse_vectors = self.sparse_embedder.embed_batch(texts)

            points = list(
                self._build_qdrant_points(to_embed, embeddings, sparse_vectors=sparse_vectors)
            )
            t0 = time.perf_counter()
            logger.info("upload_chunks: upserting %d points to Qdrant…", len(points))
            for i in range(0, len(points), embed_batch_size):
                await self.qdrant_client.upsert_points(
                    collection, points[i : i + embed_batch_size]
                )
            logger.info("upload_chunks: upsert done in %.2fs", time.perf_counter() - t0)

        # Unchanged chunks: content_hash matches → payload is already correct in Qdrant.
        # set_payload is skipped to avoid O(N) sequential Qdrant calls for large sources.
        if unchanged:
            logger.info("upload_chunks: %d unchanged chunks — skipping set_payload (hash match)", len(unchanged))

        # For payload-only changed chunks (e.g. chunk_type override), update payload without re-embedding.
        if changed_payload_only:
            payload_updates = []
            for chunk in changed_payload_only:
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
            t0 = time.perf_counter()
            logger.info(
                "upload_chunks: updating payload for %d points (no re-embed)…",
                len(payload_updates),
            )
            await self.qdrant_client.set_payload(collection, payload_updates)
            logger.info(
                "upload_chunks: payload-only updates done in %.2fs",
                time.perf_counter() - t0,
            )

        # vector_chunks mirror: write all changed/new rows (including payload-only changes)
        chunks_to_mirror = changed_embed + changed_payload_only + new
        if chunks_to_mirror:
            t0 = time.perf_counter()
            logger.info("upload_chunks: mirroring %d chunks to vector_chunks…", len(chunks_to_mirror))
            await self.vector_chunks_repository.upsert_chunks(collection, chunks_to_mirror)
            logger.info("upload_chunks: mirror done in %.2fs", time.perf_counter() - t0)

        # Cleanup stale chunk_ids for involved source_ids (sync-style), unless disabled
        stale_deleted = 0
        if not skip_cleanup:
            t0 = time.perf_counter()
            logger.info("upload_chunks: running _cleanup_stale…")
            _, stale_deleted = await self._cleanup_stale(
                collection,
                unique_chunks,
                active_ids_by_source_type=cleanup_active_ids,
            )
            logger.info("upload_chunks: _cleanup_stale done in %.2fs — %d stale deleted", time.perf_counter() - t0, stale_deleted)

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
            # "ingested" = wirklich geschrieben/überschrieben in Qdrant (embed or payload-only update).
            ingested=len(changed_embed) + len(changed_payload_only) + len(new),
            duplicates=duplicate_count,
            embedding_model=result_embedding_model,
            vector_size=result_vector_size,
            unchanged=len(unchanged),
            changed=len(changed_embed) + len(changed_payload_only),
            payload_changed=len(changed_payload_only),
            new=len(new),
            stale_deleted=stale_deleted,
        )
        logger.info(
            "upload_chunks: done in %.2fs — requested=%d ingested=%d unchanged=%d new=%d changed=%d payload_changed=%d stale_deleted=%d",
            time.perf_counter() - start_time,
            result.requested, result.ingested, result.unchanged, result.new, result.changed, result.payload_changed, result.stale_deleted,
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
            await self.vector_chunks_repository.delete_chunks(collection, chunk_ids)
        except Exception:
            pass

        return DeleteResult(
            collection=collection,
            requested=len(chunk_ids),
            deleted=len(chunk_ids),
        )

    async def cleanup_stale_vectors(
        self,
        collection: str,
        active_ids_by_source_type: Mapping[tuple[str, str], set[str]],
    ) -> int:
        """Remove Qdrant/mirror vectors not in the active rag_chunks id sets."""

        if not active_ids_by_source_type:
            return 0

        class _ScopeChunk:
            def __init__(self, source_id: str, chunk_type: str) -> None:
                self.metadata = type(
                    "Meta",
                    (),
                    {"source_id": source_id, "chunk_type": chunk_type, "chunk_id": ""},
                )()

        scope_chunks = [
            _ScopeChunk(source_id, chunk_type)
            for source_id, chunk_type in active_ids_by_source_type
        ]
        _, stale_deleted = await self._cleanup_stale(
            collection,
            scope_chunks,
            active_ids_by_source_type=active_ids_by_source_type,
        )
        return stale_deleted

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
    ) -> Tuple[List[ChunkRecord], List[ChunkRecord], List[ChunkRecord], List[ChunkRecord]]:
        """Return (unchanged, changed_embed, changed_payload_only, new) lists."""

        unchanged: List[ChunkRecord] = []
        changed_embed: List[ChunkRecord] = []
        changed_payload_only: List[ChunkRecord] = []
        new: List[ChunkRecord] = []

        for chunk in incoming:
            cid = chunk.metadata.chunk_id
            existing = existing_payloads.get(cid)
            if existing is None:
                new.append(chunk)
                continue
            prev_hash = existing.get("content_hash")
            prev_type = existing.get("chunk_type")
            if (
                isinstance(prev_hash, str)
                and prev_hash == chunk.metadata.content_hash
                and isinstance(prev_type, str)
                and prev_type == chunk.metadata.chunk_type
            ):
                if self._search_payload_unchanged(existing, chunk):
                    unchanged.append(chunk)
                else:
                    changed_payload_only.append(chunk)
            elif isinstance(prev_hash, str) and prev_hash == chunk.metadata.content_hash:
                changed_payload_only.append(chunk)
            else:
                changed_embed.append(chunk)

        return unchanged, changed_embed, changed_payload_only, new

    _SEARCH_PAYLOAD_KEYS = (
        "author",
        "book_title",
        "source_title",
        "segment_title",
        "source_type",
        "venue",
        "lecture_date",
        "parent_id",
        "lecture_id",
        "body_source_id",
        "paragraph_id",
    )

    def _search_payload_unchanged(
        self,
        existing: dict[str, object],
        chunk: ChunkRecord,
    ) -> bool:
        incoming = chunk.metadata.model_dump(mode="json")
        for key in self._SEARCH_PAYLOAD_KEYS:
            if existing.get(key) != incoming.get(key):
                return False
        return True

    def _build_qdrant_points(
        self,
        chunks: Sequence[ChunkRecord],
        embeddings: Sequence[Sequence[float]],
        *,
        sparse_vectors: list | None = None,
    ) -> Iterable[dict[str, object]]:
        """Convert chunk records into Qdrant point payloads."""

        for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            payload = chunk.metadata.model_dump(mode="json")
            payload["text"] = chunk.text
            payload["chunk_id"] = chunk.metadata.chunk_id
            payload["source_id"] = chunk.metadata.source_id
            payload["content_hash"] = chunk.metadata.content_hash

            point_uuid = uuid5(NAMESPACE_DNS, chunk.metadata.chunk_id)

            point: dict[str, object] = {
                "id": str(point_uuid),
                "payload": payload,
            }

            if sparse_vectors is not None and i < len(sparse_vectors):
                sv = sparse_vectors[i]
                point["vector"] = {
                    "": list(vector),
                    SparseEmbedder.VECTOR_NAME: sv,
                }
            else:
                point["vector"] = list(vector)

            yield point

    async def _cleanup_stale(
        self,
        collection: str,
        chunks: Sequence[ChunkRecord],
        *,
        active_ids_by_source_type: Mapping[tuple[str, str], set[str]] | None = None,
    ) -> tuple[int, int]:
        """Delete Qdrant/mirror chunk_ids not in the active set per (source_id, chunk_type).

        When ``active_ids_by_source_type`` is provided (embed-chunks), each group uses the
        full active rag_chunks id set so partial batches still remove historical orphans.
        Otherwise falls back to the ids delivered in this upload batch only.
        """

        if not chunks:
            return (0, 0)

        batch_ids_by_key: dict[tuple[str, str], set[str]] = {}
        for ch in chunks:
            key = (ch.metadata.source_id, ch.metadata.chunk_type)
            batch_ids_by_key.setdefault(key, set()).add(ch.metadata.chunk_id)

        total_stale = 0
        total_groups = 0

        for key, batch_ids in batch_ids_by_key.items():
            source_id, chunk_type = key
            if active_ids_by_source_type is not None:
                if key not in active_ids_by_source_type:
                    active_ids = batch_ids
                else:
                    active_ids = active_ids_by_source_type[key]
            else:
                active_ids = batch_ids

            total_groups += 1
            existing_points = await self.qdrant_client.scroll_all_points(
                collection,
                filter_=self._qdrant_filter_for_source_and_type(source_id, chunk_type),
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

            stale_ids = [cid for cid in existing_ids if cid not in active_ids]
            if not stale_ids:
                continue
            total_stale += len(stale_ids)
            point_uuids = [str(uuid5(NAMESPACE_DNS, cid)) for cid in stale_ids]
            await self.qdrant_client.delete_points(collection, point_uuids)
            try:
                await self.vector_chunks_repository.delete_chunks(collection, stale_ids)
            except Exception:
                pass

        return (total_groups, total_stale)

