"""FastAPI endpoints for RAG ingestion and deletion (ragprep-compatible)."""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from functools import lru_cache
from typing import Any, Dict, List, Mapping, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import func, select, text

from ..shared.models import ChunkRecord, CHUNK_TYPE_ENUM

from ..config import settings
from ..db.session import get_engine
from ..db.tables import event_metadata_table, vector_chunks_table, rag_talks_table, rag_turns_table
from ..core.providers import get_embedding_client, get_qdrant_client, get_sync_engine, get_sparse_embedder
from ..retrieval.services.providers import get_deepseek_chat, get_embedding_client as get_retrieval_embedding
from ..retrieval.services.quote_explain_service import explain_quote
from ..retrieval.services.event_recorder import EventRecorder, enqueue_record_event, enqueue_record_metadata_only
from ..core.telemetry import telemetry_client as ingestion_telemetry
from ..ingestion.repositories import RagChunksRepository, VectorChunksRepository
from ..ingestion.services import IngestionService
from ..infra.sparse_embedder import SparseEmbedder

router = APIRouter(prefix="/rag", tags=["rag"])


class StoreChunksRequest(BaseModel):
    """Persist chunks into rag_chunks (primary store) from JSONL."""

    chunks_jsonl_content: str = Field(
        ..., description="JSONL-formatted chunks (one JSON object per line)"
    )
    collection_name: str = Field(
        ...,
        description=(
            "rag_partition for this batch: assistant rag-collection, or "
            "reserved __shared__ for book/secondary_book corpus rows."
        ),
    )
    default_scope: Optional[str] = Field(
        None,
        description="Optional scope when chunk metadata has no source_type (e.g. book, assistant).",
    )


class StoreChunksResponse(BaseModel):
    """Response from store-chunks."""

    collection: str
    stored: int
    deprecated: int = Field(
        0,
        description="Number of rag_chunks rows marked deprecated (orphans not in this batch).",
    )
    deprecated_by_source: Dict[str, int] = Field(
        default_factory=dict,
        description="Per source_id: rows marked deprecated for that source in this request.",
    )


class EmbedChunksRequest(BaseModel):
    """Embed chunks for a Qdrant collection from rag_chunks into Qdrant and vector_chunks."""

    collection_name: str = Field(
        ...,
        description=(
            "Target Qdrant collection name (= assistant rag-collection). "
            "Loads assistant rag_partition rows in full plus a whitelist of __shared__ rows."
        ),
    )
    batch_size: Optional[int] = Field(
        None, ge=1, le=512, description="Embedding batch size"
    )
    embedding_model: Optional[str] = Field(
        None, description="Optional embedding model override"
    )
    skip_cleanup: bool = Field(
        False,
        description=(
            "If true, do not delete stale chunks during this embed run. "
            "Used by sync workflows that delete stale chunk_ids explicitly."
        ),
    )
    shared_source_ids: Optional[List[str]] = Field(
        None,
        description=(
            "Whitelist of source_id values for rows in rag_partition __shared__. "
            "Omit or null to include all shared rows (legacy). "
            "Pass an empty list to embed only the assistant partition."
        ),
    )
    source_ids: Optional[List[str]] = Field(
        None,
        description=(
            "Optional filter: only embed rows whose source_id is in this list. "
            "Applies to BOTH the assistant partition and __shared__. "
            "Omit or null to embed all rows (default behaviour). "
            "Use in combination with shared_source_ids for fine-grained per-source iteration."
        ),
    )


class UploadChunksResponse(BaseModel):
    """Response from embed-chunks endpoint (ingestion stats)."""

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


class DeleteChunksRequest(BaseModel):
    """Request to delete chunks by filter or delete all."""

    all: bool = Field(False, description="Delete all chunks in collection")
    filter: Optional[Dict[str, Any]] = Field(
        None, description="Metadata filter (e.g., {'book_id': '123'})"
    )
    collection_name: Optional[str] = Field(None, description="Target collection name")
    dry_run: bool = Field(False, description="Preview deletion without executing")
    limit: Optional[int] = Field(
        None, description="Safety limit on number of chunks to delete"
    )


class DeleteChunksResponse(BaseModel):
    """Response from delete-chunks endpoint."""

    collection: str
    matched: int
    deleted: int
    dry_run: bool


class ListChunksRequest(BaseModel):
    """Request to list chunk inventory for a single source_id."""

    collection_name: str = Field(..., description="Target collection name")
    source_id: str = Field(..., description="Source identifier to inventory")
    limit: int = Field(100000, ge=1, le=500000, description="Max number of chunks to return")


class ListedChunk(BaseModel):
    chunk_id: str
    content_hash: Optional[str] = None
    updated_at: Optional[str] = None
    chunk_type: Optional[str] = None


class ListChunksResponse(BaseModel):
    """Inventory response for a single source_id."""

    collection: str
    source_id: str
    chunks: List[ListedChunk]


class DeleteChunkIdsRequest(BaseModel):
    """Request to delete explicit chunk_ids (sync workflow)."""

    collection_name: str = Field(..., description="Target collection name")
    chunk_ids: List[str] = Field(..., description="Chunk IDs to delete")
    dry_run: bool = Field(False, description="Preview deletion without executing")
    limit: Optional[int] = Field(None, description="Safety limit on number of chunks to delete")


class DeleteChunkIdsResponse(BaseModel):
    collection: str
    matched: int
    deleted: int
    dry_run: bool


def get_ingestion_service() -> IngestionService:
    """Lazy singleton used as a FastAPI dependency."""
    return _get_ingestion_service()


@lru_cache(maxsize=1)
def _get_ingestion_service() -> IngestionService:
    embedding_client = get_embedding_client(batch_size=64)
    qdrant_client = get_qdrant_client()
    vector_chunks_repository = VectorChunksRepository(get_sync_engine())
    return IngestionService(
        embedding_client=embedding_client,
        qdrant_client=qdrant_client,
        vector_chunks_repository=vector_chunks_repository,
        telemetry_client=ingestion_telemetry,
        sparse_embedder=get_sparse_embedder() if settings.use_hybrid_retrieval else None,
        default_batch_size=64,
    )


@lru_cache(maxsize=1)
def _get_rag_chunks_repository() -> RagChunksRepository:
    return RagChunksRepository(get_sync_engine())


def get_rag_chunks_repository() -> RagChunksRepository:
    return _get_rag_chunks_repository()


def _parse_jsonl_chunks(content: str) -> List[ChunkRecord]:
    lines = content.strip().split("\n")
    chunks: List[ChunkRecord] = []

    for line_no, line in enumerate(lines, start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            chunk_dict = json.loads(line)
            chunk = ChunkRecord.from_dict(chunk_dict)
            chunks.append(chunk)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid JSONL at line {line_no}: {exc}",
            ) from exc
    return chunks


@router.post(
    "/store-chunks",
    response_model=StoreChunksResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def store_chunks(request: StoreChunksRequest) -> StoreChunksResponse:
    """Store chunks in rag_chunks (primary DB). Does not embed."""

    chunks = _parse_jsonl_chunks(request.chunks_jsonl_content)
    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid chunks found in JSONL content",
        )

    rag_repo = get_rag_chunks_repository()
    await rag_repo.upsert_chunks(
        request.collection_name,
        chunks,
        default_scope=request.default_scope,
    )
    active_by_source: Dict[str, List[str]] = {}
    for c in chunks:
        sid = c.metadata.source_id
        active_by_source.setdefault(sid, []).append(c.metadata.chunk_id)
    deprecated_by_source = await rag_repo.deprecate_orphans_for_sources(
        request.collection_name, active_by_source
    )
    deprecated = sum(deprecated_by_source.values())
    return StoreChunksResponse(
        collection=request.collection_name,
        stored=len(chunks),
        deprecated=deprecated,
        deprecated_by_source=deprecated_by_source,
    )


@router.post(
    "/embed-chunks",
    response_model=UploadChunksResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def embed_chunks(
    request: EmbedChunksRequest,
    service: IngestionService = Depends(get_ingestion_service),
) -> UploadChunksResponse:
    """Read all chunks for a collection from rag_chunks, embed into Qdrant, mirror vector_chunks."""

    rag_repo = get_rag_chunks_repository()
    chunks = await rag_repo.list_chunk_records_for_embed(
        request.collection_name,
        shared_source_ids=request.shared_source_ids,
        source_ids=request.source_ids,
    )

    if not chunks:
        return UploadChunksResponse(
            ingestion_id=str(uuid.uuid4()),
            collection=request.collection_name,
            requested=0,
            ingested=0,
            duplicates=0,
            embedding_model=request.embedding_model or "skipped",
            vector_size=0,
            unchanged=0,
            changed=0,
            new=0,
            stale_deleted=0,
        )

    try:
        result = await service.upload_chunks(
            collection=request.collection_name,
            chunks=chunks,
            embedding_model=request.embedding_model,
            batch_size=request.batch_size,
            skip_cleanup=bool(request.skip_cleanup),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)
        ) from exc

    await rag_repo.mark_embedded_for_embed_run(
        request.collection_name,
        [c.metadata.chunk_id for c in chunks],
    )

    chunk_ids = [c.id for c in chunks] if chunks else None
    enqueue_record_metadata_only(
        EventRecorder(),
        endpoint="rag/embed-chunks",
        collection=request.collection_name,
        metadata={
            "requested": result.requested,
            "ingested": result.ingested,
            "stale_deleted": result.stale_deleted,
            "duplicates": result.duplicates,
            "unchanged": result.unchanged,
            "changed": result.changed,
            "new": result.new,
        },
        chunk_ids=chunk_ids[:500] if chunk_ids and len(chunk_ids) > 500 else chunk_ids,
    )

    return UploadChunksResponse(**asdict(result))


def _qdrant_filter_for_source(source_id: str) -> dict[str, object]:
    return {"must": [{"key": "source_id", "match": {"value": source_id}}]}


def _qdrant_filter_from_kv_filter(filter_: Dict[str, Any]) -> dict[str, object]:
    """Convert a simple key/value metadata filter into a Qdrant filter.

    Note: This intentionally supports only the ragprep-style filter shape:
    {"some_key": "some_value", "other_key": 123}. For more complex filter DSLs,
    callers should use /delete-chunk-ids instead.
    """

    must: list[dict[str, object]] = []
    for key, value in (filter_ or {}).items():
        if not isinstance(key, str) or not key.strip():
            continue
        # Qdrant payload values are typed; we store strings in ragprep/ragrun metadata,
        # so stringify to match existing ingestion and mirror conventions.
        must.append({"key": key, "match": {"value": str(value)}})
    return {"must": must} if must else {}


async def _qdrant_chunk_ids_for_filter(
    *,
    qdrant_client: QdrantClient,
    collection: str,
    qdrant_filter: Mapping[str, object] | None,
    limit: int | None,
) -> list[str]:
    """Scroll Qdrant and return chunk_ids from payload (best-effort)."""

    out: list[str] = []
    offset: object | None = None
    page_size = 512
    max_pages = 10_000

    for _ in range(max_pages):
        points, offset = await qdrant_client.scroll_points_page(
            collection,
            filter_=qdrant_filter,
            limit=page_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break
        for p in points:
            payload = p.get("payload", {}) if isinstance(p, dict) else {}
            if not isinstance(payload, dict):
                continue
            cid = payload.get("chunk_id")
            if isinstance(cid, str) and cid.strip():
                out.append(cid)
                if limit is not None and len(out) > limit:
                    return out
        if offset is None:
            break

    return out


@router.post("/list-chunks", response_model=ListChunksResponse)
async def list_chunks(request: ListChunksRequest) -> ListChunksResponse:
    """List minimal chunk inventory for a (collection, source_id) from Qdrant."""

    collection = request.collection_name or "default"
    source_id = (request.source_id or "").strip()
    if not source_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="source_id must not be empty"
        )

    qdrant_client = get_qdrant_client()

    out: list[ListedChunk] = []
    offset: object | None = None
    remaining = int(request.limit)
    page_size = min(512, remaining)
    max_pages = 10_000

    for _ in range(max_pages):
        if remaining <= 0:
            break
        points, offset = await qdrant_client.scroll_points_page(
            collection,
            filter_=_qdrant_filter_for_source(source_id),
            limit=min(page_size, remaining),
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break
        for p in points:
            payload = p.get("payload", {}) if isinstance(p, dict) else {}
            if not isinstance(payload, dict):
                continue
            cid = payload.get("chunk_id")
            if not isinstance(cid, str) or not cid:
                continue
            out.append(
                ListedChunk(
                    chunk_id=cid,
                    content_hash=payload.get("content_hash")
                    if isinstance(payload.get("content_hash"), str)
                    else None,
                    updated_at=payload.get("updated_at")
                    if isinstance(payload.get("updated_at"), str)
                    else None,
                    chunk_type=payload.get("chunk_type")
                    if isinstance(payload.get("chunk_type"), str)
                    else None,
                )
            )
        remaining = int(request.limit) - len(out)
        if offset is None:
            break

    return ListChunksResponse(collection=collection, source_id=source_id, chunks=out)


@router.post("/delete-chunk-ids", response_model=DeleteChunkIdsResponse)
async def delete_chunk_ids(
    request: DeleteChunkIdsRequest,
    service: IngestionService = Depends(get_ingestion_service),
) -> DeleteChunkIdsResponse:
    """Delete explicit chunk_ids (sync-safe) from Qdrant and best-effort mirror."""

    collection = request.collection_name or "default"
    chunk_ids = [cid for cid in request.chunk_ids if isinstance(cid, str) and cid.strip()]

    if request.limit is not None and len(chunk_ids) > request.limit:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Matched {len(chunk_ids)} chunks, exceeds safety limit of {request.limit}",
        )

    if request.dry_run:
        return DeleteChunkIdsResponse(
            collection=collection, matched=len(chunk_ids), deleted=0, dry_run=True
        )

    if not chunk_ids:
        return DeleteChunkIdsResponse(
            collection=collection, matched=0, deleted=0, dry_run=False
        )

    # Use the same deletion path as ingestion (UUIDv5 ids).
    try:
        result = await service.delete_chunks(collection=collection, chunk_ids=chunk_ids)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    try:
        await get_rag_chunks_repository().delete_chunks(collection, chunk_ids)
    except Exception:
        pass

    return DeleteChunkIdsResponse(
        collection=collection,
        matched=len(chunk_ids),
        deleted=result.deleted,
        dry_run=False,
    )


@router.post("/delete-chunks", response_model=DeleteChunksResponse)
async def delete_chunks(
    request: DeleteChunksRequest,
    service: IngestionService = Depends(get_ingestion_service),
) -> DeleteChunksResponse:
    """Delete chunks by metadata filter or delete all (ragprep-compatible endpoint)."""

    collection = request.collection_name or "default"

    if request.dry_run:
        # Preview deletion by counting matching chunks
        engine = get_engine()
        query = select(func.count()).select_from(vector_chunks_table).where(
            vector_chunks_table.c.collection == collection
        )

        if not request.all and request.filter:
            # Build filter conditions
            for key, value in request.filter.items():
                # Use JSONB containment for metadata fields
                query = query.where(
                    vector_chunks_table.c.metadata[key].as_string() == str(value)
                )
        elif not request.all:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Must provide either 'all=true' or 'filter' parameter",
            )

        with engine.connect() as conn:
            matched = conn.execute(query).scalar() or 0

        # Fallback: mirror may be missing/out-of-date while Qdrant contains points.
        # If mirror count is 0, estimate via Qdrant scroll (bounded by limit).
        if matched == 0:
            # Best-effort only: dry-run should never fail just because Qdrant is down.
            try:
                qdrant_client = get_qdrant_client()
                if request.all:
                    # Without a limit, this could be very expensive.
                    if request.limit is None:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=(
                                "dry_run with all=true requires 'limit' when mirror is empty "
                                "(needed to bound Qdrant scan)"
                            ),
                        )
                    qdrant_ids = await _qdrant_chunk_ids_for_filter(
                        qdrant_client=qdrant_client,
                        collection=collection,
                        qdrant_filter=None,
                        limit=request.limit,
                    )
                    matched = len(qdrant_ids)
                elif request.filter:
                    qdrant_filter = _qdrant_filter_from_kv_filter(request.filter)
                    qdrant_ids = await _qdrant_chunk_ids_for_filter(
                        qdrant_client=qdrant_client,
                        collection=collection,
                        qdrant_filter=qdrant_filter if qdrant_filter else None,
                        limit=request.limit,
                    )
                    matched = len(qdrant_ids)
            except HTTPException:
                raise
            except Exception:
                # Keep matched=0 on any connectivity/scroll error.
                pass

        if request.limit and matched > request.limit:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Matched {matched} chunks, exceeds safety limit of {request.limit}",
            )

        return DeleteChunksResponse(
            collection=collection, matched=matched, deleted=0, dry_run=True
        )

    # Actual deletion: query Postgres for matching chunk_ids
    engine = get_engine()
    query = select(vector_chunks_table.c.chunk_id).where(
        vector_chunks_table.c.collection == collection
    )

    if not request.all and request.filter:
        for key, value in request.filter.items():
            query = query.where(vector_chunks_table.c.metadata[key].as_string() == str(value))
    elif not request.all:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must provide either 'all=true' or 'filter' parameter",
        )

    with engine.connect() as conn:
        result_rows = conn.execute(query).fetchall()
        chunk_ids = [row[0] for row in result_rows]

    # Fallback: if mirror has no rows but Qdrant still has points, delete via Qdrant filter.
    if not chunk_ids:
        qdrant_client = get_qdrant_client()
        if request.all:
            # Deleting "all" without a mirror can be dangerously expensive; require an explicit limit.
            if request.limit is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        "Mirror returned 0 rows for all=true; refusing to delete-all from Qdrant "
                        "without an explicit --limit"
                    ),
                )
            try:
                qdrant_ids = await _qdrant_chunk_ids_for_filter(
                    qdrant_client=qdrant_client,
                    collection=collection,
                    qdrant_filter=None,
                    limit=request.limit,
                )
                chunk_ids = qdrant_ids
            except Exception as exc:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Failed to scan Qdrant for delete-all fallback: {exc}",
                ) from exc
        elif request.filter:
            qdrant_filter = _qdrant_filter_from_kv_filter(request.filter)
            try:
                qdrant_ids = await _qdrant_chunk_ids_for_filter(
                    qdrant_client=qdrant_client,
                    collection=collection,
                    qdrant_filter=qdrant_filter if qdrant_filter else None,
                    limit=request.limit,
                )
                chunk_ids = qdrant_ids
            except Exception as exc:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Failed to scan Qdrant for delete fallback: {exc}",
                ) from exc

    if request.limit and len(chunk_ids) > request.limit:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Matched {len(chunk_ids)} chunks, exceeds safety limit of {request.limit}",
        )

    if not chunk_ids:
        return DeleteChunksResponse(
            collection=collection, matched=0, deleted=0, dry_run=False
        )

    try:
        result = await service.delete_chunks(
            collection=collection,
            chunk_ids=chunk_ids,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    try:
        await get_rag_chunks_repository().delete_chunks(collection, chunk_ids)
    except Exception:
        pass

    enqueue_record_metadata_only(
        EventRecorder(),
        endpoint="rag/delete-chunks",
        collection=collection,
        metadata={"matched": len(chunk_ids), "deleted": result.deleted},
    )

    return DeleteChunksResponse(
        collection=collection,
        matched=len(chunk_ids),
        deleted=result.deleted,
        dry_run=False,
    )


@router.get("/books/titles")
async def list_book_titles(
    collection_name: Optional[str] = None,
    min_count: int = 1,
    limit: int = 100,
    include_author: bool = True,
    chunk_types: Optional[str] = None,
) -> Dict[str, Any]:
    """List distinct book titles in a collection with chunk counts."""

    collection = collection_name or "default"
    engine = get_engine()

    # Parse chunk_types as comma-separated list if provided
    types_filter: Optional[list[str]] = None
    if chunk_types:
        types_filter = [t.strip() for t in chunk_types.split(",") if t.strip()]
        if not types_filter:
            types_filter = None

    # Query for distinct book_title with counts (includes source_id and lecture_date for context selection)
    if include_author:
        # Prefer explicit book_title; fall back to source_title for older ingestions
        query = text(
            """
            SELECT
                metadata->>'chunk_type' as chunk_type,
                metadata->>'author' as author,
                COALESCE(metadata->>'book_title', metadata->>'source_title') as book_title,
                source_id,
                MIN(metadata->>'lecture_date') as lecture_date,
                COUNT(*) as count
            FROM vector_chunks
            WHERE collection = :collection
              AND COALESCE(metadata->>'book_title', metadata->>'source_title') IS NOT NULL
              {chunk_filter}
            GROUP BY metadata->>'chunk_type', metadata->>'author', COALESCE(metadata->>'book_title', metadata->>'source_title'), source_id
            HAVING COUNT(*) >= :min_count
            ORDER BY count DESC, book_title
            LIMIT :limit
        """
        )
    else:
        query = text(
            """
            SELECT
                metadata->>'chunk_type' as chunk_type,
                COALESCE(metadata->>'book_title', metadata->>'source_title') as book_title,
                source_id,
                MIN(metadata->>'lecture_date') as lecture_date,
                COUNT(*) as count
            FROM vector_chunks
            WHERE collection = :collection
              AND COALESCE(metadata->>'book_title', metadata->>'source_title') IS NOT NULL
              {chunk_filter}
            GROUP BY metadata->>'chunk_type', COALESCE(metadata->>'book_title', metadata->>'source_title'), source_id
            HAVING COUNT(*) >= :min_count
            ORDER BY count DESC, book_title
            LIMIT :limit
        """
        )

    chunk_filter_sql = ""
    params: Dict[str, Any] = {"collection": collection, "min_count": min_count, "limit": limit}
    if types_filter:
        chunk_filter_sql = "AND metadata->>'chunk_type' = ANY(:chunk_types)"
        params["chunk_types"] = types_filter
    # Inject the optional filter into the query text
    query = text(query.text.replace("{chunk_filter}", chunk_filter_sql))

    with engine.connect() as conn:
        rows = conn.execute(query, params).fetchall()

    if include_author:
        titles = [
            {
                "chunk_type": row[0],
                "author": row[1],
                "book_title": row[2],
                "source_id": row[3],
                "lecture_date": row[4],
                "count": row[5],
            }
            for row in rows
        ]
    else:
        titles = [
            {
                "chunk_type": row[0],
                "book_title": row[1],
                "source_id": row[2],
                "lecture_date": row[3],
                "count": row[4],
            }
            for row in rows
        ]

    return {
        "collection": collection,
        "total_distinct_titles": len(titles),
        "titles": titles,
    }


@router.get("/books/chapters")
async def list_book_chapters(
    collection_name: Optional[str] = None,
    source_id: Optional[str] = None,
) -> Dict[str, Any]:
    """List distinct chapters (segments) for a book/talk source, ordered by reading position."""

    collection = collection_name or "default"
    source_id = (source_id or "").strip()
    if not source_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="source_id must not be empty"
        )

    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT
                    metadata->>'segment_id'    AS segment_id,
                    metadata->>'segment_title' AS segment_title,
                    COUNT(*)                   AS chunk_count,
                    MIN((metadata->>'source_index')::int) AS min_source_index
                FROM vector_chunks
                WHERE collection = :collection
                  AND source_id = :source_id
                  AND metadata->>'segment_id' IS NOT NULL
                  AND metadata->>'segment_id' != ''
                GROUP BY metadata->>'segment_id', metadata->>'segment_title'
                ORDER BY min_source_index NULLS LAST
                """
            ),
            {"collection": collection, "source_id": source_id},
        ).fetchall()

    chapters = [
        {
            "segment_id": row[0],
            "segment_title": row[1],
            "chunk_count": row[2],
        }
        for row in rows
    ]
    return {"collection": collection, "source_id": source_id, "chapters": chapters}


@router.get("/books/context-chunks")
async def get_context_chunks(
    collection_name: Optional[str] = None,
    source_id: Optional[str] = None,
    segment_id: Optional[str] = None,
    paragraph: Optional[int] = None,
) -> Dict[str, Any]:
    """Fetch chunk texts for a selected book/chapter/paragraph context."""

    collection = collection_name or "default"
    source_id = (source_id or "").strip()
    if not source_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="source_id must not be empty"
        )

    engine = get_engine()

    def _fetch(seg_id: Optional[str], para: Optional[int]) -> list:
        conditions = [
            "collection = :collection",
            "source_id = :source_id",
        ]
        params: Dict[str, Any] = {"collection": collection, "source_id": source_id}
        if seg_id:
            conditions.append("metadata->>'segment_id' = :segment_id")
            params["segment_id"] = seg_id
        if para is not None:
            # paragraph_numbers is not stored in metadata; paragraph markers are embedded in
            # the chunk text as "NN| ..." at the start or after "\n\n" between paragraphs.
            params["para_regex"] = f"(^|\\n\\n){para}\\|"
            conditions.append("text ~ :para_regex")
        where = " AND ".join(conditions)
        with engine.connect() as conn:
            return conn.execute(
                text(
                    f"""
                    SELECT
                        chunk_id,
                        text,
                        metadata->>'segment_id'    AS segment_id,
                        metadata->>'segment_title' AS segment_title,
                        '[]'::jsonb AS paragraph_numbers,
                        (metadata->>'source_index')::int AS source_index
                    FROM vector_chunks
                    WHERE {where}
                    ORDER BY source_index NULLS LAST
                    """
                ),
                params,
            ).fetchall()

    rows = _fetch(segment_id, paragraph)
    fallback_used = False

    # Fallback: paragraph not found → entire chapter/source
    if not rows and paragraph is not None:
        rows = _fetch(segment_id, None)
        fallback_used = True

    chunks = [
        {
            "chunk_id": row[0],
            "text": row[1],
            "segment_id": row[2],
            "segment_title": row[3],
            "paragraph_numbers": row[4] if row[4] is not None else [],
            "source_index": row[5],
        }
        for row in rows
    ]
    return {
        "collection": collection,
        "source_id": source_id,
        "segment_id": segment_id,
        "paragraph": paragraph,
        "chunks": chunks,
        "fallback_used": fallback_used,
    }


@router.get("/monitoring/chunks")
async def monitoring_chunks(
    collection: str,
) -> Dict[str, Any]:
    """Chunk statistics for the monitoring widget. Returns all known chunk types (including 0 count)."""

    engine = get_engine()
    with engine.connect() as conn:
        types_rows = conn.execute(
            text(
                """
                SELECT
                    chunk_type,
                    COUNT(*) AS count,
                    ROUND(SUM(octet_length(COALESCE(text, ''))) / 1048576.0, 2) AS text_mb,
                    MIN(updated_at) AS oldest,
                    MAX(updated_at) AS newest
                FROM vector_chunks
                WHERE collection = :c
                GROUP BY chunk_type
                """
            ),
            {"c": collection},
        ).fetchall()

        # Merge with full CHUNK_TYPE_ENUM so all types are shown (0 for missing)
        type_stats = {r[0]: r for r in types_rows}
        chunk_types_result = []
        for t in CHUNK_TYPE_ENUM:
            r = type_stats.get(t)
            chunk_types_result.append({
                "chunk_type": t,
                "count": r[1] if r else 0,
                "text_mb": float(r[2]) if r else 0.0,
                "oldest": r[3].date().isoformat() if r and r[3] else None,
                "newest": r[4].date().isoformat() if r and r[4] else None,
            })
        # Add any DB types not in enum (legacy/custom)
        for t, r in type_stats.items():
            if t not in CHUNK_TYPE_ENUM:
                chunk_types_result.append({
                    "chunk_type": t,
                    "count": r[1],
                    "text_mb": float(r[2]),
                    "oldest": r[3].date().isoformat() if r[3] else None,
                    "newest": r[4].date().isoformat() if r[4] else None,
                })
        chunk_types_result.sort(key=lambda x: -x["count"])

        # Books with chunk count + usage count from event_metadata + links from event_content
        books_rows = conn.execute(
            text(
                """
                WITH usage AS (
                    SELECT cid AS chunk_id, COUNT(*) AS cnt
                    FROM event_metadata em,
                         jsonb_array_elements_text(em.chunk_ids) AS cid
                    WHERE (em.collection = :c OR em.collection IS NULL)
                      AND em.chunk_ids IS NOT NULL
                      AND jsonb_typeof(em.chunk_ids) = 'array'
                    GROUP BY cid
                ),
                links AS (
                    SELECT cid AS chunk_id, COUNT(*) AS cnt
                    FROM event_content ec
                    JOIN event_metadata em ON em.id = ec.event_metadata_id,
                         jsonb_array_elements_text(ec.context_refs) AS cid
                    WHERE (em.collection = :c OR em.collection IS NULL)
                      AND ec.context_refs IS NOT NULL
                      AND jsonb_typeof(ec.context_refs) = 'array'
                    GROUP BY cid
                ),
                books_base AS (
                    SELECT
                        COALESCE(metadata->>'book_title', metadata->>'source_title') AS book_title,
                        metadata->>'author' AS author,
                        chunk_type,
                        COUNT(*) AS chunk_count,
                        COALESCE(SUM(u.cnt), 0)::bigint AS usage_count,
                        COALESCE(SUM(lnk.cnt), 0)::bigint AS links_count
                    FROM vector_chunks rc
                    LEFT JOIN usage u ON u.chunk_id = rc.chunk_id
                    LEFT JOIN links lnk ON lnk.chunk_id = rc.chunk_id
                    WHERE rc.collection = :c
                      AND COALESCE(rc.metadata->>'book_title', rc.metadata->>'source_title') IS NOT NULL
                    GROUP BY 1, 2, 3
                )
                SELECT book_title, author, chunk_type, chunk_count, usage_count, links_count
                FROM books_base
                ORDER BY chunk_count DESC
                LIMIT 100
                """
            ),
            {"c": collection},
        ).fetchall()

        total_usage = sum(r[4] for r in books_rows)

    return {
        "collection": collection,
        "chunk_types": chunk_types_result,
        "books": [
            {
                "book_title": r[0],
                "author": r[1],
                "chunk_type": r[2],
                "count": r[3],
                "usage_count": r[4],
                "usage_pct": round(100.0 * r[4] / total_usage, 1) if total_usage > 0 else 0.0,
                "links_count": r[5],
            }
            for r in books_rows
        ],
        "total_usage": total_usage,
    }


@router.get("/monitoring/events")
async def monitoring_events(
    collection: str,
    limit: int = 50,
) -> Dict[str, Any]:
    """Event statistics and log for the monitoring widget."""

    engine = get_engine()
    with engine.connect() as conn:
        # Include events where collection matches OR is NULL (untyped events)
        volume_rows = conn.execute(
            text(
                """
                SELECT endpoint, COUNT(*) AS event_count
                FROM event_metadata
                WHERE collection = :c OR collection IS NULL
                GROUP BY endpoint
                ORDER BY event_count DESC
                """
            ),
            {"c": collection},
        ).fetchall()

        # Log: one row per event_metadata AND one row per event_content (unified)
        log_rows = conn.execute(
            text(
                """
                (
                    SELECT
                        em.created_at,
                        em.endpoint,
                        CASE
                            WHEN em.chunk_ids IS NOT NULL AND jsonb_typeof(em.chunk_ids) = 'array'
                            THEN jsonb_array_length(em.chunk_ids)
                            ELSE 0
                        END AS chunk_count,
                        NULL::text AS concept,
                        'metadata' AS source
                    FROM event_metadata em
                    WHERE (em.collection = :c OR em.collection IS NULL)
                )
                UNION ALL
                (
                    SELECT
                        ec.created_at,
                        em.endpoint,
                        NULL::bigint AS chunk_count,
                        ec.concept,
                        'content' AS source
                    FROM event_content ec
                    JOIN event_metadata em ON em.id = ec.event_metadata_id
                    WHERE (em.collection = :c OR em.collection IS NULL)
                )
                ORDER BY created_at DESC
                LIMIT :lim
                """
            ),
            {"c": collection, "lim": limit},
        ).fetchall()

    return {
        "collection": collection,
        "volume": [{"endpoint": r[0], "event_count": r[1]} for r in volume_rows],
        "log": [
            {
                "endpoint": r[1],
                "created_at": r[0].isoformat() if r[0] else None,
                "chunk_count": r[2],
                "concept": r[3],
                "source": r[4],
            }
            for r in log_rows
        ],
    }


def _qdrant_sparse_non_empty(vec: object) -> bool:
    """True if Qdrant sparse payload has at least one index or value."""
    if vec is None:
        return False
    if isinstance(vec, dict):
        idx = vec.get("indices")
        if isinstance(idx, list) and len(idx) > 0:
            return True
        vals = vec.get("values")
        if isinstance(vals, list) and len(vals) > 0:
            return True
    return False


def _qdrant_point_sparse_vector(point_vector: object, name: str) -> object | None:
    """Resolve named sparse vector from a point ``vector`` field (multi-vector map)."""
    if not isinstance(point_vector, dict):
        return None
    if name in point_vector:
        return point_vector.get(name)
    # Single-vector response: indices/values at top level
    if "indices" in point_vector or "values" in point_vector:
        return point_vector
    return None


@router.get("/collections/{collection_name}/verify-sparse")
async def verify_collection_sparse(
    collection_name: str,
    sample_limit: int = Query(20, ge=1, le=256, description="Max points to scroll for sampling"),
) -> Dict[str, Any]:
    """Check collection config for sparse slot and sample points for non-empty BM25 vectors.

    Used by migration / ``rag:embed --verify-qdrant`` to confirm data in Qdrant matches
    Postgres monitoring counts and that ``text-sparse`` is populated.
    """

    qdrant_client = get_qdrant_client()
    sparse_name = SparseEmbedder.VECTOR_NAME
    info = await qdrant_client.get_collection_info(collection_name)
    if info is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Qdrant collection not found: {collection_name}",
        )

    params = info.get("config", {}) if isinstance(info, Mapping) else {}
    if isinstance(params, dict):
        inner = params.get("params", {})
        params = inner if isinstance(inner, dict) else {}
    sparse_cfg: dict[str, object] = {}
    if isinstance(params, dict):
        raw_sv = params.get("sparse_vectors")
        if isinstance(raw_sv, dict):
            sparse_cfg = raw_sv
    sparse_slot_configured = sparse_name in sparse_cfg
    points_count = int(info.get("points_count", 0) or 0)

    sample_size = 0
    sample_with_non_empty_sparse = 0
    issues: list[str] = []

    if points_count == 0:
        issues.append("collection_is_empty")
    elif not sparse_slot_configured:
        issues.append("sparse_slot_missing_in_schema")
    else:
        take = min(sample_limit, points_count, 256)
        points, _ = await qdrant_client.scroll_points_page(
            collection_name,
            limit=take,
            with_payload=False,
            with_vector_names=[sparse_name],
        )
        sample_size = len(points)
        for p in points:
            raw_v = p.get("vector") if isinstance(p, Mapping) else None
            sp = _qdrant_point_sparse_vector(raw_v, sparse_name)
            if _qdrant_sparse_non_empty(sp):
                sample_with_non_empty_sparse += 1
        if sample_size > 0 and sample_with_non_empty_sparse < sample_size:
            issues.append(
                f"sparse_incomplete:{sample_with_non_empty_sparse}/{sample_size}",
            )

    hybrid = bool(getattr(settings, "use_hybrid_retrieval", False))
    ok = (
        sparse_slot_configured
        and points_count > 0
        and sample_size > 0
        and sample_with_non_empty_sparse == sample_size
        and hybrid
    )
    if not hybrid:
        issues.append("RAGRUN_USE_HYBRID_RETRIEVAL_not_enabled_ingestion_may_skip_sparse")

    return {
        "collection": collection_name,
        "points_count": points_count,
        "sparse_vector_name": sparse_name,
        "sparse_slot_configured": sparse_slot_configured,
        "hybrid_retrieval_enabled": hybrid,
        "sample_size": sample_size,
        "sample_with_non_empty_sparse": sample_with_non_empty_sparse,
        "issues": issues,
        "ok": ok,
    }


@router.get("/collections")
async def list_collections() -> Dict[str, Any]:
    """List all collections from Qdrant."""

    qdrant_client = get_qdrant_client()

    collections = await qdrant_client.list_collections()

    return {"collections": collections, "vector_db_path": str(settings.qdrant_url)}


class QuoteExplainRequest(BaseModel):
    """Request for quote explanation."""

    quote: str = Field(..., min_length=1, description="The quote to explain")
    assistant: Optional[str] = Field(
        "philo-von-freisinn",
        description="Assistant name (from assistant-manifest.yaml)",
    )
    language: Optional[str] = Field(
        None,
        description="BCP 47 locale (e.g. en-US, de-DE). Default: de-DE.",
    )


class QuoteExplainResponse(BaseModel):
    """Response: chunk-shaped object with text and metadata."""

    text: str
    metadata: Dict[str, Any]


@router.post("/quote-explain", response_model=QuoteExplainResponse)
async def quote_explain(request: QuoteExplainRequest) -> QuoteExplainResponse:
    """
    Explain a quote using retrieval from primary book + lecture and DeepSeek.

    Retrieves k=8 chunks (4 book + 4 lecture), generates ~600-token explanation,
    evaluates chunk relevance, returns chunk-shaped object.
    """
    quote = (request.quote or "").strip()
    if not quote:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="quote is required",
        )
    assistant = (request.assistant or "philo-von-freisinn").strip()
    language = (request.language or "de-DE").strip()

    try:
        result = await explain_quote(
            quote=quote,
            assistant=assistant,
            language=language,
            embedding_client=get_retrieval_embedding(),
            qdrant_client=get_qdrant_client(),
            chat_client=get_deepseek_chat(),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"quote-explain failed: {exc}",
        ) from exc

    graph_event_id = str(uuid.uuid4())
    enqueue_record_event(
        EventRecorder(),
        endpoint="rag/quote-explain",
        graph_event_id=graph_event_id,
        graph_name="quote_explain",
        step="explain",
        collection=result.get("collection"),
        chunk_ids=result.get("chunk_ids"),
        concept=None,
        query_text=quote[:512] if quote else None,
        response_text=result["text"][:2000] if result.get("text") else None,
        context_refs=[r.get("chunk_id") for r in result.get("metadata", {}).get("references", []) if isinstance(r.get("chunk_id"), str)],
    )

    return QuoteExplainResponse(text=result["text"], metadata=result["metadata"])


# ---------------------------------------------------------------------------
# Published Talks – für assistant:chunk in ragprep
# ---------------------------------------------------------------------------

class PublishedTurn(BaseModel):
    turn_index: int
    user_message: str
    assistant_message: str


class PublishedTalk(BaseModel):
    talk_id: str
    slug: str
    title: str
    summary: Optional[str] = None
    mensch_name: str
    turns: List[PublishedTurn]


class PublishedTalksResponse(BaseModel):
    collection: str
    count: int
    talks: List[PublishedTalk]


@router.get(
    "/talks/published",
    response_model=PublishedTalksResponse,
    summary="Published talks with turns for RAG chunking",
)
async def get_published_talks(
    collection: str = Query(..., description="Qdrant collection / rag-collection name"),
) -> PublishedTalksResponse:
    """Return all talks with publishing_status='published' for the given collection,
    including their turns ordered by turn_index. Used by ragprep assistant:chunk."""

    engine = get_sync_engine()

    def _fetch() -> List[PublishedTalk]:
        with engine.begin() as conn:
            talks_rows = conn.execute(
                select(
                    rag_talks_table.c.talk_id,
                    rag_talks_table.c.slug,
                    rag_talks_table.c.title,
                    rag_talks_table.c.summary,
                    rag_talks_table.c.mensch_name,
                )
                .where(
                    rag_talks_table.c.collection == collection,
                    rag_talks_table.c.publishing_status == "published",
                )
                .order_by(rag_talks_table.c.created_at)
            ).mappings().all()

            if not talks_rows:
                return []

            talk_ids = [str(r["talk_id"]) for r in talks_rows]
            turns_rows = conn.execute(
                select(
                    rag_turns_table.c.talk_id,
                    rag_turns_table.c.turn_index,
                    rag_turns_table.c.user_message,
                    rag_turns_table.c.assistant_message,
                )
                .where(rag_turns_table.c.talk_id.in_(talk_ids))
                .order_by(rag_turns_table.c.talk_id, rag_turns_table.c.turn_index)
            ).mappings().all()

            turns_by_talk: Dict[str, List[PublishedTurn]] = {}
            for tr in turns_rows:
                tid = str(tr["talk_id"])
                turns_by_talk.setdefault(tid, []).append(
                    PublishedTurn(
                        turn_index=int(tr["turn_index"]),
                        user_message=str(tr["user_message"]),
                        assistant_message=str(tr["assistant_message"]),
                    )
                )

            return [
                PublishedTalk(
                    talk_id=str(r["talk_id"]),
                    slug=str(r["slug"]),
                    title=str(r["title"]),
                    summary=str(r["summary"]) if r["summary"] else None,
                    mensch_name=str(r["mensch_name"]),
                    turns=turns_by_talk.get(str(r["talk_id"]), []),
                )
                for r in talks_rows
            ]

    import asyncio
    talks = await asyncio.to_thread(_fetch)
    return PublishedTalksResponse(collection=collection, count=len(talks), talks=talks)
