"""FastAPI endpoints for RAG ingestion and deletion (ragprep-compatible)."""
from __future__ import annotations

import json
from dataclasses import asdict
from functools import lru_cache
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import func, select, text

from ..shared.models import ChunkRecord

from ..config import settings
from ..db.session import get_engine
from ..db.tables import chunks_table
from ..core.providers import get_embedding_client, get_qdrant_client, get_sync_engine
from ..core.telemetry import telemetry_client as ingestion_telemetry
from ..ingestion.repositories import ChunkMirrorRepository
from ..ingestion.services import IngestionService

router = APIRouter(prefix="/rag", tags=["rag"])


class UploadChunksRequest(BaseModel):
    """Request to upload chunks from JSONL content."""

    chunks_jsonl_content: str = Field(
        ..., description="JSONL-formatted chunks (one JSON object per line)"
    )
    collection_name: str = Field(..., description="Target collection name")
    batch_size: Optional[int] = Field(
        None, ge=1, le=512, description="Embedding batch size"
    )
    embedding_model: Optional[str] = Field(
        None, description="Optional embedding model override"
    )
    skip_cleanup: bool = Field(
        False,
        description=(
            "If true, do not delete stale chunks during this upload. "
            "Used by sync workflows that delete stale chunk_ids explicitly."
        ),
    )


class UploadChunksResponse(BaseModel):
    """Response from upload-chunks endpoint."""

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
    mirror_repository = ChunkMirrorRepository(get_sync_engine())
    return IngestionService(
        embedding_client=embedding_client,
        qdrant_client=qdrant_client,
        mirror_repository=mirror_repository,
        telemetry_client=ingestion_telemetry,
        default_batch_size=64,
    )


@router.post(
    "/upload-chunks",
    response_model=UploadChunksResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_chunks(
    request: UploadChunksRequest,
    service: IngestionService = Depends(get_ingestion_service),
) -> UploadChunksResponse:
    """Upload chunks from JSONL content (ragprep-compatible endpoint)."""

    # Parse JSONL content into ChunkRecord objects
    lines = request.chunks_jsonl_content.strip().split("\n")
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

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid chunks found in JSONL content",
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
        query = select(func.count()).select_from(chunks_table).where(
            chunks_table.c.collection == collection
        )

        if not request.all and request.filter:
            # Build filter conditions
            for key, value in request.filter.items():
                # Use JSONB containment for metadata fields
                query = query.where(
                    chunks_table.c.metadata[key].as_string() == str(value)
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
    query = select(chunks_table.c.chunk_id).where(
        chunks_table.c.collection == collection
    )

    if not request.all and request.filter:
        for key, value in request.filter.items():
            query = query.where(chunks_table.c.metadata[key].as_string() == str(value))
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

    # Query for distinct book_title with counts
    if include_author:
        # Prefer explicit book_title; fall back to source_title for older ingestions
        query = text(
            """
            SELECT 
                metadata->>'chunk_type' as chunk_type,
                metadata->>'author' as author,
                COALESCE(metadata->>'book_title', metadata->>'source_title') as book_title,
                COUNT(*) as count
            FROM rag_chunks
            WHERE collection = :collection
              AND COALESCE(metadata->>'book_title', metadata->>'source_title') IS NOT NULL
              {chunk_filter}
            GROUP BY metadata->>'chunk_type', metadata->>'author', COALESCE(metadata->>'book_title', metadata->>'source_title')
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
                COUNT(*) as count
            FROM rag_chunks
            WHERE collection = :collection
              AND COALESCE(metadata->>'book_title', metadata->>'source_title') IS NOT NULL
              {chunk_filter}
            GROUP BY metadata->>'chunk_type', COALESCE(metadata->>'book_title', metadata->>'source_title')
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
                "count": row[3],
            }
            for row in rows
        ]
    else:
        titles = [
            {"chunk_type": row[0], "book_title": row[1], "count": row[2]} for row in rows
        ]

    return {
        "collection": collection,
        "total_distinct_titles": len(titles),
        "titles": titles,
    }


@router.get("/collections")
async def list_collections() -> Dict[str, Any]:
    """List all collections from Qdrant."""

    qdrant_client = get_qdrant_client()

    collections = await qdrant_client.list_collections()

    return {"collections": collections, "vector_db_path": str(settings.qdrant_url)}
