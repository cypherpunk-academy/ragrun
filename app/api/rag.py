"""FastAPI endpoints for RAG ingestion and deletion (ragprep-compatible)."""
from __future__ import annotations

import json
from dataclasses import asdict
from functools import lru_cache
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import func, select, text

from ragrun.models import ChunkRecord

from ..config import settings
from ..db.session import get_engine
from ..db.tables import chunks_table
from ..services.embedding_client import EmbeddingClient
from ..services.ingestion_service import IngestionService
from ..services.mirror_repository import ChunkMirrorRepository
from ..services.qdrant_client import QdrantClient
from ..services.telemetry import telemetry_client as ingestion_telemetry

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


class UploadChunksResponse(BaseModel):
    """Response from upload-chunks endpoint."""

    ingestion_id: str
    collection: str
    requested: int
    ingested: int
    duplicates: int
    embedding_model: str
    vector_size: int


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


def get_ingestion_service() -> IngestionService:
    """Lazy singleton used as a FastAPI dependency."""
    return _get_ingestion_service()


@lru_cache(maxsize=1)
def _get_ingestion_service() -> IngestionService:
    embedding_client = EmbeddingClient(
        settings.embeddings_base_url,
        batch_size=64,
    )
    qdrant_client = QdrantClient(
        settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )
    mirror_repository = ChunkMirrorRepository(get_engine())
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
) -> Dict[str, Any]:
    """List distinct book titles in a collection with chunk counts."""

    collection = collection_name or "default"
    engine = get_engine()

    # Query for distinct book_title with counts
    if include_author:
        query = text(
            """
            SELECT 
                metadata->>'author' as author,
                metadata->>'book_title' as book_title,
                COUNT(*) as count
            FROM rag_chunks
            WHERE collection = :collection
              AND metadata->>'book_title' IS NOT NULL
            GROUP BY metadata->>'author', metadata->>'book_title'
            HAVING COUNT(*) >= :min_count
            ORDER BY count DESC, book_title
            LIMIT :limit
        """
        )
    else:
        query = text(
            """
            SELECT 
                metadata->>'book_title' as book_title,
                COUNT(*) as count
            FROM rag_chunks
            WHERE collection = :collection
              AND metadata->>'book_title' IS NOT NULL
            GROUP BY metadata->>'book_title'
            HAVING COUNT(*) >= :min_count
            ORDER BY count DESC, book_title
            LIMIT :limit
        """
        )

    with engine.connect() as conn:
        rows = conn.execute(
            query, {"collection": collection, "min_count": min_count, "limit": limit}
        ).fetchall()

    if include_author:
        titles = [
            {"author": row[0], "book_title": row[1], "count": row[2]} for row in rows
        ]
    else:
        titles = [{"book_title": row[0], "count": row[1]} for row in rows]

    return {
        "collection": collection,
        "total_distinct_titles": len(titles),
        "titles": titles,
    }


@router.get("/collections")
async def list_collections() -> Dict[str, Any]:
    """List all collections from Qdrant."""

    qdrant_client = QdrantClient(
        settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )

    collections = await qdrant_client.list_collections()

    return {"collections": collections, "vector_db_path": str(settings.qdrant_url)}
