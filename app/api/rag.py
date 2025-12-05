"""REST router for ingestion and maintenance endpoints."""
from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, constr

from ragrun.models import ChunkRecord

from ..config import settings
from ..db.session import get_engine
from ..services.embedding_client import EmbeddingClient
from ..services.ingestion_service import IngestionService
from ..services.mirror_repository import ChunkMirrorRepository
from ..services.qdrant_client import QdrantClient
from ..services.telemetry import telemetry_client as ingestion_telemetry

router = APIRouter(prefix="/rag", tags=["rag"])


class UploadRequest(BaseModel):
    collection: constr(min_length=1)  # type: ignore[valid-type]
    chunks: List[ChunkRecord]
    embedding_model: Optional[str] = Field(
        None, description="Optional embedding model override"
    )
    batch_size: Optional[int] = Field(
        None,
        ge=1,
        le=512,
        description="Override default embedding batch size",
    )


class UploadResponse(BaseModel):
    ingestion_id: str
    collection: str
    requested: int
    ingested: int
    duplicates: int
    embedding_model: str
    vector_size: int


class DeleteRequest(BaseModel):
    collection: constr(min_length=1)  # type: ignore[valid-type]
    chunk_ids: List[str] = Field(..., min_length=1)
    cascade: bool = Field(
        False,
        description="Reserved for future cascading deletes (parent/child).",
    )


class DeleteResponse(BaseModel):
    collection: str
    requested: int
    deleted: int


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


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_chunks(
    request: UploadRequest,
    service: IngestionService = Depends(get_ingestion_service),
) -> UploadResponse:
    """Validate, embed, and upsert JSONL chunk payloads."""

    try:
        result = await service.upload_chunks(
            collection=request.collection,
            chunks=request.chunks,
            embedding_model=request.embedding_model,
            batch_size=request.batch_size,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    return UploadResponse(**asdict(result))


@router.delete("/delete", response_model=DeleteResponse)
async def delete_chunks(
    request: DeleteRequest,
    service: IngestionService = Depends(get_ingestion_service),
) -> DeleteResponse:
    """Delete chunks from Qdrant (and mirrored stores)."""

    try:
        result = await service.delete_chunks(
            collection=request.collection,
            chunk_ids=request.chunk_ids,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return DeleteResponse(**asdict(result))

