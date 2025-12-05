import logging
import time
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.batch_service import batch_service
from app.services.embedding_service import embedding_service
from app.services.telemetry import telemetry_client

logger = logging.getLogger(__name__)
router = APIRouter()


class EmbeddingRequest(BaseModel):
    texts: Union[str, List[str]] = Field(
        ..., description="Text or list of texts to embed"
    )
    model: Optional[str] = Field(None, description="Optional model override")


class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    dimensions: int
    model: str
    processing_time: float
    count: int


class SimilaritySearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    documents: List[str] = Field(..., description="Documents to search")
    top_k: int = Field(5, description="Number of top results to return")
    model: Optional[str] = Field(None, description="Optional model override")


class SimilaritySearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    processing_time: float
    query: str


class BatchRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to process")
    chunk_size: int = Field(32, description="Chunk size for batch processing")
    model: Optional[str] = Field(None, description="Optional model override")


@router.post("/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """Generate embeddings for provided texts."""

    start_time = time.time()

    try:
        if not embedding_service.ready:
            raise HTTPException(
                status_code=503,
                detail="Embedding service not ready. Please wait for model to load.",
            )

        try:
            model_name = embedding_service.resolve_model_name(request.model)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        embeddings = await embedding_service.encode_texts(
            request.texts, model_name=model_name
        )
        processing_time = time.time() - start_time

        embeddings_list, count, dimensions = _normalize_embeddings(embeddings)

        await telemetry_client.record_embedding_batch(
            route="POST /api/v1/embeddings",
            count=count,
            duration_seconds=processing_time,
            model_name=model_name,
        )

        return EmbeddingResponse(
            embeddings=embeddings_list,
            dimensions=dimensions,
            model=model_name,
            processing_time=processing_time,
            count=count,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Embedding generation failed: %s", exc)
        raise HTTPException(
            status_code=500, detail=f"Embedding generation failed: {exc}"
        )


@router.post("/embeddings/batch", response_model=EmbeddingResponse)
async def create_batch_embeddings(request: BatchRequest):
    """Generate embeddings for a large batch of texts."""

    start_time = time.time()

    try:
        if not embedding_service.ready:
            raise HTTPException(
                status_code=503,
                detail="Embedding service not ready. Please wait for model to load.",
            )

        if len(request.texts) == 0:
            raise HTTPException(status_code=400, detail="No texts provided")

        try:
            model_name = embedding_service.resolve_model_name(request.model)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        embeddings_list = await batch_service.process_batch(
            request.texts,
            request.chunk_size,
            model_name=model_name,
        )
        processing_time = time.time() - start_time

        await telemetry_client.record_embedding_batch(
            route="POST /api/v1/embeddings/batch",
            count=len(embeddings_list),
            duration_seconds=processing_time,
            model_name=model_name,
        )

        return EmbeddingResponse(
            embeddings=embeddings_list,
            dimensions=len(embeddings_list[0]) if embeddings_list else 0,
            model=model_name,
            processing_time=processing_time,
            count=len(embeddings_list),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Batch embedding generation failed: %s", exc)
        raise HTTPException(
            status_code=500, detail=f"Batch embedding generation failed: {exc}"
        )


@router.post("/search", response_model=SimilaritySearchResponse)
async def similarity_search(request: SimilaritySearchRequest):
    """Perform similarity search using embeddings."""

    start_time = time.time()

    try:
        if not embedding_service.ready:
            raise HTTPException(
                status_code=503,
                detail="Embedding service not ready. Please wait for model to load.",
            )

        if len(request.documents) == 0:
            raise HTTPException(status_code=400, detail="No documents provided")

        try:
            model_name = embedding_service.resolve_model_name(request.model)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        results = await embedding_service.similarity_search(
            request.query,
            request.documents,
            request.top_k,
            model_name=model_name,
        )
        processing_time = time.time() - start_time

        return SimilaritySearchResponse(
            results=results,
            processing_time=processing_time,
            query=request.query,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Similarity search failed: %s", exc)
        raise HTTPException(
            status_code=500, detail=f"Similarity search failed: {exc}"
        )


@router.get("/info")
async def get_service_info():
    """Get information about the embedding service."""

    try:
        return embedding_service.get_service_info()
    except Exception as exc:
        logger.error("Failed to get service info: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to get service info: {exc}")


def _normalize_embeddings(embeddings) -> tuple[List[List[float]], int, int]:
    if embeddings.ndim == 1:
        embeddings_list = [embeddings.tolist()]
        count = 1
        dimensions = embeddings.shape[0]
    else:
        embeddings_list = embeddings.tolist()
        count = len(embeddings_list)
        dimensions = len(embeddings_list[0]) if embeddings_list else 0
    return embeddings_list, count, dimensions
