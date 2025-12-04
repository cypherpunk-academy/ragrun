from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Union, Dict, Any
import time
import logging
from app.services.embedding_service import embedding_service
from app.services.batch_service import batch_service

logger = logging.getLogger(__name__)
router = APIRouter()

class EmbeddingRequest(BaseModel):
    texts: Union[str, List[str]] = Field(..., description="Text or list of texts to embed")

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

class SimilaritySearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    processing_time: float
    query: str

class BatchRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to process")
    chunk_size: int = Field(32, description="Chunk size for batch processing")

@router.post("/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """Generate embeddings for provided texts."""
    start_time = time.time()

    try:
        if not embedding_service.ready:
            raise HTTPException(
                status_code=503, 
                detail="Embedding service not ready. Please wait for model to load."
            )

        embeddings = await embedding_service.encode_texts(request.texts)
        processing_time = time.time() - start_time

        # The model always returns 2D array: (num_texts, embedding_dim)
        # Convert to list format for JSON response
        embeddings_list = embeddings.tolist()
        count = len(embeddings_list)
        dimensions = len(embeddings_list[0]) if embeddings_list else 0

        return EmbeddingResponse(
            embeddings=embeddings_list,
            dimensions=dimensions,
            model="multilingual-e5-large",
            processing_time=processing_time,
            count=count
        )
    except Exception as e:
        logger.error(f"Embedding generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")

@router.post("/embeddings/batch", response_model=EmbeddingResponse)
async def create_batch_embeddings(request: BatchRequest):
    """Generate embeddings for a large batch of texts."""
    start_time = time.time()

    try:
        if not embedding_service.ready:
            raise HTTPException(
                status_code=503, 
                detail="Embedding service not ready. Please wait for model to load."
            )

        if len(request.texts) == 0:
            raise HTTPException(status_code=400, detail="No texts provided")

        embeddings_list = await batch_service.process_batch(
            request.texts, 
            request.chunk_size
        )
        processing_time = time.time() - start_time

        return EmbeddingResponse(
            embeddings=embeddings_list,
            dimensions=len(embeddings_list[0]) if embeddings_list else 0,
            model="multilingual-e5-large",
            processing_time=processing_time,
            count=len(embeddings_list)
        )
    except Exception as e:
        logger.error(f"Batch embedding generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch embedding generation failed: {str(e)}")

@router.post("/search", response_model=SimilaritySearchResponse)
async def similarity_search(request: SimilaritySearchRequest):
    """Perform similarity search using embeddings."""
    start_time = time.time()

    try:
        if not embedding_service.ready:
            raise HTTPException(
                status_code=503, 
                detail="Embedding service not ready. Please wait for model to load."
            )

        if len(request.documents) == 0:
            raise HTTPException(status_code=400, detail="No documents provided")

        results = await embedding_service.similarity_search(
            request.query,
            request.documents,
            request.top_k
        )
        processing_time = time.time() - start_time

        return SimilaritySearchResponse(
            results=results,
            processing_time=processing_time,
            query=request.query
        )
    except Exception as e:
        logger.error(f"Similarity search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Similarity search failed: {str(e)}")

@router.get("/info")
async def get_service_info():
    """Get information about the embedding service."""
    try:
        return embedding_service.get_service_info()
    except Exception as e:
        logger.error(f"Failed to get service info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get service info: {str(e)}") 