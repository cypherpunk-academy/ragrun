import pytest
import asyncio
import numpy as np
from app.services.embedding_service import LocalEmbeddingService

@pytest.mark.asyncio
async def test_embedding_service_initialization():
    """Test that the embedding service can be initialized."""
    service = LocalEmbeddingService()
    assert service is not None
    assert not service.ready

@pytest.mark.asyncio
async def test_single_text_embedding():
    """Test embedding generation for a single text."""
    service = LocalEmbeddingService()
    await service.load_model()
    
    text = "This is a test sentence."
    embedding = await service.encode_texts(text)
    
    assert embedding is not None
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape[0] == 1024  # e5-large dimension
    assert not np.isnan(embedding).any()

@pytest.mark.asyncio 
async def test_batch_text_embedding():
    """Test embedding generation for multiple texts."""
    service = LocalEmbeddingService()
    await service.load_model()
    
    texts = [
        "First test sentence.",
        "Second test sentence.",
        "Third test sentence."
    ]
    embeddings = await service.encode_texts(texts)
    
    assert embeddings is not None
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (3, 1024)  # 3 texts, 1024 dimensions
    assert not np.isnan(embeddings).any()

@pytest.mark.asyncio
async def test_similarity_search():
    """Test similarity search functionality."""
    service = LocalEmbeddingService()
    await service.load_model()
    
    query = "machine learning algorithms"
    documents = [
        "Deep learning is a subset of machine learning.",
        "Natural language processing uses various algorithms.",
        "Computer vision applies machine learning techniques.",
        "Database systems store and retrieve data efficiently.",
        "Web development involves creating websites and applications."
    ]
    
    results = await service.similarity_search(query, documents, top_k=3)
    
    assert len(results) == 3
    assert all("score" in result for result in results)
    assert all("document" in result for result in results)
    assert all("index" in result for result in results)
    
    # Results should be sorted by score (highest first)
    scores = [result["score"] for result in results]
    assert scores == sorted(scores, reverse=True)

@pytest.mark.asyncio
async def test_health_check():
    """Test health check functionality."""
    service = LocalEmbeddingService()
    await service.load_model()
    
    health = await service.health_check()
    
    assert health["status"] == "healthy"
    assert "model" in health
    assert "device" in health
    assert "embedding_dimension" in health
    assert health["embedding_dimension"] == 1024 