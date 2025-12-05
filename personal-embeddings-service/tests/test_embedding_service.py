import numpy as np
import pytest

from app.config import settings
from app.services.embedding_service import LocalEmbeddingService


@pytest.fixture(autouse=True)
def fake_embedding_backend(monkeypatch):
    """Prevent tests from downloading large models by mocking the backend."""

    def fake_load(self, model_name=None):
        return True

    def fake_encode(self, texts, model_name=None):
        if isinstance(texts, str):
            return np.ones(settings.embedding_dimension, dtype=np.float32)
        return np.ones((len(texts), settings.embedding_dimension), dtype=np.float32)

    monkeypatch.setattr(
        "app.models.embedding_model.EmbeddingModel.load_model", fake_load, raising=False
    )
    monkeypatch.setattr(
        "app.models.embedding_model.EmbeddingModel.encode", fake_encode, raising=False
    )


@pytest.mark.asyncio
async def test_embedding_service_initialization():
    service = LocalEmbeddingService()
    assert service is not None
    assert not service.ready


@pytest.mark.asyncio
async def test_single_text_embedding():
    service = LocalEmbeddingService()
    await service.load_model()

    embedding = await service.encode_texts("This is a test sentence.")

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape[0] == settings.embedding_dimension
    assert not np.isnan(embedding).any()


@pytest.mark.asyncio
async def test_batch_text_embedding():
    service = LocalEmbeddingService()
    await service.load_model()

    texts = ["First", "Second", "Third"]
    embeddings = await service.encode_texts(texts)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (3, settings.embedding_dimension)
    assert not np.isnan(embeddings).any()


@pytest.mark.asyncio
async def test_similarity_search():
    service = LocalEmbeddingService()
    await service.load_model()

    query = "machine learning"
    documents = ["doc1", "doc2", "doc3", "doc4"]

    results = await service.similarity_search(query, documents, top_k=2)

    assert len(results) == 2
    assert all("score" in result for result in results)
    assert all("document" in result for result in results)


@pytest.mark.asyncio
async def test_health_check():
    service = LocalEmbeddingService()
    await service.load_model()

    health = await service.health_check()

    assert health["status"] == "healthy"
    assert health["embedding_dimension"] == settings.embedding_dimension