"""Factories for retrieval dependencies (clients, config)."""
from __future__ import annotations

from functools import lru_cache

from app.config import settings
from app.services.deepseek_client import DeepSeekClient
from app.services.embedding_client import EmbeddingClient
from app.services.qdrant_client import QdrantClient


@lru_cache(maxsize=1)
def get_embedding_client() -> EmbeddingClient:
    return EmbeddingClient(settings.embeddings_base_url, batch_size=32)


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    return QdrantClient(settings.qdrant_url, api_key=settings.qdrant_api_key)


@lru_cache(maxsize=1)
def get_deepseek_client() -> DeepSeekClient:
    if not settings.deepseek_api_key:
        raise RuntimeError("RAGRUN_DEEPSEEK_API_KEY is required for retrieval agents")
    return DeepSeekClient(settings.deepseek_api_key)


