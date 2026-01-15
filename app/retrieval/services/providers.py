"""Factories for retrieval dependencies (clients, config)."""
from __future__ import annotations

from app.core.providers import (
    get_deepseek_chat_client,
    get_deepseek_reasoner_client,
    get_embedding_client as _get_embedding_client,
    get_qdrant_client as _get_qdrant_client,
)
from app.infra.deepseek_client import DeepSeekClient
from app.infra.embedding_client import EmbeddingClient
from app.infra.qdrant_client import QdrantClient


def get_embedding_client() -> EmbeddingClient:
    # Retrieval favors slightly smaller batch to reduce latency.
    return _get_embedding_client(batch_size=32)


def get_qdrant_client() -> QdrantClient:
    return _get_qdrant_client()


def get_deepseek_client() -> DeepSeekClient:
    return get_deepseek_chat_client()


def get_deepseek_reasoner() -> DeepSeekClient:
    return get_deepseek_reasoner_client()


def get_deepseek_chat() -> DeepSeekClient:
    return get_deepseek_chat_client()


