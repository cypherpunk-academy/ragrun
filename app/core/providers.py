"""Centralized dependency providers for infra clients and DB handles.

This keeps construction in one place so ingestion/retrieval modules share
consistent configuration and make DI/testing easier.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from app.config import settings
from app.db.session import get_engine
from app.db.async_session import get_async_sessionmaker
from app.infra.embedding_client import EmbeddingClient
from app.infra.qdrant_client import QdrantClient
from app.infra.deepseek_client import DeepSeekClient


@lru_cache(maxsize=1)
def get_embedding_client(batch_size: int | None = None) -> EmbeddingClient:
    return EmbeddingClient(
        settings.embeddings_base_url,
        batch_size=batch_size or 64,
    )


@lru_cache(maxsize=1)
def get_qdrant_client(timeout: float | None = None) -> QdrantClient:
    return QdrantClient(
        settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=timeout or 30.0,
    )


def _deepseek_cache_key(model: Optional[str]) -> str:
    return model or "deepseek-chat"


@lru_cache(maxsize=8)
def get_deepseek_client(model: Optional[str] = None) -> DeepSeekClient:
    if not settings.deepseek_api_key:
        raise RuntimeError("RAGRUN_DEEPSEEK_API_KEY is required for LLM calls")
    return DeepSeekClient(
        settings.deepseek_api_key,
        model=model or settings.deepseek_chat_model or "deepseek-chat",
        base_url=settings.deepseek_base_url,
    )


def get_deepseek_reasoner_client() -> DeepSeekClient:
    return get_deepseek_client(model=settings.deepseek_reasoner_model)


def get_deepseek_chat_client() -> DeepSeekClient:
    return get_deepseek_client(model=settings.deepseek_chat_model)


@lru_cache(maxsize=1)
def get_sync_engine():  # pragma: no cover - thin wrapper
    return get_engine()


def get_async_sessionmaker_cached():  # pragma: no cover - thin wrapper
    return get_async_sessionmaker()
