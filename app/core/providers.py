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
from app.infra.sparse_embedder import SparseEmbedder


@lru_cache(maxsize=1)
def get_embedding_client(batch_size: int | None = None) -> EmbeddingClient:
    return EmbeddingClient(
        settings.embeddings_base_url,
        timeout=settings.embeddings_timeout_seconds,
        batch_size=batch_size or 64,
    )


@lru_cache(maxsize=1)
def get_qdrant_client(timeout: float | None = None) -> QdrantClient:
    return QdrantClient(
        settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=timeout if timeout is not None else settings.qdrant_timeout_seconds,
    )


def _deepseek_cache_key(model: Optional[str]) -> str:
    return model or "deepseek-v4-flash"


@lru_cache(maxsize=8)
def get_deepseek_client(
    model: Optional[str] = None,
    thinking_type: Optional[str] = None,
) -> DeepSeekClient:
    if not settings.deepseek_api_key:
        raise RuntimeError("RAGRUN_DEEPSEEK_API_KEY is required for LLM calls")
    return DeepSeekClient(
        settings.deepseek_api_key,
        model=model or settings.deepseek_chat_model or "deepseek-v4-flash",
        base_url=settings.deepseek_base_url,
        timeout=getattr(settings, "deepseek_timeout_seconds", 120.0),
        thinking={"type": thinking_type} if thinking_type else None,
    )


def get_deepseek_reasoner_client() -> DeepSeekClient:
    # Old deepseek-reasoner behavior = thinking mode enabled.
    return get_deepseek_client(model=settings.deepseek_reasoner_model, thinking_type="enabled")


def get_deepseek_chat_client() -> DeepSeekClient:
    # Old deepseek-chat behavior = non-thinking. deepseek-v4-flash defaults to
    # thinking "enabled" if the field is omitted, so this must be explicit.
    return get_deepseek_client(model=settings.deepseek_chat_model, thinking_type="disabled")


@lru_cache(maxsize=1)
def get_sparse_embedder() -> SparseEmbedder:
    return SparseEmbedder()


@lru_cache(maxsize=1)
def get_sync_engine():  # pragma: no cover - thin wrapper
    return get_engine()


def get_async_sessionmaker_cached():  # pragma: no cover - thin wrapper
    return get_async_sessionmaker()
