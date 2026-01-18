"""Runtime configuration for the ragrun FastAPI service."""
import os
from functools import lru_cache
from typing import Optional

from pydantic import AnyHttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings pulled from environment variables."""

    app_env: str = "development"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    qdrant_url: AnyHttpUrl = "http://qdrant:6333"
    qdrant_api_key: Optional[str] = None

    postgres_dsn: str = "postgresql+psycopg://ragrun:ragrun@postgres:5432/ragrun"
    embeddings_base_url: AnyHttpUrl = "http://embedding-service:8001"
    deepseek_base_url: AnyHttpUrl = "https://api.deepseek.com"

    langfuse_host: Optional[str] = None
    langfuse_public_key: Optional[str] = None
    langfuse_secret_key: Optional[str] = None
    langfuse_encryption_key: Optional[str] = None
    langfuse_ingestion_dataset: Optional[str] = "ingestion_runs"
    langfuse_retrieval_dataset: Optional[str] = "retrieval_runs"
    telemetry_timeout_seconds: float = 2.0

    deepseek_api_key: Optional[str] = None
    deepseek_reasoner_model: Optional[str] = "deepseek-reasoner"
    deepseek_chat_model: Optional[str] = "deepseek-chat"
    deepseek_model_probe: bool = True

    use_hybrid_retrieval: bool = False
    hybrid_prefer_short_concepts: bool = True
    hybrid_short_concept_max_words: int = 2
    hybrid_short_concept_max_chars: int = 32
    hybrid_fallback_on_thin: bool = True

    # concept_explain_worldviews graph retrieval sizing (final reranked chunk counts)
    # Note: base and widen sizes are derived from these finals in the graph to ensure
    # the system can actually return up to k_final (i.e. widen/base are >= k_final).
    cewv_k_final_concept: int = 6
    cewv_k_final_context1: int = 4
    cewv_k_final_context2: int = 6

    # In some environments (e.g. restricted sandboxes) a present `.env` file may be
    # unreadable; fall back to environment variables only in that case.
    _env_file: str | None = ".env"
    try:
        if _env_file and os.path.exists(_env_file):
            with open(_env_file, "rb"):
                pass
    except OSError:
        _env_file = None

    model_config = SettingsConfigDict(
        env_file=_env_file,
        env_prefix="RAGRUN_",
        env_ignore_empty=True,
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings accessor used across the codebase."""

    return Settings()  # type: ignore[arg-type]


settings = get_settings()
