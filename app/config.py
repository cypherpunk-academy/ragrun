"""Runtime configuration for the ragrun FastAPI service."""
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

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="RAGRUN_",
        env_ignore_empty=True,
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings accessor used across the codebase."""

    return Settings()  # type: ignore[arg-type]


settings = get_settings()
