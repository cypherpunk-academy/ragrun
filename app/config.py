"""Runtime configuration for the ragrun FastAPI service."""
import os
from functools import lru_cache
from typing import Optional

from pydantic import AliasChoices, AnyHttpUrl, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def normalize_postgres_dsn(dsn: str) -> str:
    """Force psycopg v3 driver — bare postgresql:// would default to psycopg2."""
    s = (dsn or "").strip()
    if not s:
        return s
    if s.startswith("postgresql+psycopg://"):
        return s
    if s.startswith("postgresql+psycopg2://"):
        return "postgresql+psycopg://" + s.removeprefix("postgresql+psycopg2://")
    if s.startswith("postgresql://"):
        return "postgresql+psycopg://" + s.removeprefix("postgresql://")
    if s.startswith("postgres://"):
        return "postgresql+psycopg://" + s.removeprefix("postgres://")
    return s


class Settings(BaseSettings):
    """Application settings pulled from environment variables."""

    app_env: str = "development"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    qdrant_url: AnyHttpUrl = "http://qdrant:6333"
    qdrant_api_key: Optional[str] = None
    # httpx connect/read/write/pool; large upsert bodies need a high write budget (default was 30s → WriteTimeout).
    qdrant_timeout_seconds: float = 300.0

    postgres_dsn: str = "postgresql+psycopg://ragrun:ragrun@postgres:5432/ragrun"

    @field_validator("postgres_dsn", mode="before")
    @classmethod
    def _normalize_postgres_dsn(cls, v: object) -> object:
        if isinstance(v, str):
            return normalize_postgres_dsn(v)
        return v

    embeddings_base_url: AnyHttpUrl = "http://embedding-service:8001"
    embeddings_timeout_seconds: float = 180.0
    # "http" = personal-embeddings-service; "huggingface" = HF Inference feature-extraction.
    embeddings_provider: str = "http"
    hf_token: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices(
            "RAGRUN_HF_TOKEN",
            "HF_TOKEN",
            "HUGGINGFACE_HUB_TOKEN",
        ),
    )
    embeddings_hf_model: str = "intfloat/multilingual-e5-large"
    # Override full feature-extraction URL if needed; default is router.huggingface.co.
    embeddings_hf_api_url: Optional[str] = None
    embeddings_hf_max_texts_per_request: int = 8
    embeddings_hf_max_retries: int = 4
    # Soft client guard: refuse embed_texts batches larger than this on HF.
    embeddings_hf_max_batch_texts: int = 32
    # Hard ingest guard: block POST /rag/embed-chunks when provider is huggingface.
    embeddings_hf_forbid_ingest: bool = True
    deepseek_base_url: AnyHttpUrl = "https://api.deepseek.com"

    langfuse_host: Optional[str] = None
    langfuse_public_key: Optional[str] = None
    langfuse_secret_key: Optional[str] = None
    langfuse_encryption_key: Optional[str] = None
    langfuse_ingestion_dataset: Optional[str] = "ingestion_runs"
    langfuse_retrieval_dataset: Optional[str] = "retrieval_runs"
    telemetry_timeout_seconds: float = 2.0

    deepseek_api_key: Optional[str] = None
    # "deepseek-chat"/"deepseek-reasoner" are retired 2026-07-24 15:59 UTC.
    # Both aliases already route to deepseek-v4-flash today, distinguished only
    # by the "thinking" request parameter (see app/core/providers.py) — not by
    # a separate model name. deepseek-v4-pro exists as a higher-tier model but
    # is not required to preserve current behavior.
    deepseek_reasoner_model: Optional[str] = "deepseek-v4-flash"
    deepseek_chat_model: Optional[str] = "deepseek-v4-flash"
    deepseek_model_probe: bool = True
    deepseek_timeout_seconds: float = 120.0

    # Prompt loading
    # Note: in this repo, assistant prompt files live under `ragrun/ragkeep/assistants`.
    # Keep this configurable to support alternative layouts / Docker packaging.
    assistants_root: str = "ragkeep/assistants"

    # External chat-personalities directory (outside the repo, configured via .env).
    # When empty, only the built-in assistant-host personality is available.
    personalities_root: str = ""

    # "Standard chunk" sizing (used by authentic_concept_explain prompts).
    # Aligned with quote_explanation budget (~200–300 words, e5-safe with passage prefix).
    ace_chunk_min_words: int = 180
    ace_chunk_target_words: int = 240
    ace_chunk_max_words: int = 300

    # CORS: comma-separated list of allowed origins (e.g. "http://localhost:8080,https://example.com")
    cors_origins: str = ""

    # Shared secret for internal CLI routes (/api/v1/rag/* and /api/v1/admin/*).
    # When set, every request to those routes must supply the same value in the
    # X-Api-Key header.  When empty (default), the routes remain open — safe for
    # local / trusted-LAN deployments.  Generate with: openssl rand -hex 32
    internal_api_key: str = ""

    # Supabase (JWT validation for /app/* and sync RPC forwarding)
    supabase_url: str = Field(
        default="",
        validation_alias=AliasChoices("RAGRUN_SUPABASE_URL", "EXPO_PUBLIC_SUPABASE_URL"),
    )
    supabase_anon_key: str = Field(
        default="",
        validation_alias=AliasChoices("RAGRUN_SUPABASE_ANON_KEY", "EXPO_PUBLIC_SUPABASE_ANON_KEY"),
    )
    supabase_jwt_secret: str = Field(
        default="",
        validation_alias=AliasChoices("RAGRUN_SUPABASE_JWT_SECRET", "SUPABASE_JWT_SECRET"),
    )

    # Default assistant for app search when collection is omitted
    app_default_assistant_slug: str = "philo-von-freisinn"

    # Embedding prefixes for instruction-tuned models (e.g. multilingual-e5-large).
    # Leave empty for models that do not require prefixes (e.g. cross-en-de-roberta).
    # Set via RAGRUN_EMBEDDING_PREFIX_PASSAGE and RAGRUN_EMBEDDING_PREFIX_QUERY in .env.
    embedding_prefix_passage: str = ""
    embedding_prefix_query: str = ""

    use_hybrid_retrieval: bool = False
    hybrid_prefer_short_concepts: bool = True
    hybrid_short_concept_max_words: int = 2
    hybrid_short_concept_max_chars: int = 32
    hybrid_fallback_on_thin: bool = True

    # Chat citation filtering: minimum relevance (0.0–1.0) for a chunk to be cited
    citation_relevance_threshold: float = 0.3

    # Retrieval: filter out very short chunks (e.g. "13| Dr. Rudolf Steiner.")
    min_chunk_chars: int = 80
    retrieval_overfetch_multiplier: int = 2

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
        populate_by_name=True,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings accessor used across the codebase."""

    return Settings()  # type: ignore[arg-type]


settings = get_settings()
