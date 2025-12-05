# pyright: reportMissingImports=false
from typing import List, Optional

from pydantic import AnyHttpUrl, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Environment configuration for the embedding service."""

    model_name: str = "T-Systems-onsite/cross-en-de-roberta-sentence-transformer"
    allowed_models: List[str] = []
    max_seq_length: int = 512
    batch_size: int = 32
    embedding_dimension: int = 768

    use_half_precision: bool = True
    max_workers: int = 4
    cache_dir: str = "/app/models"

    host: str = "0.0.0.0"
    port: int = 8001

    langfuse_host: Optional[AnyHttpUrl] = None
    langfuse_public_key: Optional[str] = None
    langfuse_secret_key: Optional[str] = None
    langfuse_dataset: str = "embedding_runs"
    telemetry_timeout_seconds: float = 3.0

    class Config:
        env_file = ".env"
        env_prefix = "EMBEDDINGS_"

    @field_validator("allowed_models", mode="before")
    @classmethod
    def _parse_allowed_models(cls, value):
        if value is None or value == "":
            return []
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value


settings = Settings()