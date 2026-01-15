"""Compatibility layer for legacy imports under app.services.

New code should import from `app.infra`, `app.core`, or `app.ingestion`.
"""
from app.infra.embedding_client import EmbeddingClient  # noqa: F401
from app.infra.qdrant_client import QdrantClient  # noqa: F401
from app.infra.deepseek_client import DeepSeekClient  # noqa: F401
from app.ingestion.repositories import ChunkMirrorRepository  # noqa: F401
from app.ingestion.services import IngestionService, UploadResult, DeleteResult  # noqa: F401
from app.core.telemetry import IngestionTelemetryClient, telemetry_client  # noqa: F401
