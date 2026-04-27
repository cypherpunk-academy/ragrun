"""Compatibility re-export; canonical implementation lives in app.ingestion.services."""
from __future__ import annotations

from app.ingestion.services.ingestion_service import (
    DeleteResult,
    IngestionService,
    UploadResult,
)

__all__ = ["DeleteResult", "IngestionService", "UploadResult"]
