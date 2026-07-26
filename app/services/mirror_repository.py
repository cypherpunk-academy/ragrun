"""Compatibility re-export; canonical implementation lives in ingestion.repositories."""
from __future__ import annotations

from app.ingestion.repositories.mirror_repository import (
    ChunkMirrorRepository,
    VectorChunksRepository,
)

__all__ = ["ChunkMirrorRepository", "VectorChunksRepository"]
