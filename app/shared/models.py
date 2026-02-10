"""Shared domain models used by ingestion and retrieval."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, constr

# Allowed chunk types per CHUNK_METADATA_MODEL §4
CHUNK_TYPE_ENUM = (
    "book",
    "secondary_book",
    "chapter_summary",
    "begriff_list",
    "talk",
    "talk_summary",
    "essay",
    "essay_summary",
    "quote",
    "explanation",
    "explanation_summary",
    "typology",
)


class ChunkMetadata(BaseModel):
    """Metadata payload stored alongside each chunk."""

    author: Optional[str] = Field(None, description="Primary author/creator of the source.")
    source_id: str = Field(..., description="Stable identifier for the originating artifact.")
    source_title: Optional[str] = Field(None, description="Human-readable title.")
    source_index: Optional[int] = Field(
        None,
        ge=0,
        description="Zero-based index of the chunk within the same source_id.",
    )
    segment_id: Optional[str] = Field(
        None,
        description="Identifier for the logical segment (chapter, concept entry, etc.).",
    )
    segment_title: Optional[str] = Field(None, description="Title/label for the segment.")
    segment_index: Optional[int] = Field(
        None,
        ge=0,
        description="Zero-based index of the chunk within the segment.",
    )
    parent_id: Optional[str] = Field(
        None,
        description="Reference back to the original artifact when chunk is AI-generated.",
    )
    chunk_id: str = Field(..., description="Unique chunk identifier (must match point id).")
    chunk_type: constr(strip_whitespace=True) = Field(..., description="Chunk category enum.")
    worldviews: Optional[List[str]] = Field(
        None,
        description="List of worldviews where this chunk is a reference.",
    )
    importance: int = Field(
        5,
        ge=1,
        le=10,
        description="Ranking used for scoring adjustments and curation workflows.",
    )
    text: Optional[str] = Field(None, description="Actual chunk text; stored separately when needed.")
    content_hash: str = Field(..., description="SHA-256 of canonical text for dedupe.")
    created_at: datetime = Field(..., description="ISO timestamp when chunk was created.")
    updated_at: datetime = Field(..., description="ISO timestamp when metadata was last modified.")
    source_type: Optional[str] = Field(
        None,
        description="Optional container type (book, begriff_list, essay, etc.).",
    )
    language: str = Field(..., min_length=2, max_length=5, description="ISO language code.")
    tags: List[str] = Field(default_factory=list, description="Free-form tags for downstream filters.")
    references: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="References to chunks that influenced this chunk's generation.",
    )

    @classmethod
    def json_schema(cls) -> dict:
        """Return the JSON schema used by clients (explicit alias)."""

        return cls.model_json_schema()

    @classmethod
    def validate_chunk_type(cls, value: str) -> str:
        """Ensure chunk_type matches the allowed enum list."""

        if value not in CHUNK_TYPE_ENUM:
            raise ValueError(f"chunk_type '{value}' is not supported")
        return value

    @classmethod
    def _normalize_worldviews(cls, payload: dict) -> None:
        """Ensure worldviews is a list of strings and reject legacy worldview."""

        if "worldview" in payload and payload.get("worldview") is not None:
            raise ValueError("legacy 'worldview' is no longer supported; use 'worldviews' array")

        worldviews = payload.get("worldviews")
        if worldviews is None:
            return
        if not isinstance(worldviews, list):
            raise ValueError("'worldviews' must be a list of strings")

        normalized: list[str] = []
        for entry in worldviews:
            if not isinstance(entry, str) or not entry.strip():
                raise ValueError("'worldviews' entries must be non-empty strings")
            normalized.append(entry.strip())

        # Deduplicate while preserving order
        payload["worldviews"] = list(dict.fromkeys(normalized))

    @classmethod
    def from_payload(cls, payload: dict) -> "ChunkMetadata":
        """Parse metadata, applying enum validation for chunk_type."""

        chunk_type = payload.get("chunk_type")
        if chunk_type is None:
            raise ValueError("chunk_type is required")
        payload["chunk_type"] = cls.validate_chunk_type(chunk_type)
        cls._normalize_worldviews(payload)
        return cls(**payload)


class ChunkRecord(BaseModel):
    """Top-level chunk container including text and metadata."""

    id: str = Field(..., description="Primary identifier, mirrors chunk_id.")
    text: str = Field(..., description="Chunk content (<=512 tokens recommended).")
    metadata: ChunkMetadata
    embedding: Optional[List[float]] = Field(
        None,
        description="Vector embedding (filled post-embedding step).",
    )

    @classmethod
    def from_dict(cls, payload: dict) -> "ChunkRecord":
        """Construct a chunk record from raw dict input."""

        metadata = payload.get("metadata")
        if metadata is None:
            raise ValueError("metadata is required")
        metadata_model = ChunkMetadata.from_payload(metadata)
        return cls(
            id=payload.get("id") or metadata_model.chunk_id,
            text=payload["text"],
            metadata=metadata_model,
            embedding=payload.get("embedding"),
        )


__all__ = [
    "ChunkRecord",
    "ChunkMetadata",
    "CHUNK_TYPE_ENUM",
]
