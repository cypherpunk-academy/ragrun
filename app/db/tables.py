"""SQLAlchemy table definitions."""
from __future__ import annotations

from sqlalchemy import JSON, Column, DateTime, Integer, MetaData, String, Table, Text
from sqlalchemy.dialects.postgresql import JSONB

metadata = MetaData()

# Prefer JSONB when available (Postgres) but gracefully fall back to generic JSON.
JSONType = JSONB().with_variant(JSON(), "sqlite")

chunks_table = Table(
    "rag_chunks",
    metadata,
    Column("collection", String(128), primary_key=True),
    Column("chunk_id", String(256), primary_key=True),
    Column("source_id", String(256), nullable=False),
    Column("chunk_type", String(64), nullable=False),
    Column("language", String(8), nullable=False),
    Column("worldview", String(128)),
    Column("importance", Integer),
    Column("content_hash", String(128), nullable=False),
    Column("text", Text),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Column("metadata", JSONType, nullable=False),
)

