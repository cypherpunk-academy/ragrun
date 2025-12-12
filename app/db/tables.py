"""SQLAlchemy table definitions."""
from __future__ import annotations

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    func,
)
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


retrieval_events_table = Table(
    "retrieval_events",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("concept", String(512), nullable=False),
    Column("branch", String(128), nullable=False),
    Column("collection", String(128), nullable=False),
    Column("answer", Text),
    Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
)


retrieval_chunks_table = Table(
    "retrieval_chunks",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column(
        "event_id",
        Integer,
        ForeignKey("retrieval_events.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("kind", String(32), nullable=False),
    Column("text", Text, nullable=False),
    Column("score", Float),
    Column("source_id", String(256)),
    Column("chunk_type", String(128)),
    Column("metadata", JSONType),
    Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
)

