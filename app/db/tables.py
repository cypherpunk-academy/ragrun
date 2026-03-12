"""SQLAlchemy table definitions."""
from __future__ import annotations

from sqlalchemy import (
    ARRAY,
    JSON,
    BigInteger,
    Column,
    DateTime,
    ForeignKey,
    Index,
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
    Column("worldviews", ARRAY(String)),
    Column("importance", Integer),
    Column("content_hash", String(128), nullable=False),
    Column("text", Text),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Column("metadata", JSONType, nullable=False),
    Column("references", JSONType),
)


# --- event_metadata: dauerhaftes Logging (Metadaten, chunk_ids, Timestamps) ---
event_metadata_table = Table(
    "event_metadata",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("endpoint", String(256), nullable=False),
    Column("graph_event_id", String(64)),
    Column("graph_name", String(128)),
    Column("step", String(128)),
    Column("collection", String(128)),
    Column("worldview", String(256)),
    Column("retrieval_mode", String(64)),
    Column("sufficiency", String(32)),
    Column("error_count", Integer),
    Column("metadata", JSONType),
    Column("chunk_ids", JSONType),
    Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
)

Index("idx_em_endpoint", event_metadata_table.c.endpoint)
Index("idx_em_created_at", event_metadata_table.c.created_at)
Index("idx_em_graph_event_id", event_metadata_table.c.graph_event_id)

# --- event_content: 7-Tage-Rotation (Inhalte, Zwischenergebnisse) ---
event_content_table = Table(
    "event_content",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column(
        "event_metadata_id",
        BigInteger,
        ForeignKey("event_metadata.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("thread_id", String(64), nullable=True),
    Column("concept", String(512)),
    Column("query_text", Text),
    Column("prompt_messages", JSONType),
    Column("context_refs", JSONType),
    Column("context_source", JSONType),
    Column("context_text", Text),
    Column("response_text", Text),
    Column("errors", JSONType),
    Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
)

Index("idx_ec_event_metadata_id", event_content_table.c.event_metadata_id)
Index("idx_ec_thread_id", event_content_table.c.thread_id)
Index("idx_ec_created_at", event_content_table.c.created_at)

