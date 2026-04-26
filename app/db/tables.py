"""SQLAlchemy table definitions."""
from __future__ import annotations

from sqlalchemy import (
    ARRAY,
    JSON,
    BigInteger,
    Boolean,
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
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID

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

# --- rag_usage: Abrechnungsdaten pro LLM-Aufruf ---
rag_usage_table = Table(
    "rag_usage",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("account_id", String(128), nullable=False, server_default="anonymous"),
    Column("thread_id", String(64), nullable=True),
    Column("endpoint", String(256), nullable=True),
    Column("model", String(128), nullable=True),
    Column("provider", String(64), nullable=False, server_default="deepseek"),
    Column("prompt_tokens", Integer, nullable=True),
    Column("completion_tokens", Integer, nullable=True),
    Column("total_tokens", Integer, nullable=True),
    Column("extra", JSONType, nullable=True),
    Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
)

Index("idx_ru_account_id", rag_usage_table.c.account_id)
Index("idx_ru_created_at", rag_usage_table.c.created_at)


# --- rag_talks: Single Source of Truth für Gespräche ---
rag_talks_table = Table(
    "rag_talks",
    metadata,
    Column("talk_id", UUID(as_uuid=False), primary_key=True, server_default=text("gen_random_uuid()")),
    Column("collection", String(128), nullable=False),
    Column("mensch_id", String(128), nullable=False, server_default=""),
    Column("mensch_name", String(256), nullable=False, server_default=""),
    Column("slug", String(256), nullable=False),
    Column("title", Text, nullable=False),
    Column("action_id", String(128), nullable=True),
    Column("summary", Text, nullable=True),
    Column("usage", JSONB().with_variant(JSON(), "sqlite"), nullable=True),
    Column("kontext_meta", JSONB().with_variant(JSON(), "sqlite"), nullable=True),
    Column("publishing_status", String(16), nullable=False, server_default="draft"),
    Column("bug_description", Text, nullable=True),
    Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
    Column("updated_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
)

Index("idx_rt_collection", rag_talks_table.c.collection)
Index("idx_rt_collection_slug", rag_talks_table.c.collection, rag_talks_table.c.slug, unique=True)
Index("idx_rt_created_at", rag_talks_table.c.created_at)


# --- rag_turns: Einzelne Gesprächsrunden ---
rag_turns_table = Table(
    "rag_turns",
    metadata,
    Column("turn_id", UUID(as_uuid=False), primary_key=True, server_default=text("gen_random_uuid()")),
    Column(
        "talk_id",
        UUID(as_uuid=False),
        ForeignKey("rag_talks.talk_id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("turn_index", Integer, nullable=False),
    Column("action_id", String(128), nullable=True),
    Column("assistant_personality", String(128), nullable=True),
    Column("user_message", Text, nullable=False),
    Column("assistant_message", Text, nullable=False),
    Column("usage", JSONB().with_variant(JSON(), "sqlite"), nullable=True),
    Column("references", JSONB().with_variant(JSON(), "sqlite"), nullable=True),
    Column("collection", String(128), nullable=True),
    Column("is_relay", Boolean, nullable=False, server_default="false"),
    Column("chunk_index_map", JSONB().with_variant(JSON(), "sqlite"), nullable=True),
    Column("kontext_meta", JSONB().with_variant(JSON(), "sqlite"), nullable=True),
    Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
    Column("updated_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
)

Index("idx_rtu_talk_id", rag_turns_table.c.talk_id)
Index("idx_rtu_talk_id_index", rag_turns_table.c.talk_id, rag_turns_table.c.turn_index, unique=True)
Index("idx_rtu_created_at", rag_turns_table.c.created_at)

