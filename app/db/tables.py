"""SQLAlchemy table definitions."""
from __future__ import annotations

from sqlalchemy import (
    ARRAY,
    JSON,
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
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

# Mirror of Qdrant chunk payloads (formerly rag_chunks).
vector_chunks_table = Table(
    "vector_chunks",
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

# Primary chunk store (DB-first); embedded_at set after successful Qdrant/vector_chunks sync.
rag_chunks_table = Table(
    "rag_chunks",
    metadata,
    Column("rag_partition", String(128), primary_key=True),
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
    Column("scope", String(64)),
    Column("embedded_at", DateTime(timezone=True)),
    Column("deprecated_at", DateTime(timezone=True)),
)

Index("idx_rag_chunks_source_id", rag_chunks_table.c.source_id)
Index(
    "idx_rag_chunks_rag_partition_embedded_at",
    rag_chunks_table.c.rag_partition,
    rag_chunks_table.c.embedded_at,
)

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
    Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
    Column("turn_id", UUID(as_uuid=False), ForeignKey("rag_turns.turn_id", ondelete="SET NULL"), nullable=True),
    Column("talk_id", UUID(as_uuid=False), ForeignKey("rag_talks.talk_id", ondelete="SET NULL"), nullable=True),
)

Index("idx_ru_account_id", rag_usage_table.c.account_id)
Index("idx_ru_created_at", rag_usage_table.c.created_at)
Index("idx_ru_turn_id", rag_usage_table.c.turn_id)
Index("idx_ru_talk_id", rag_usage_table.c.talk_id)


# --- llm_pricing: Modellpreise für on-demand Kostenkalkulation ---
llm_pricing_table = Table(
    "llm_pricing",
    metadata,
    Column("model", String(128), primary_key=True),
    Column("provider", String(64), nullable=False, server_default="deepseek"),
    Column("prompt_per_1m_usd", Float, nullable=False),
    Column("completion_per_1m_usd", Float, nullable=False),
    Column("updated_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
    Column("note", Text, nullable=True),
)


# --- rag_talks: Single Source of Truth für Gespräche ---
rag_talks_table = Table(
    "rag_talks",
    metadata,
    Column("talk_id", UUID(as_uuid=False), primary_key=True, server_default=text("gen_random_uuid()")),
    Column("collection", String(128), nullable=False),
    Column("user_id", String(128), nullable=False, server_default=""),
    Column("user_name", String(256), nullable=False, server_default=""),
    Column("title", Text, nullable=False),
    Column("personality", String(128), nullable=True),
    Column("summary", Text, nullable=True),
    Column("usage", JSONB().with_variant(JSON(), "sqlite"), nullable=True),
    Column("kontext_meta", JSONB().with_variant(JSON(), "sqlite"), nullable=True),
    Column("kontext_source_id", Text, nullable=True),
    Column("kontext_paragraph_id", Text, nullable=True),
    Column("kontext_paragraph", Text, nullable=True),
    Column("publishing_status", String(16), nullable=False, server_default="draft"),
    Column("pinned", Boolean, nullable=False, server_default=text("false")),
    Column("mode", Text, nullable=False, server_default="chat"),
    Column("compressed_up_to_turn_index", Integer, nullable=True),
    Column("compressed_summary", Text, nullable=True),
    Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
    Column("updated_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
)

Index("idx_rt_collection", rag_talks_table.c.collection)
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
    Column("personality", String(128), nullable=True),
    Column("user_message", Text, nullable=False),
    Column("assistant_message", Text, nullable=False),
    Column("usage", JSONB().with_variant(JSON(), "sqlite"), nullable=True),
    Column("collection", String(128), nullable=True),
    Column("chunk_index_map", JSONB().with_variant(JSON(), "sqlite"), nullable=True),
    Column("kontext_meta", JSONB().with_variant(JSON(), "sqlite"), nullable=True),
    Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
    Column("updated_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
)

Index("idx_rtu_talk_id", rag_turns_table.c.talk_id)
Index("idx_rtu_talk_id_index", rag_turns_table.c.talk_id, rag_turns_table.c.turn_index, unique=True)
Index("idx_rtu_created_at", rag_turns_table.c.created_at)


# --- rag_references: Normalisierte Referenzen pro Turn ---
rag_references_table = Table(
    "rag_references",
    metadata,
    Column("ref_id", UUID(as_uuid=False), primary_key=True, server_default=text("gen_random_uuid()")),
    Column(
        "turn_id",
        UUID(as_uuid=False),
        ForeignKey("rag_turns.turn_id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("ref_index", Integer, nullable=False),
    Column("chunk_id", String(64), nullable=True),
    Column("relevance", Float, nullable=True),
    Column("source_title", Text, nullable=True),
    Column("segment_title", Text, nullable=True),
    Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
)

Index("idx_rref_turn_id", rag_references_table.c.turn_id)
Index("idx_rref_chunk_id", rag_references_table.c.chunk_id)


# --- users: Dashboard users authenticated via GitHub OIDC ---
users_table = Table(
    "users",
    metadata,
    Column("user_id", UUID(as_uuid=False), primary_key=True, server_default=text("gen_random_uuid()")),
    Column("github_id", String(64), nullable=False, unique=True),
    Column("github_login", String(128), nullable=False),
    Column("email", String(256), nullable=True),
    Column("name", String(256), nullable=True),
    Column("avatar_url", Text, nullable=True),
    Column("role", String(32), nullable=False, server_default="viewer"),
    Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
    Column("updated_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
)

Index("idx_users_github_login", users_table.c.github_login)


# --- invitations: Einladungsbasierte Registrierung ---
invitations_table = Table(
    "invitations",
    metadata,
    Column("id", UUID(as_uuid=False), primary_key=True, server_default=text("gen_random_uuid()")),
    Column("inviter_user_id", String(128), nullable=False),
    Column("inviter_email", String(256), nullable=True),
    Column("invitee_email", String(256), nullable=False),
    Column("code", String(4), nullable=False),
    Column("status", String(16), nullable=False, server_default="pending"),
    Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
    Column("expires_at", DateTime(timezone=True), nullable=False),
    Column("redeemed_at", DateTime(timezone=True), nullable=True),
)

Index("idx_inv_invitee_code", invitations_table.c.invitee_email, invitations_table.c.code)
Index("idx_inv_inviter", invitations_table.c.inviter_user_id)

