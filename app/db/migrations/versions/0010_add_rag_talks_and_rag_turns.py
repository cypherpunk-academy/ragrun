"""add rag_talks and rag_turns tables

rag_talks: Single Source of Truth für Gespräche (collection, mensch, slug, title, usage).
rag_turns: Einzelne Gesprächsrunden mit Referenzen und per-Turn-Usage.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "0010"
down_revision = "0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "rag_talks",
        sa.Column(
            "talk_id",
            postgresql.UUID(as_uuid=False),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("collection", sa.String(128), nullable=False),
        sa.Column("mensch_id", sa.String(128), nullable=False, server_default=""),
        sa.Column("mensch_name", sa.String(256), nullable=False, server_default=""),
        sa.Column("slug", sa.String(256), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("action_id", sa.String(128), nullable=True),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("usage", postgresql.JSONB(), nullable=True),
        sa.Column("kontext_meta", postgresql.JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("idx_rt_collection", "rag_talks", ["collection"])
    op.create_index("idx_rt_collection_slug", "rag_talks", ["collection", "slug"], unique=True)
    op.create_index("idx_rt_created_at", "rag_talks", ["created_at"])

    op.create_table(
        "rag_turns",
        sa.Column(
            "turn_id",
            postgresql.UUID(as_uuid=False),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "talk_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("rag_talks.talk_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("turn_index", sa.Integer(), nullable=False),
        sa.Column("action_id", sa.String(128), nullable=True),
        sa.Column("assistant_personality", sa.String(128), nullable=True),
        sa.Column("user_message", sa.Text(), nullable=False),
        sa.Column("assistant_message", sa.Text(), nullable=False),
        sa.Column("usage", postgresql.JSONB(), nullable=True),
        sa.Column("references", postgresql.JSONB(), nullable=True),
        sa.Column("collection", sa.String(128), nullable=True),
        sa.Column("is_relay", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("chunk_index_map", postgresql.JSONB(), nullable=True),
        sa.Column("kontext_meta", postgresql.JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("idx_rtu_talk_id", "rag_turns", ["talk_id"])
    op.create_index("idx_rtu_talk_id_index", "rag_turns", ["talk_id", "turn_index"], unique=True)
    op.create_index("idx_rtu_created_at", "rag_turns", ["created_at"])


def downgrade() -> None:
    op.drop_index("idx_rtu_created_at", table_name="rag_turns")
    op.drop_index("idx_rtu_talk_id_index", table_name="rag_turns")
    op.drop_index("idx_rtu_talk_id", table_name="rag_turns")
    op.drop_table("rag_turns")

    op.drop_index("idx_rt_created_at", table_name="rag_talks")
    op.drop_index("idx_rt_collection_slug", table_name="rag_talks")
    op.drop_index("idx_rt_collection", table_name="rag_talks")
    op.drop_table("rag_talks")
