"""add event_metadata and event_content tables

Dual-table event recording: event_metadata for permanent stats (no concept),
event_content for 7-day retention (content, prompts, responses).
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    metadata_type = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")

    op.create_table(
        "event_metadata",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("endpoint", sa.String(length=256), nullable=False),
        sa.Column("graph_event_id", sa.String(length=64), nullable=True),
        sa.Column("graph_name", sa.String(length=128), nullable=True),
        sa.Column("step", sa.String(length=128), nullable=True),
        sa.Column("collection", sa.String(length=128), nullable=True),
        sa.Column("worldview", sa.String(length=256), nullable=True),
        sa.Column("retrieval_mode", sa.String(length=64), nullable=True),
        sa.Column("sufficiency", sa.String(length=32), nullable=True),
        sa.Column("error_count", sa.Integer(), nullable=True),
        sa.Column("metadata", metadata_type, nullable=True),
        sa.Column("chunk_ids", metadata_type, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("idx_em_endpoint", "event_metadata", ["endpoint"])
    op.create_index("idx_em_created_at", "event_metadata", ["created_at"])
    op.create_index("idx_em_graph_event_id", "event_metadata", ["graph_event_id"])

    op.create_table(
        "event_content",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("event_metadata_id", sa.BigInteger(), sa.ForeignKey("event_metadata.id", ondelete="CASCADE"), nullable=False),
        sa.Column("concept", sa.String(length=512), nullable=True),
        sa.Column("query_text", sa.Text(), nullable=True),
        sa.Column("prompt_messages", metadata_type, nullable=True),
        sa.Column("context_refs", metadata_type, nullable=True),
        sa.Column("context_source", metadata_type, nullable=True),
        sa.Column("context_text", sa.Text(), nullable=True),
        sa.Column("response_text", sa.Text(), nullable=True),
        sa.Column("errors", metadata_type, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("idx_ec_event_metadata_id", "event_content", ["event_metadata_id"])
    op.create_index("idx_ec_created_at", "event_content", ["created_at"])


def downgrade() -> None:
    op.drop_index("idx_ec_created_at", table_name="event_content")
    op.drop_index("idx_ec_event_metadata_id", table_name="event_content")
    op.drop_table("event_content")
    op.drop_index("idx_em_graph_event_id", table_name="event_metadata")
    op.drop_index("idx_em_created_at", table_name="event_metadata")
    op.drop_index("idx_em_endpoint", table_name="event_metadata")
    op.drop_table("event_metadata")
