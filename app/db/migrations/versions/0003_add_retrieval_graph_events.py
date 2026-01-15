"""add retrieval_graph_events table"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    metadata_type = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")

    op.create_table(
        "retrieval_graph_events",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("graph_event_id", sa.String(length=64), nullable=False),
        sa.Column("graph_name", sa.String(length=128), nullable=False),
        sa.Column("step", sa.String(length=128), nullable=False),
        sa.Column("concept", sa.String(length=512), nullable=False),
        sa.Column("worldview", sa.String(length=256), nullable=True),
        sa.Column("query_text", sa.Text(), nullable=True),
        sa.Column("prompt_messages", metadata_type, nullable=True),
        sa.Column("context_refs", metadata_type, nullable=True),
        sa.Column("context_text", sa.Text(), nullable=True),
        sa.Column("response_text", sa.Text(), nullable=True),
        sa.Column("retrieval_mode", sa.String(length=64), nullable=True),
        sa.Column("sufficiency", sa.String(length=32), nullable=True),
        sa.Column("errors", metadata_type, nullable=True),
        sa.Column("metadata", metadata_type, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    op.create_index("idx_rge_graph_event_id", "retrieval_graph_events", ["graph_event_id"])
    op.create_index("idx_rge_graph_event_step", "retrieval_graph_events", ["graph_event_id", "step"])
    op.create_index("idx_rge_graph_event_worldview", "retrieval_graph_events", ["graph_event_id", "worldview"])


def downgrade() -> None:
    op.drop_index("idx_rge_graph_event_worldview", table_name="retrieval_graph_events")
    op.drop_index("idx_rge_graph_event_step", table_name="retrieval_graph_events")
    op.drop_index("idx_rge_graph_event_id", table_name="retrieval_graph_events")
    op.drop_table("retrieval_graph_events")

