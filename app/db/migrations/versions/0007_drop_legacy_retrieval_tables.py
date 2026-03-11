"""drop legacy retrieval logging tables

retrieval_events, retrieval_chunks, retrieval_graph_events are superseded
by event_metadata and event_content (migration 0006).
"""
from __future__ import annotations

from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_index("idx_rge_graph_event_worldview", table_name="retrieval_graph_events")
    op.drop_index("idx_rge_graph_event_step", table_name="retrieval_graph_events")
    op.drop_index("idx_rge_graph_event_id", table_name="retrieval_graph_events")
    op.drop_table("retrieval_graph_events")
    op.drop_table("retrieval_chunks")
    op.drop_table("retrieval_events")


def downgrade() -> None:
    import sqlalchemy as sa
    from sqlalchemy.dialects import postgresql

    metadata_type = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")

    op.create_table(
        "retrieval_events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("concept", sa.String(length=512), nullable=False),
        sa.Column("branch", sa.String(length=128), nullable=False),
        sa.Column("collection", sa.String(length=128), nullable=False),
        sa.Column("answer", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_table(
        "retrieval_chunks",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("event_id", sa.Integer(), sa.ForeignKey("retrieval_events.id", ondelete="CASCADE"), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("source_id", sa.String(length=256), nullable=True),
        sa.Column("chunk_type", sa.String(length=128), nullable=True),
        sa.Column("metadata", metadata_type, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
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
        sa.Column("context_source", metadata_type, nullable=True),
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
