"""Add kontext_source_id, kontext_segment_id, kontext_paragraph to rag_talks.

For mit-kontext talks: denormalized query keys alongside kontext_meta JSONB.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0021"
down_revision = "0020"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("rag_talks", sa.Column("kontext_source_id", sa.Text(), nullable=True))
    op.add_column("rag_talks", sa.Column("kontext_segment_id", sa.Text(), nullable=True))
    op.add_column("rag_talks", sa.Column("kontext_paragraph", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("rag_talks", "kontext_paragraph")
    op.drop_column("rag_talks", "kontext_segment_id")
    op.drop_column("rag_talks", "kontext_source_id")
