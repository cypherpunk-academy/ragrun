"""rag_chunks: add deprecated_at column for soft-retiring orphaned chunks.

Chunks with deprecated_at IS NOT NULL are excluded from embedding (Qdrant /
vector_chunks) but kept in rag_chunks so that old references (e.g. chunk_ids
cited in rag_references) can still be resolved.

deprecated_at is set automatically when a rechunking run produces a different
set of chunk_ids for a source (orphan detection), or manually via the CLI.
An upsert of the same chunk_id always clears deprecated_at (chunk is active again).
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0019"
down_revision = "0018"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "rag_chunks",
        sa.Column("deprecated_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("rag_chunks", "deprecated_at")
