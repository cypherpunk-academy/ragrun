"""Rename rag_chunks table to vector_chunks (Qdrant mirror).

The name rag_chunks is reserved for the new primary chunk store (0016).
"""
from __future__ import annotations

from alembic import op

revision = "0015"
down_revision = "0014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.rename_table("rag_chunks", "vector_chunks")
    # Free the name rag_chunks_pkey for the new rag_chunks table (0016).
    op.execute(
        "ALTER TABLE vector_chunks RENAME CONSTRAINT rag_chunks_pkey TO vector_chunks_pkey"
    )
    # Indexes keep their names until renamed; align names with new table purpose.
    op.execute(
        "ALTER INDEX IF EXISTS idx_chunks_source_id RENAME TO idx_vector_chunks_source_id"
    )
    op.execute(
        "ALTER INDEX IF EXISTS idx_chunks_created_at RENAME TO idx_vector_chunks_created_at"
    )


def downgrade() -> None:
    op.execute(
        "ALTER INDEX IF EXISTS idx_vector_chunks_created_at RENAME TO idx_chunks_created_at"
    )
    op.execute(
        "ALTER INDEX IF EXISTS idx_vector_chunks_source_id RENAME TO idx_chunks_source_id"
    )
    op.execute(
        "ALTER TABLE vector_chunks RENAME CONSTRAINT vector_chunks_pkey TO rag_chunks_pkey"
    )
    op.rename_table("vector_chunks", "rag_chunks")
