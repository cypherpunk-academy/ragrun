"""rag_chunks: rename collection -> rag_partition; book rows -> __shared__.

Shared corpus chunks (chunk_type book, secondary_book) live under rag_partition
__shared__. Assistant-specific rows keep rag_partition = assistant rag-collection.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy import text

revision = "0018"
down_revision = "0017"
branch_labels = None
depends_on = None

_SHARED = "__shared__"


def upgrade() -> None:
    op.drop_index("idx_rag_chunks_collection_embedded_at", table_name="rag_chunks")
    op.alter_column(
        "rag_chunks",
        "collection",
        new_column_name="rag_partition",
        existing_type=sa.String(length=128),
        existing_nullable=False,
    )
    op.execute(
        text(
            "UPDATE rag_chunks SET rag_partition = :shared "
            "WHERE chunk_type IN ('book', 'secondary_book')"
        ).bindparams(shared=_SHARED)
    )
    # Collapse duplicate shared rows (same chunk_id), keep newest updated_at.
    op.execute(
        text(
            """
            DELETE FROM rag_chunks a
            USING (
                SELECT ctid,
                       ROW_NUMBER() OVER (
                           PARTITION BY chunk_id
                           ORDER BY updated_at DESC NULLS LAST, source_id
                       ) AS rn
                FROM rag_chunks
                WHERE rag_partition = :shared
            ) d
            WHERE a.ctid = d.ctid AND d.rn > 1
            """
        ).bindparams(shared=_SHARED)
    )
    op.create_index(
        "idx_rag_chunks_rag_partition_embedded_at",
        "rag_chunks",
        ["rag_partition", "embedded_at"],
    )


def downgrade() -> None:
    op.drop_index("idx_rag_chunks_rag_partition_embedded_at", table_name="rag_chunks")
    op.alter_column(
        "rag_chunks",
        "rag_partition",
        new_column_name="collection",
        existing_type=sa.String(length=128),
        existing_nullable=False,
    )
    op.create_index(
        "idx_rag_chunks_collection_embedded_at",
        "rag_chunks",
        ["collection", "embedded_at"],
    )
