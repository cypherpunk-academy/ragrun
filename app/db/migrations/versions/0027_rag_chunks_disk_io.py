"""Cut rag_chunks disk IO: title trigram indexes, chunk_id lookup, HOT updates.

Title search was seq-scanning rag_chunks and detoasting text. Quote marking
looked up body chunks by chunk_id, which is not the leading primary-key column.
idx_rag_chunks_updated changed on every upsert, so updates could not be HOT.
"""
from __future__ import annotations

from alembic import op

revision = "0027_rag_chunks_disk_io"
down_revision = "0026"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # CONCURRENTLY cannot run inside the migration transaction.
    with op.get_context().autocommit_block():
        op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm WITH SCHEMA extensions")
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS idx_rag_chunks_updated")
        op.execute("ALTER TABLE rag_chunks SET (fillfactor = 90)")
        # Retry-safe: a failed CONCURRENTLY build leaves an invalid index.
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS idx_rag_chunks_chunk_id_active")
        op.execute(
            """
            CREATE INDEX CONCURRENTLY idx_rag_chunks_chunk_id_active
            ON rag_chunks (chunk_id)
            WHERE deprecated_at IS NULL
            """
        )
        # Not partial: the deployed title query has no deprecated_at predicate,
        # and a partial index would be ignored until that code is deployed.
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS idx_rag_chunks_segment_title_trgm")
        op.execute(
            """
            CREATE INDEX CONCURRENTLY idx_rag_chunks_segment_title_trgm
            ON rag_chunks USING gin ((metadata->>'segment_title') extensions.gin_trgm_ops)
            """
        )
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS idx_rag_chunks_source_title_trgm")
        op.execute(
            """
            CREATE INDEX CONCURRENTLY idx_rag_chunks_source_title_trgm
            ON rag_chunks USING gin ((metadata->>'source_title') extensions.gin_trgm_ops)
            """
        )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS idx_rag_chunks_source_title_trgm")
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS idx_rag_chunks_segment_title_trgm")
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS idx_rag_chunks_chunk_id_active")
        op.execute("ALTER TABLE rag_chunks RESET (fillfactor)")
        op.execute(
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_rag_chunks_updated
            ON rag_chunks (updated_at)
            """
        )
