"""Create rag_chunks as the primary chunk store (DB-first).

vector_chunks remains the mirror of Qdrant payloads. rag_chunks holds all chunks
before/after embedding; embedded_at marks rows synced to Qdrant/vector_chunks.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0016"
down_revision = "0015"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # DBs that ran 0015 before the PK rename was added still have rag_chunks_pkey on vector_chunks.
    op.execute(
        """
        DO $rename_pk$
        BEGIN
            IF EXISTS (
                SELECT 1
                FROM pg_constraint c
                JOIN pg_class t ON t.oid = c.conrelid
                WHERE c.conname = 'rag_chunks_pkey'
                  AND t.relname = 'vector_chunks'
            ) THEN
                ALTER TABLE vector_chunks RENAME CONSTRAINT rag_chunks_pkey TO vector_chunks_pkey;
            END IF;
        END
        $rename_pk$;
        """
    )
    op.create_table(
        "rag_chunks",
        sa.Column("collection", sa.String(length=128), nullable=False),
        sa.Column("chunk_id", sa.String(length=256), nullable=False),
        sa.Column("source_id", sa.String(length=256), nullable=False),
        sa.Column("chunk_type", sa.String(length=64), nullable=False),
        sa.Column("language", sa.String(length=8), nullable=False),
        sa.Column("worldviews", postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column("importance", sa.Integer(), nullable=True),
        sa.Column("content_hash", sa.String(length=128), nullable=False),
        sa.Column("text", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("references", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("scope", sa.String(length=64), nullable=True),
        sa.Column("embedded_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("collection", "chunk_id", name="rag_chunks_pkey"),
    )
    op.create_index("idx_rag_chunks_source_id", "rag_chunks", ["source_id"])
    op.execute("CREATE INDEX idx_rag_chunks_created_at ON rag_chunks (created_at DESC)")
    op.create_index(
        "idx_rag_chunks_collection_embedded_at",
        "rag_chunks",
        ["collection", "embedded_at"],
    )


def downgrade() -> None:
    op.drop_index("idx_rag_chunks_collection_embedded_at", table_name="rag_chunks")
    op.execute("DROP INDEX IF EXISTS idx_rag_chunks_created_at")
    op.drop_index("idx_rag_chunks_source_id", table_name="rag_chunks")
    op.drop_table("rag_chunks")
