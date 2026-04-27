"""Extract rag_turns.references JSONB into rag_references table.

One row per reference, ordered by ref_index.
Drops rag_turns.references column after migration.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "0013"
down_revision = "0012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "rag_references",
        sa.Column(
            "ref_id",
            UUID(as_uuid=False),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "turn_id",
            UUID(as_uuid=False),
            sa.ForeignKey("rag_turns.turn_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("ref_index", sa.Integer(), nullable=False),
        sa.Column("chunk_id", sa.String(64), nullable=True),
        sa.Column("relevance", sa.Float(), nullable=True),
        sa.Column("source_title", sa.Text(), nullable=True),
        sa.Column("segment_title", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    op.create_index("idx_rref_turn_id", "rag_references", ["turn_id"])
    op.create_index("idx_rref_chunk_id", "rag_references", ["chunk_id"])

    # Migrate existing data from rag_turns.references JSONB array
    op.execute(sa.text("""
        INSERT INTO rag_references
            (turn_id, ref_index, chunk_id, relevance, source_title, segment_title)
        SELECT
            t.turn_id,
            (r.ordinality - 1)::int AS ref_index,
            r.value->>'chunk_id',
            (r.value->>'relevance')::float,
            r.value->>'source_title',
            r.value->>'segment_title'
        FROM rag_turns t,
             jsonb_array_elements(t.references) WITH ORDINALITY AS r(value, ordinality)
        WHERE t.references IS NOT NULL
          AND jsonb_typeof(t.references) = 'array'
    """))

    op.drop_column("rag_turns", "references")


def downgrade() -> None:
    op.add_column(
        "rag_turns",
        sa.Column("references", JSONB(), nullable=True),
    )

    op.drop_index("idx_rref_chunk_id", table_name="rag_references")
    op.drop_index("idx_rref_turn_id", table_name="rag_references")
    op.drop_table("rag_references")
