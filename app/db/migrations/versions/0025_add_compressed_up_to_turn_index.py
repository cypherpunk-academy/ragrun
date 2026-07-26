"""Add compressed_up_to_turn_index, compressed_summary to rag_talks.

Welle 5b — „Verdichten": fasst ältere Turns eines Gesprächs zusammen und
merkt sich, bis zu welchem turn_index bereits verdichtet wurde.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0025_compressed_summary"
down_revision = "0024_pinned_mode_rag_talks"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "rag_talks",
        sa.Column("compressed_up_to_turn_index", sa.Integer(), nullable=True),
    )
    op.add_column(
        "rag_talks",
        sa.Column("compressed_summary", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("rag_talks", "compressed_summary")
    op.drop_column("rag_talks", "compressed_up_to_turn_index")
