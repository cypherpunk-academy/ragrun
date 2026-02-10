"""add references to rag_chunks

Stores chunk influence references as JSON for generated chunks.
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "rag_chunks",
        sa.Column("references", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("rag_chunks", "references")
