"""Add pinned, mode to rag_talks.

pinned: Welle 5a — Gespraeche pinnen, schuetzt vor Nacht-Cleanup unpinned Talks.
mode: Welle 5c — Chat/Nachdenken-Modus, persistiert je Talk.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0024_pinned_mode_rag_talks"
down_revision = "0023_rag_paragraphs_uuid"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "rag_talks",
        sa.Column("pinned", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column(
        "rag_talks",
        sa.Column("mode", sa.Text(), nullable=False, server_default="chat"),
    )


def downgrade() -> None:
    op.drop_column("rag_talks", "mode")
    op.drop_column("rag_talks", "pinned")
