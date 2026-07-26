"""add publishing_status to rag_talks

Values:
  unknown   – auto-saved (final-save on exit)
  private   – not released
  peers     – released to friends
  candidate – released to all
  staged    – curated, not yet published
  archive   – curated, should not appear
  published – publicly available

One-time: all existing talks set to 'published'.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "0011"
down_revision = "0010"
branch_labels = None
depends_on = None

VALID_VALUES = ("unknown", "private", "peers", "candidate", "staged", "archive", "published")
CHECK_NAME = "rag_talks_publishing_status_check"


def upgrade() -> None:
    op.add_column(
        "rag_talks",
        sa.Column(
            "publishing_status",
            sa.String(16),
            nullable=False,
            server_default="unknown",
        ),
    )
    op.create_check_constraint(
        CHECK_NAME,
        "rag_talks",
        "publishing_status IN ('unknown','private','peers','candidate','staged','archive','published')",
    )
    # One-time: mark all existing talks as published
    op.execute("UPDATE rag_talks SET publishing_status = 'published'")


def downgrade() -> None:
    op.drop_constraint(CHECK_NAME, "rag_talks", type_="check")
    op.drop_column("rag_talks", "publishing_status")
