"""add thread_id to event_content

Denormalisiert graph_event_id (thread_id) in event_content fuer einfache
Abfragen ohne JOIN auf event_metadata.
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "event_content",
        sa.Column("thread_id", sa.String(length=64), nullable=True),
    )
    op.create_index("idx_ec_thread_id", "event_content", ["thread_id"])


def downgrade() -> None:
    op.drop_index("idx_ec_thread_id", table_name="event_content")
    op.drop_column("event_content", "thread_id")
