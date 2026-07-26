"""Add turn_id and talk_id FK columns to rag_usage.

Links usage rows to the specific turn and talk they belong to.
No backfill — existing rows keep NULL.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision = "0014"
down_revision = "0013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "rag_usage",
        sa.Column(
            "turn_id",
            UUID(as_uuid=False),
            sa.ForeignKey("rag_turns.turn_id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.add_column(
        "rag_usage",
        sa.Column(
            "talk_id",
            UUID(as_uuid=False),
            sa.ForeignKey("rag_talks.talk_id", ondelete="SET NULL"),
            nullable=True,
        ),
    )

    op.create_index("idx_ru_turn_id", "rag_usage", ["turn_id"])
    op.create_index("idx_ru_talk_id", "rag_usage", ["talk_id"])


def downgrade() -> None:
    op.drop_index("idx_ru_talk_id", table_name="rag_usage")
    op.drop_index("idx_ru_turn_id", table_name="rag_usage")
    op.drop_column("rag_usage", "talk_id")
    op.drop_column("rag_usage", "turn_id")
