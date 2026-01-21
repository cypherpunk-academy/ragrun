"""add context_source to retrieval_graph_events

Stores source/segment titles (and optionally other non-text provenance) for retrieved contexts,
so we can avoid persisting large raw context windows in `context_text`.
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    metadata_type = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")
    op.add_column(
        "retrieval_graph_events",
        sa.Column("context_source", metadata_type, nullable=True),
    )


def downgrade() -> None:
    op.drop_column("retrieval_graph_events", "context_source")

