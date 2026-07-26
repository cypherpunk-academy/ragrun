"""Rename talk publishing_status values, add bug_description.

- unknown -> draft
- private -> personal
- add bug, draft, personal to check constraint; remove unknown, private
- add nullable bug_description (text)
- server default publishing_status: draft
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0012"
down_revision = "0011"
branch_labels = None
depends_on = None

CHECK_NAME = "rag_talks_publishing_status_check"
NEW_CHECK = (
    "publishing_status IN ("
    "'draft','personal','peers','candidate','staged','archive','published','bug'"
    ")"
)


def upgrade() -> None:
    # Drop check so we can rename / migrate values
    op.drop_constraint(CHECK_NAME, "rag_talks", type_="check")

    op.execute("UPDATE rag_talks SET publishing_status = 'draft' WHERE publishing_status = 'unknown'")
    op.execute("UPDATE rag_talks SET publishing_status = 'personal' WHERE publishing_status = 'private'")

    op.add_column(
        "rag_talks",
        sa.Column("bug_description", sa.Text(), nullable=True),
    )

    op.execute(sa.text("ALTER TABLE rag_talks ALTER COLUMN publishing_status SET DEFAULT 'draft'"))

    op.create_check_constraint(CHECK_NAME, "rag_talks", NEW_CHECK)


def downgrade() -> None:
    op.drop_constraint(CHECK_NAME, "rag_talks", type_="check")

    op.execute(sa.text("ALTER TABLE rag_talks ALTER COLUMN publishing_status SET DEFAULT 'unknown'"))

    op.drop_column("rag_talks", "bug_description")

    op.execute("UPDATE rag_talks SET publishing_status = 'unknown' WHERE publishing_status = 'draft'")
    op.execute("UPDATE rag_talks SET publishing_status = 'private' WHERE publishing_status = 'personal'")

    # Remove rows with 'bug' (cannot map 1:1) — set to personal as fallback
    op.execute("UPDATE rag_talks SET publishing_status = 'personal' WHERE publishing_status = 'bug'")

    op.create_check_constraint(
        CHECK_NAME,
        "rag_talks",
        "publishing_status IN ('unknown','private','peers','candidate','staged','archive','published')",
    )
