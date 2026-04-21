"""rename begrifflist to begriff

Revision ID: 0008b
Revises: 0007
Create Date: 2026-03-30
"""
from alembic import op

revision = "0008b"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("UPDATE rag_chunks SET chunk_type = 'begriff' WHERE chunk_type = 'begriff_list'")


def downgrade() -> None:
    op.execute("UPDATE rag_chunks SET chunk_type = 'begriff_list' WHERE chunk_type = 'begriff'")
