"""add users table for dashboard authentication"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0020"
down_revision = "0019"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("user_id", sa.UUID(), nullable=False, server_default=sa.text("gen_random_uuid()")),
        sa.Column("github_id", sa.String(length=64), nullable=False),
        sa.Column("github_login", sa.String(length=128), nullable=False),
        sa.Column("email", sa.String(length=256), nullable=True),
        sa.Column("name", sa.String(length=256), nullable=True),
        sa.Column("avatar_url", sa.Text(), nullable=True),
        sa.Column("role", sa.String(length=32), nullable=False, server_default="viewer"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("user_id"),
        sa.UniqueConstraint("github_id"),
    )
    op.create_index("idx_users_github_login", "users", ["github_login"], unique=False)


def downgrade() -> None:
    op.drop_index("idx_users_github_login", table_name="users")
    op.drop_table("users")
