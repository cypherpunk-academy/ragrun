"""add rag_usage table for LLM billing tracking

Stores one row per LLM API call with token counts, model, provider and account_id.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "0009"
down_revision = ("0008", "0008b")
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "rag_usage",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("account_id", sa.String(128), nullable=False, server_default="anonymous"),
        sa.Column("thread_id", sa.String(64), nullable=True),
        sa.Column("endpoint", sa.String(256), nullable=True),
        sa.Column("model", sa.String(128), nullable=True),
        sa.Column("provider", sa.String(64), nullable=False, server_default="deepseek"),
        sa.Column("prompt_tokens", sa.Integer(), nullable=True),
        sa.Column("completion_tokens", sa.Integer(), nullable=True),
        sa.Column("total_tokens", sa.Integer(), nullable=True),
        sa.Column("extra", postgresql.JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("idx_ru_account_id", "rag_usage", ["account_id"])
    op.create_index("idx_ru_created_at", "rag_usage", ["created_at"])


def downgrade() -> None:
    op.drop_index("idx_ru_created_at", table_name="rag_usage")
    op.drop_index("idx_ru_account_id", table_name="rag_usage")
    op.drop_table("rag_usage")
