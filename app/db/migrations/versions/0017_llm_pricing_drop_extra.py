"""Add llm_pricing table, drop rag_usage.extra column, create rag_usage_costs view.

Kosten werden nicht mehr gespeichert, sondern on-demand aus Tokenanzahl
und aktueller Preistabelle berechnet.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0017"
down_revision = "0016"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    tables = insp.get_table_names()

    # 1. Preistabelle (idempotent: DBs that already have llm_pricing from a partial/manual run)
    if "llm_pricing" not in tables:
        op.create_table(
            "llm_pricing",
            sa.Column("model", sa.String(128), primary_key=True),
            sa.Column("provider", sa.String(64), nullable=False, server_default="deepseek"),
            sa.Column("prompt_per_1m_usd", sa.Numeric(12, 6), nullable=False),
            sa.Column("completion_per_1m_usd", sa.Numeric(12, 6), nullable=False),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=False,
            ),
            sa.Column("note", sa.Text(), nullable=True),
        )

    # Startwerte (Stand April 2026); skip rows that already exist
    op.execute(
        sa.text(
            """
        INSERT INTO llm_pricing (model, provider, prompt_per_1m_usd, completion_per_1m_usd, note)
        VALUES
          ('deepseek-chat',           'deepseek', 0.27, 1.10, 'DeepSeek V3, cache miss'),
          ('deepseek-reasoner',       'deepseek', 0.55, 2.19, 'DeepSeek R1, cache miss'),
          ('claude-opus-4-6',         'anthropic', 15.0, 75.0, 'Claude Opus 4.6'),
          ('claude-sonnet-4-6',       'anthropic',  3.0, 15.0, 'Claude Sonnet 4.6'),
          ('claude-haiku-4-5-20251001','anthropic', 0.80,  4.0, 'Claude Haiku 4.5')
        ON CONFLICT (model) DO NOTHING
    """
        )
    )

    # 2. View für Kostensicht
    op.execute(sa.text("DROP VIEW IF EXISTS rag_usage_costs"))
    op.execute(
        sa.text(
            """
        CREATE VIEW rag_usage_costs AS
        SELECT
            u.id,
            u.account_id,
            u.thread_id,
            u.turn_id,
            u.talk_id,
            u.endpoint,
            u.model,
            u.provider,
            u.prompt_tokens,
            u.completion_tokens,
            u.total_tokens,
            u.created_at,
            ROUND(
                (COALESCE(u.prompt_tokens, 0)     / 1000000.0 * COALESCE(p.prompt_per_1m_usd, 0)
               + COALESCE(u.completion_tokens, 0) / 1000000.0 * COALESCE(p.completion_per_1m_usd, 0))::numeric,
                6
            ) AS cost_usd
        FROM rag_usage u
        LEFT JOIN llm_pricing p ON p.model = u.model
    """
        )
    )

    # 3. extra-Spalte droppen (idempotent)
    rag_cols = {c["name"] for c in insp.get_columns("rag_usage")}
    if "extra" in rag_cols:
        op.drop_column("rag_usage", "extra")


def downgrade() -> None:
    op.add_column(
        "rag_usage",
        sa.Column("extra", JSONB(), nullable=True),
    )

    op.execute(sa.text("DROP VIEW IF EXISTS rag_usage_costs"))
    op.drop_table("llm_pricing")
