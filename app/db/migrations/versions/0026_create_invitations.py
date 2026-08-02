"""create invitations table for invite-only registration"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0026"
down_revision = "0025"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS invitations (
            id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            inviter_user_id VARCHAR(128) NOT NULL,
            inviter_email   VARCHAR(256),
            invitee_email   VARCHAR(256) NOT NULL,
            code            VARCHAR(4)   NOT NULL,
            status          VARCHAR(16)  NOT NULL DEFAULT 'pending',
            created_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
            expires_at      TIMESTAMPTZ  NOT NULL,
            redeemed_at     TIMESTAMPTZ
        )
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_inv_invitee_code
        ON invitations (invitee_email, code)
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_inv_inviter
        ON invitations (inviter_user_id)
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_inv_inviter")
    op.execute("DROP INDEX IF EXISTS idx_inv_invitee_code")
    op.execute("DROP TABLE IF EXISTS invitations")
