"""Enable RLS and revoke Data API grants on server-only tables.

users, invitations, and alembic_version were publicly writable via the
anon/authenticated roles because RLS was off. The API uses the Postgres
session role (bypasses RLS). No policies and no FORCE: Data API returns
no rows; server path keeps working.
"""
from __future__ import annotations

from alembic import op

revision = "0028_enable_rls_server_tables"
down_revision = "0027_rag_chunks_disk_io"
branch_labels = None
depends_on = None

_TABLES = ("users", "invitations", "alembic_version")


def upgrade() -> None:
    for table in _TABLES:
        op.execute(f"ALTER TABLE public.{table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"REVOKE ALL ON TABLE public.{table} FROM anon, authenticated")


def downgrade() -> None:
    for table in _TABLES:
        op.execute(
            f"GRANT ALL ON TABLE public.{table} TO anon, authenticated"
        )
        op.execute(f"ALTER TABLE public.{table} DISABLE ROW LEVEL SECURITY")
