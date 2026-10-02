#!/usr/bin/env python3
"""Remove [TEST] MCP fixture data for Anton Testo and Antonia Testa.

Deletes app_notes, protocol_entries, empty protocols, and handoffs that are
tagged with the [TEST] prefix (title, content, marked_text, or user_question).
Rows tagged [FIXTURE] are left alone.

Usage:
  set -a && source .env.dev && set +a
  python scripts/mcp_tests/test_cleanup.py
"""
from __future__ import annotations

import os
import sys

from sqlalchemy import create_engine, text

EMAIL_A = "anton.testo@ragxxx.com"
EMAIL_B = "antonia.testa@ragxxx.com"


def _dsn() -> str:
    raw = os.environ.get("RAGRUN_POSTGRES_DSN") or os.environ.get("DATABASE_URL") or ""
    if not raw:
        print("Missing RAGRUN_POSTGRES_DSN", file=sys.stderr)
        sys.exit(1)
    if raw.startswith("postgresql://"):
        return raw.replace("postgresql://", "postgresql+psycopg://", 1)
    return raw


def _user_ids(conn) -> tuple[str, str]:
    ids: list[str] = []
    for email in (EMAIL_A, EMAIL_B):
        uid = conn.execute(
            text("SELECT id::text FROM auth.users WHERE lower(email) = lower(:e)"),
            {"e": email},
        ).scalar()
        if not uid:
            print(f"No auth user for {email}", file=sys.stderr)
            sys.exit(1)
        ids.append(str(uid))
    return ids[0], ids[1]


def main() -> None:
    engine = create_engine(_dsn())
    with engine.begin() as conn:
        uid_a, uid_b = _user_ids(conn)

        n_notes = conn.execute(
            text(
                """
                WITH deleted AS (
                  DELETE FROM app_notes
                  WHERE user_id IN (CAST(:a AS uuid), CAST(:b AS uuid))
                    AND deleted_at IS NULL
                    AND (
                      title LIKE '[TEST]%'
                      OR content LIKE '[TEST]%'
                    )
                  RETURNING 1
                )
                SELECT count(*) FROM deleted
                """
            ),
            {"a": uid_a, "b": uid_b},
        ).scalar()

        n_entries = conn.execute(
            text(
                """
                WITH deleted AS (
                  DELETE FROM protocol_entries pe
                  USING protocols p
                  WHERE pe.protocol_id = p.id
                    AND p.user_id IN (CAST(:a AS uuid), CAST(:b AS uuid))
                    AND pe.content LIKE '[TEST]%'
                  RETURNING 1
                )
                SELECT count(*) FROM deleted
                """
            ),
            {"a": uid_a, "b": uid_b},
        ).scalar()

        n_protocols = conn.execute(
            text(
                """
                WITH deleted AS (
                  DELETE FROM protocols p
                  WHERE p.user_id IN (CAST(:a AS uuid), CAST(:b AS uuid))
                    AND NOT EXISTS (
                      SELECT 1 FROM protocol_entries pe WHERE pe.protocol_id = p.id
                    )
                  RETURNING 1
                )
                SELECT count(*) FROM deleted
                """
            ),
            {"a": uid_a, "b": uid_b},
        ).scalar()

        n_handoffs = conn.execute(
            text(
                """
                WITH deleted AS (
                  DELETE FROM handoffs
                  WHERE user_id IN (CAST(:a AS uuid), CAST(:b AS uuid))
                    AND (
                      COALESCE(marked_text, '') LIKE '[TEST]%'
                      OR COALESCE(user_question, '') LIKE '[TEST]%'
                    )
                  RETURNING 1
                )
                SELECT count(*) FROM deleted
                """
            ),
            {"a": uid_a, "b": uid_b},
        ).scalar()

    print(f"deleted app_notes: {n_notes}")
    print(f"deleted protocol_entries: {n_entries}")
    print(f"deleted empty protocols: {n_protocols}")
    print(f"deleted handoffs: {n_handoffs}")
    print(f"total: {n_notes + n_entries + n_protocols + n_handoffs}")


if __name__ == "__main__":
    main()
