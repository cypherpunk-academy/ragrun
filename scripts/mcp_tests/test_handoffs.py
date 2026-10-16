#!/usr/bin/env python3
"""Refresh [TEST] handoffs for Philo MCP test plan.

Deletes prior [TEST] handoffs for Anton Testo and Antonia Testa, inserts:
  - one valid handoff for Anton (24h TTL)
  - one expired handoff for Anton
  - one valid handoff for Antonia

Prints handoff_id, handoff_old, handoff_b on stdout (YAML-friendly).

Usage:
  set -a && source .env.dev && set +a
  python scripts/mcp_tests/test_handoffs.py
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EMAIL_A = "anton.testo@ragxxx.com"
EMAIL_B = "antonia.testa@ragxxx.com"
SOURCE_ID = "4b8e4c2a-3f1b-4d2e-9c4a-8e4f4b2c3a1d"
SEGMENT = "xiii-der-wert-des-lebens-pessimismus-und-optimismus"
PARA_ID = "6dd87d4c-73a3-4796-b690-61cba295afd1"

_HANDOFF_CHARS = "abcdefghijkmnpqrstuvwxyz23456789"


def _rand_handoff_id(length: int = 5) -> str:
    return "".join(random.choice(_HANDOFF_CHARS) for _ in range(length))


def _dsn() -> str:
    raw = os.environ.get("RAGRUN_POSTGRES_DSN") or os.environ.get("DATABASE_URL") or ""
    if not raw:
        print("Missing RAGRUN_POSTGRES_DSN", file=sys.stderr)
        sys.exit(1)
    if raw.startswith("postgresql://"):
        return raw.replace("postgresql://", "postgresql+psycopg://", 1)
    return raw


def _user_id(conn, email: str) -> str:
    uid = conn.execute(
        text("SELECT id::text FROM auth.users WHERE lower(email) = lower(:e)"),
        {"e": email},
    ).scalar()
    if not uid:
        print(f"No auth user for {email}", file=sys.stderr)
        sys.exit(1)
    return str(uid)


def main() -> None:
    engine = create_engine(_dsn())
    handoff_id = _rand_handoff_id()
    handoff_old = _rand_handoff_id()
    handoff_b = _rand_handoff_id()

    with engine.begin() as conn:
        uid_a = _user_id(conn, EMAIL_A)
        uid_b = _user_id(conn, EMAIL_B)

        conn.execute(
            text(
                """
                DELETE FROM handoffs
                WHERE user_id IN (CAST(:a AS uuid), CAST(:b AS uuid))
                  AND (
                    COALESCE(marked_text, '') LIKE '[TEST]%'
                    OR COALESCE(user_question, '') LIKE '[TEST]%'
                  )
                """
            ),
            {"a": uid_a, "b": uid_b},
        )

        conn.execute(
            text(
                """
                INSERT INTO handoffs (
                  id, user_id, paragraph_id, source_id, segment_slug,
                  marked_text, user_question, return_url, expires_at
                )
                VALUES
                  (:h1, CAST(:a AS uuid), :para, :sid, :seg, :mt1, :q1, :url, now() + interval '24 hours'),
                  (:h2, CAST(:a AS uuid), :para, :sid, :seg, :mt2, :q2, :url, now() - interval '1 hour'),
                  (:h3, CAST(:b AS uuid), :para, :sid, :seg, :mt3, :q3, :url, now() + interval '24 hours')
                """
            ),
            {
                "a": uid_a,
                "b": uid_b,
                "h1": handoff_id,
                "h2": handoff_old,
                "h3": handoff_b,
                "para": PARA_ID,
                "sid": SOURCE_ID,
                "seg": SEGMENT,
                "url": "https://dev.example/philo/return",
                "mt1": "[TEST] markierter Text aus Absatz 1 (Anton gültig)",
                "q1": "[TEST] Was meint Steiner hier mit Lebenszweck?",
                "mt2": "[TEST] abgelaufener Handoff Anton",
                "q2": "[TEST] Alte Frage (abgelaufen)",
                "mt3": "[TEST] markierter Text Antonia",
                "q3": "[TEST] Antonia Handoff-Frage",
            },
        )

    print("handoff_id:", handoff_id)
    print("handoff_old:", handoff_old)
    print("handoff_b:", handoff_b)


if __name__ == "__main__":
    main()
