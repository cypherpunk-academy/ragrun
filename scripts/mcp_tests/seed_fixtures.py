#!/usr/bin/env python3
"""Seed durable [FIXTURE] work texts for Anton and Antonia.

IDs come from ``tests/mcp/mcp_testdata.yaml``. Existing rows with those IDs are
left unchanged. Cleanup only deletes ``[TEST]`` rows, so fixtures survive.

Usage:
  set -a && source .env.dev && set +a
  python scripts/mcp_tests/seed_fixtures.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[2]
TESTDATA = ROOT / "tests" / "mcp" / "mcp_testdata.yaml"

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


def _user_id(conn, email: str) -> str:
    uid = conn.execute(
        text("SELECT id::text FROM auth.users WHERE lower(email) = lower(:e)"),
        {"e": email},
    ).scalar()
    if not uid:
        print(f"No auth user for {email}", file=sys.stderr)
        sys.exit(1)
    return str(uid)


def _load_testdata() -> dict[str, Any]:
    data = yaml.safe_load(TESTDATA.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"unexpected testdata in {TESTDATA}")
    return data


def seed(conn, data: dict[str, Any]) -> tuple[int, int]:
    """Insert missing fixture notes. Returns (created, skipped)."""
    uid_a = _user_id(conn, EMAIL_A)
    uid_b = _user_id(conn, EMAIL_B)
    para = data["paragraph"]["para_id"]
    work = data["work_texts"]
    specs = [
        {
            "id": work["note_a"]["note_id"],
            "user_id": uid_a,
            "title": "[FIXTURE] Anton Notiz",
            "content": "[FIXTURE] Dauerhafte Notiz von Anton Testo für MCP-Tests.",
            "text_type": work["note_a"].get("text_type") or "note",
            "paragraph_id": para,
        },
        {
            "id": work["essay_a"]["note_id"],
            "user_id": uid_a,
            "title": "[FIXTURE] Anton Essay",
            "content": "[FIXTURE] Dauerhafter Essay von Anton Testo für MCP-Tests.",
            "text_type": work["essay_a"].get("text_type") or "essay",
            "paragraph_id": None,
        },
        {
            "id": work["note_b"]["note_id"],
            "user_id": uid_b,
            "title": "[FIXTURE] Antonia Notiz",
            "content": "[FIXTURE] Dauerhafte Notiz von Antonia Testa für MCP-Tests.",
            "text_type": work["note_b"].get("text_type") or "note",
            "paragraph_id": None,
        },
    ]
    created = 0
    skipped = 0
    for spec in specs:
        exists = conn.execute(
            text("SELECT 1 FROM app_notes WHERE id = :id AND deleted_at IS NULL"),
            {"id": spec["id"]},
        ).scalar()
        if exists:
            skipped += 1
            continue
        conn.execute(
            text(
                """
                INSERT INTO app_notes (
                  id, user_id, title, content, text_type, paragraph_id,
                  conversation_url, status, version, created_by
                )
                VALUES (
                  :id, CAST(:uid AS uuid), :title, :content, :text_type, :para,
                  NULL, 'draft', 1, 'user'
                )
                """
            ),
            {
                "id": spec["id"],
                "uid": spec["user_id"],
                "title": spec["title"],
                "content": spec["content"],
                "text_type": spec["text_type"],
                "para": spec["paragraph_id"],
            },
        )
        created += 1
    return created, skipped


def main() -> None:
    data = _load_testdata()
    engine = create_engine(_dsn())
    with engine.begin() as conn:
        created, skipped = seed(conn, data)
    print(f"fixture notes created: {created}")
    print(f"fixture notes skipped: {skipped}")


if __name__ == "__main__":
    main()
