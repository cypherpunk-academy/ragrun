#!/usr/bin/env python3
"""Mint a short-lived Supabase access token for Anton or Antonia.

Uses the admin generate_link + verify flow (same pattern as ragapp
``scripts/dev-login.mjs``). Tokens stay in memory / env; this module never
prints them.

Usage:
  set -a && source .env.dev && set +a
  python scripts/mcp_tests/mint.py --user a
  # prints only: minted <email>
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EMAIL_A = "anton.testo@ragxxx.com"
EMAIL_B = "antonia.testa@ragxxx.com"
USER_EMAILS = {
    "a": EMAIL_A,
    "anton": EMAIL_A,
    "b": EMAIL_B,
    "antonia": EMAIL_B,
}


def load_env_files(*paths: Path) -> None:
    """Fill missing variables from env files. Existing values win."""
    for path in paths:
        if not path.is_file():
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def resolve_email(user: str) -> str:
    key = user.strip().lower()
    if key not in USER_EMAILS:
        raise ValueError(f"unbekannter User {user!r}; erwartet a|b|anton|antonia")
    return USER_EMAILS[key]


def _supabase_url() -> str:
    raw = (
        os.environ.get("RAGRUN_SUPABASE_URL")
        or os.environ.get("EXPO_PUBLIC_SUPABASE_URL")
        or os.environ.get("SUPABASE_URL")
        or ""
    ).strip().rstrip("/")
    if not raw:
        raise RuntimeError("RAGRUN_SUPABASE_URL fehlt")
    return raw


def _anon_key() -> str:
    raw = (
        os.environ.get("RAGRUN_SUPABASE_ANON_KEY")
        or os.environ.get("EXPO_PUBLIC_SUPABASE_ANON_KEY")
        or os.environ.get("SUPABASE_ANON_KEY")
        or ""
    ).strip()
    if not raw:
        raise RuntimeError("RAGRUN_SUPABASE_ANON_KEY fehlt")
    return raw


def _service_role_key() -> str:
    raw = (
        os.environ.get("RAGRUN_SUPABASE_SERVICE_ROLE_KEY")
        or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        or ""
    ).strip()
    if not raw:
        raise RuntimeError("RAGRUN_SUPABASE_SERVICE_ROLE_KEY fehlt")
    return raw


def mint_access_token(email: str, *, timeout: float = 30) -> str:
    """Return a fresh access_token for ``email``. Never logs the token."""
    base = _supabase_url()
    anon = _anon_key()
    service = _service_role_key()

    link = httpx.post(
        f"{base}/auth/v1/admin/generate_link",
        headers={
            "apikey": service,
            "Authorization": f"Bearer {service}",
            "Content-Type": "application/json",
        },
        json={"type": "magiclink", "email": email},
        timeout=timeout,
    )
    link.raise_for_status()
    payload: dict[str, Any] = link.json()
    hashed = payload.get("hashed_token")
    if not hashed and isinstance(payload.get("properties"), dict):
        hashed = payload["properties"].get("hashed_token")
    if not hashed:
        raise RuntimeError("generate_link lieferte kein hashed_token")

    verify = httpx.post(
        f"{base}/auth/v1/verify",
        headers={
            "apikey": anon,
            "Authorization": f"Bearer {anon}",
            "Content-Type": "application/json",
        },
        json={"type": "magiclink", "token_hash": hashed},
        timeout=timeout,
    )
    verify.raise_for_status()
    session = verify.json()
    token = session.get("access_token")
    if not token:
        raise RuntimeError("verify lieferte kein access_token")
    return str(token)


def mint_user(user: str) -> str:
    return mint_access_token(resolve_email(user))


def main() -> int:
    parser = argparse.ArgumentParser(description="Mint MCP test-user access token")
    parser.add_argument("--user", required=True, help="a|b|anton|antonia")
    parser.add_argument(
        "--env-file",
        action="append",
        default=[],
        help="zusätzliche .env-Datei (mehrfach möglich)",
    )
    parser.add_argument(
        "--export",
        choices=("a", "b"),
        default=None,
        help="setzt MCP_TOKEN_USER_A oder MCP_TOKEN_USER_B in der aktuellen shell nicht; "
        "nur für Aufrufer die os.environ lesen",
    )
    args = parser.parse_args()
    for path in args.env_file:
        load_env_files(Path(path))
    load_env_files(ROOT / ".env.dev", ROOT / ".env", ROOT / "tests" / ".env.mcp")

    email = resolve_email(args.user)
    token = mint_access_token(email)
    if args.export == "a":
        os.environ["MCP_TOKEN_USER_A"] = token
    elif args.export == "b":
        os.environ["MCP_TOKEN_USER_B"] = token
    print(f"minted {email}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
