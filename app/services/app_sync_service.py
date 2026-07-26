"""Supabase RPC proxy for WatermelonDB sync (JWT forwarding)."""
from __future__ import annotations

from typing import Any

import httpx
from fastapi import HTTPException, status

from app.api.auth import AuthUser
from app.config import settings


def _supabase_rest_base() -> str:
    base = (settings.supabase_url or "").strip().rstrip("/")
    if not base:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Supabase URL is not configured on ragrun",
        )
    return f"{base}/rest/v1"


def _supabase_headers(user: AuthUser) -> dict[str, str]:
    anon = (settings.supabase_anon_key or "").strip()
    if not anon:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Supabase anon key is not configured on ragrun",
        )
    return {
        "Authorization": f"Bearer {user.raw_token}",
        "apikey": anon,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


async def pull_changes(
    user: AuthUser,
    *,
    last_pulled_at: int | None,
    schema_version: int | None,
) -> dict[str, Any]:
    url = f"{_supabase_rest_base()}/rpc/pull_changes"
    body = {
        "last_pulled_at": last_pulled_at or 0,
        "schema_version": schema_version or 1,
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=15.0)) as client:
        response = await client.post(url, json=body, headers=_supabase_headers(user))
    if response.status_code >= 400:
        detail = response.text
        try:
            parsed = response.json()
            if isinstance(parsed, dict):
                detail = str(parsed.get("message") or parsed.get("error") or detail)
        except Exception:
            pass
        raise HTTPException(status_code=response.status_code, detail=detail or "pull_changes failed")
    data = response.json()
    if not isinstance(data, dict):
        raise HTTPException(status_code=502, detail="pull_changes returned invalid payload")
    return data


async def push_changes(
    user: AuthUser,
    *,
    changes: dict[str, Any],
    last_pulled_at: int | None,
) -> None:
    url = f"{_supabase_rest_base()}/rpc/push_changes"
    body = {
        "changes": changes,
        "last_pulled_at": last_pulled_at or 0,
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=15.0)) as client:
        response = await client.post(url, json=body, headers=_supabase_headers(user))
    if response.status_code >= 400:
        detail = response.text
        try:
            parsed = response.json()
            if isinstance(parsed, dict):
                detail = str(parsed.get("message") or parsed.get("error") or detail)
        except Exception:
            pass
        raise HTTPException(status_code=response.status_code, detail=detail or "push_changes failed")
