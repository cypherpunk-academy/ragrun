"""Thin wrapper around the Supabase Admin API (service-role key)."""
from __future__ import annotations

import logging

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


async def create_user(email: str) -> dict:
    """Create a Supabase auth user with confirmed email via Admin API."""
    base = (settings.supabase_url or "").rstrip("/")
    key = settings.supabase_service_role_key
    if not base or not key:
        raise RuntimeError("supabase_url and supabase_service_role_key must be configured")

    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
        resp = await client.post(
            f"{base}/auth/v1/admin/users",
            json={"email": email, "email_confirm": True},
            headers={
                "Authorization": f"Bearer {key}",
                "apikey": key,
                "Content-Type": "application/json",
            },
        )
    if resp.status_code == 422:
        body = resp.json()
        msg = body.get("msg") or body.get("message") or str(body)
        if "already been registered" in msg.lower() or "already exists" in msg.lower():
            logger.info("User %s already exists in Supabase", email)
            return {"email": email, "already_exists": True}
        resp.raise_for_status()
    resp.raise_for_status()
    return resp.json()


async def generate_magic_link_otp(email: str) -> str:
    """Generate a magic link via Admin API and return the plain OTP code.

    This avoids sending a second email — the caller can pass the OTP
    directly to the client for immediate verification.
    """
    base = (settings.supabase_url or "").rstrip("/")
    key = settings.supabase_service_role_key
    if not base or not key:
        raise RuntimeError("supabase_url and supabase_service_role_key must be configured")

    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
        resp = await client.post(
            f"{base}/auth/v1/admin/generate_link",
            json={"type": "magiclink", "email": email},
            headers={
                "Authorization": f"Bearer {key}",
                "apikey": key,
                "Content-Type": "application/json",
            },
        )
    resp.raise_for_status()
    data = resp.json()
    otp = data.get("properties", {}).get("email_otp")
    if not otp:
        raise RuntimeError("Supabase did not return an email_otp in generate_link response")
    return otp


async def check_user_exists(email: str) -> bool:
    """Check whether a Supabase auth user with this email exists."""
    base = (settings.supabase_url or "").rstrip("/")
    key = settings.supabase_service_role_key
    if not base or not key:
        raise RuntimeError("supabase_url and supabase_service_role_key must be configured")

    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
        resp = await client.get(
            f"{base}/auth/v1/admin/users",
            params={"filter": f"email eq {email}", "page": 1, "per_page": 1},
            headers={
                "Authorization": f"Bearer {key}",
                "apikey": key,
            },
        )
    resp.raise_for_status()
    data = resp.json()
    users = data.get("users", [])
    return any(u.get("email", "").lower() == email.lower() for u in users)
