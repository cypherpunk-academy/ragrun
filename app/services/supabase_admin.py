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


async def create_user_and_get_otp(email: str) -> str:
    """Create a Supabase auth user and return a magic-link OTP in one step.

    Uses admin/generate_link with type=signup which creates the user
    AND returns the plain OTP — no email sent, no rate-limit issue.
    Falls back to create_user + separate generate_link if user already exists.
    """
    base = (settings.supabase_url or "").rstrip("/")
    key = settings.supabase_service_role_key
    if not base or not key:
        raise RuntimeError("supabase_url and supabase_service_role_key must be configured")

    headers = {
        "Authorization": f"Bearer {key}",
        "apikey": key,
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
        # Try signup link — creates user + returns OTP in one call
        resp = await client.post(
            f"{base}/auth/v1/admin/generate_link",
            json={"type": "signup", "email": email},
            headers=headers,
        )

        if resp.status_code == 422:
            # User already exists — create a magiclink OTP instead
            logger.info("User %s already exists, generating magiclink OTP", email)
            resp = await client.post(
                f"{base}/auth/v1/admin/generate_link",
                json={"type": "magiclink", "email": email},
                headers=headers,
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

    email_lower = email.lower().strip()
    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
        # GoTrue Admin: use `email` query param (exact). The PostgREST-style
        # `filter=email eq …` returns an empty list on current Auth versions.
        resp = await client.get(
            f"{base}/auth/v1/admin/users",
            params={"email": email_lower, "page": 1, "per_page": 1},
            headers={
                "Authorization": f"Bearer {key}",
                "apikey": key,
            },
        )
    resp.raise_for_status()
    data = resp.json()
    users = data.get("users", [])
    return any(u.get("email", "").lower() == email_lower for u in users)
