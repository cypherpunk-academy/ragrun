"""Shared-secret authentication for internal CLI routes (/api/v1/rag/*, /api/v1/admin/*).

When RAGRUN_INTERNAL_API_KEY is set, every request to those routes must supply
the matching value in the X-Api-Key header.  When the setting is empty the
dependency is a no-op, preserving backwards compatibility for local/LAN setups.
"""
from __future__ import annotations

import secrets

from fastapi import Header, HTTPException, status

from app.config import settings


async def require_internal_key(x_api_key: str = Header(default="", alias="X-Api-Key")) -> None:
    """FastAPI dependency: enforce X-Api-Key when RAGRUN_INTERNAL_API_KEY is configured."""
    expected = (settings.internal_api_key or "").strip()
    if not expected:
        return  # key not configured → open access (local / LAN mode)
    provided = (x_api_key or "").strip()
    if not provided:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-Api-Key header",
        )
    if not secrets.compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
