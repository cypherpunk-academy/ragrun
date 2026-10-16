"""Extra headers for outbound calls to Qdrant/embeddings (e.g. Cloudflare Access)."""
from __future__ import annotations

from typing import Mapping

from app.config import settings


def merge_service_http_headers(base: Mapping[str, str] | None) -> dict[str, str] | None:
    """Merge Qdrant/api-key headers with optional Cloudflare Access service token."""
    headers: dict[str, str] = dict(base) if base else {}
    client_id = (getattr(settings, "cf_access_client_id", None) or "").strip()
    client_secret = (getattr(settings, "cf_access_client_secret", None) or "").strip()
    if client_id and client_secret:
        headers["CF-Access-Client-Id"] = client_id
        headers["CF-Access-Client-Secret"] = client_secret
    return headers or None
