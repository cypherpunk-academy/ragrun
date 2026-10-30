"""Public paths for static chapter HTML."""
from __future__ import annotations

from app.config import settings


def chapter_object_key(source_id: str, segment_slug: str) -> str:
    prefix = (settings.r2_prefix or "").strip().strip("/")
    rel = f"books/{source_id}/{segment_slug}.html"
    return f"{prefix}/{rel}" if prefix else rel


def chapter_public_url(
    source_id: str,
    segment_slug: str,
    paragraph_id: str | None = None,
) -> str:
    base = (settings.corpus_public_base_url or "").strip().rstrip("/")
    if not base:
        return ""
    url = f"{base}/books/{source_id}/{segment_slug}.html"
    pid = (paragraph_id or "").strip()
    if pid:
        # Query only. A fragment on a 302 Location is dropped by many browsers,
        # and reader.js then skips the jump when a hash is already present.
        url = f"{url}?p={pid}"
    return url


def passage_link(paragraph_id: str) -> str:
    pid = (paragraph_id or "").strip()
    if not pid:
        return ""
    base = (settings.mcp_base_url or "").strip().rstrip("/")
    if not base:
        return f"/passage/{pid}"
    return f"{base}/passage/{pid}"
