"""Extra public hostnames for the same /mcp endpoint.

The canonical origin stays ``RAGRUN_MCP_BASE_URL`` (passage links, the app).
An alias such as ``https://philo.ragxxx.com`` is accepted as its own OAuth
resource ``https://philo.ragxxx.com/mcp``. Clients compare that string exactly.
"""

from __future__ import annotations

from urllib.parse import urlparse

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.config import settings


def mcp_resource_url(mcp_base: str) -> str:
    """Canonical MCP resource URL (no trailing slash).

    Cursor compares this string exactly to the URL in the MCP config.
    Claude and Cursor both register ``https://host/mcp``; advertising
    ``…/mcp/`` makes Cursor fail with a protected-resource mismatch.
    """
    return f"{mcp_base.rstrip('/')}/mcp"


def resource_metadata_url(resource_url: str) -> str:
    """RFC 9728 metadata URL for a resource that ends in ``/mcp``."""
    parsed = urlparse(resource_url)
    path = "" if parsed.path == "/" else parsed.path
    return f"{parsed.scheme}://{parsed.netloc}/.well-known/oauth-protected-resource{path}"


def alias_bases_from(raw: str, *, canonical_base: str = "") -> list[str]:
    """HTTPS origins only, no path, and not the canonical host."""
    canonical_host = urlparse((canonical_base or "").strip()).netloc.lower()
    seen: list[str] = []
    for part in (raw or "").split(","):
        base = part.strip().rstrip("/")
        if not base:
            continue
        parsed = urlparse(base)
        if parsed.scheme != "https" or not parsed.hostname or parsed.path not in ("",):
            continue
        if parsed.query or parsed.fragment:
            continue
        if parsed.netloc.lower() == canonical_host:
            continue
        if base not in seen:
            seen.append(base)
    return seen


def _canonical_base() -> str:
    raw = getattr(settings, "mcp_base_url", "")
    if not isinstance(raw, str):
        return ""
    return raw.strip().rstrip("/")


def alias_bases() -> list[str]:
    raw = getattr(settings, "mcp_alias_base_urls", "")
    if not isinstance(raw, str):
        return []
    return alias_bases_from(raw, canonical_base=_canonical_base())


def mcp_allowed_hosts() -> list[str]:
    hosts: list[str] = []
    canonical = _canonical_base()
    if canonical:
        host = urlparse(canonical).netloc
        if host:
            hosts.append(host)
    for base in alias_bases():
        host = urlparse(base).netloc
        if host and host not in hosts:
            hosts.append(host)
    return hosts


def alias_base_for_host(host_header: str | None) -> str | None:
    if not isinstance(host_header, str):
        return None
    host = host_header.split(",")[0].strip().lower()
    if not host:
        return None
    for base in alias_bases():
        alias_host = urlparse(base).netloc.lower()
        if host == alias_host or host == f"{alias_host}:443":
            return base
    return None


class McpAliasResourceMiddleware:
    """Rewrite the OAuth resource metadata URL when the request Host is an alias.

    The MCP SDK stamps ``WWW-Authenticate`` with the canonical resource at
    startup. Alias clients must see their own metadata URL; other hosts are
    left untouched.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        canonical = _canonical_base()
        alias = alias_base_for_host(Headers(scope=scope).get("host"))
        if not canonical or not alias:
            await self.app(scope, receive, send)
            return

        canonical_meta = resource_metadata_url(mcp_resource_url(canonical))
        alias_meta = resource_metadata_url(mcp_resource_url(alias))

        async def send_alias(message: Message) -> None:
            if message["type"] == "http.response.start" and canonical_meta != alias_meta:
                raw_headers = message.get("headers")
                if raw_headers is not None:
                    headers = MutableHeaders(raw=raw_headers)
                    www = headers.get("www-authenticate")
                    if www and canonical_meta in www:
                        headers["www-authenticate"] = www.replace(canonical_meta, alias_meta)
            await send(message)

        await self.app(scope, receive, send_alias)
