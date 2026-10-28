"""Second public hostname for /mcp, without moving the canonical origin."""

from __future__ import annotations

import asyncio

import pytest
from starlette.requests import Request

from app.config import settings
from app.mcp_server.aliases import (
    McpAliasResourceMiddleware,
    alias_bases_from,
    mcp_allowed_hosts,
)
from app.mcp_server.server import create_resource_metadata_route

_CANONICAL_META = (
    'Bearer error="invalid_token", '
    'resource_metadata="https://api.ragxxx.com/.well-known/oauth-protected-resource/mcp"'
)


@pytest.fixture
def alias_settings(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "mcp_base_url", "https://api.ragxxx.com")
    monkeypatch.setattr(settings, "mcp_alias_base_urls", "https://philo.ragxxx.com")
    monkeypatch.setattr(settings, "supabase_url", "https://example.supabase.co")


def test_alias_bases_ignore_canonical_and_non_origins() -> None:
    raw = (
        "https://philo.ragxxx.com, https://api.ragxxx.com, "
        "http://philo.ragxxx.com, https://philo.ragxxx.com/mcp"
    )
    assert alias_bases_from(raw, canonical_base="https://api.ragxxx.com") == [
        "https://philo.ragxxx.com"
    ]


def test_allowed_hosts_keep_canonical_first(alias_settings) -> None:
    assert mcp_allowed_hosts() == ["api.ragxxx.com", "philo.ragxxx.com"]


def test_metadata_resource_follows_host(alias_settings) -> None:
    route = create_resource_metadata_route()
    assert route is not None

    async def body_for(host: str) -> dict:
        scope = {
            "type": "http",
            "http_version": "1.1",
            "method": "GET",
            "scheme": "https",
            "path": route.path,
            "raw_path": route.path.encode(),
            "query_string": b"",
            "headers": [(b"host", host.encode())],
            "client": ("127.0.0.1", 123),
            "server": (host, 443),
        }
        response = await route.endpoint(Request(scope))
        return response.body

    canonical = asyncio.run(body_for("api.ragxxx.com"))
    alias = asyncio.run(body_for("philo.ragxxx.com"))
    assert b'"resource":"https://api.ragxxx.com/mcp"' in canonical
    assert b'"resource":"https://philo.ragxxx.com/mcp"' in alias


def _scope(host: str) -> dict:
    return {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "scheme": "https",
        "path": "/mcp",
        "raw_path": b"/mcp",
        "query_string": b"",
        "headers": [(b"host", host.encode())],
        "client": ("127.0.0.1", 123),
        "server": (host, 443),
    }


async def _inner(scope, receive, send) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": 401,
            "headers": [(b"www-authenticate", _CANONICAL_META.encode())],
        }
    )
    await send({"type": "http.response.body", "body": b"{}"})


async def _receive() -> dict:
    return {"type": "http.request", "body": b"", "more_body": False}


def _www_authenticate(messages: list[dict]) -> str:
    for key, value in messages[0]["headers"]:
        if key == b"www-authenticate":
            return value.decode()
    raise AssertionError("missing www-authenticate")


def test_www_authenticate_rewritten_only_for_alias_host(alias_settings) -> None:
    async def run(host: str) -> str:
        messages: list[dict] = []

        async def send(message: dict) -> None:
            messages.append(message)

        await McpAliasResourceMiddleware(_inner)(_scope(host), _receive, send)
        return _www_authenticate(messages)

    alias = asyncio.run(run("philo.ragxxx.com"))
    canonical = asyncio.run(run("api.ragxxx.com"))
    assert (
        alias
        == 'Bearer error="invalid_token", resource_metadata="https://philo.ragxxx.com/.well-known/oauth-protected-resource/mcp"'
    )
    assert canonical == _CANONICAL_META
