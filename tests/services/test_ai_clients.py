"""Unit tests for OAuth redirect host → client_kind mapping."""
from __future__ import annotations

from app.services.ai_clients import (
    classify_oauth_client,
    clear_ai_clients_cache,
    hosts_from_redirect_uris,
    pick_redirect_host,
    resolve_client_kind,
)


def setup_function() -> None:
    clear_ai_clients_cache()


def test_hosts_from_redirect_uris_csv() -> None:
    hosts = hosts_from_redirect_uris(
        "https://claude.ai/api/mcp/auth_callback,http://localhost:8787/callback"
    )
    assert hosts == ["claude.ai", "localhost"]


def test_pick_redirect_host_prefers_public() -> None:
    assert pick_redirect_host(["localhost", "claude.ai"]) == "claude.ai"
    assert pick_redirect_host(["127.0.0.1"]) == "127.0.0.1"


def test_resolve_client_kind_known_hosts() -> None:
    assert resolve_client_kind("claude.ai") == "claude"
    assert resolve_client_kind("www.cursor.com") == "cursor"
    assert resolve_client_kind("chatgpt.com") == "chatgpt"
    assert resolve_client_kind("chat.openai.com") == "chatgpt"
    assert resolve_client_kind("example.com") == "unknown"
    assert resolve_client_kind(None) == "unknown"


def test_classify_oauth_client_claude() -> None:
    out = classify_oauth_client(
        client_name="Claude",
        redirect_uris="https://claude.ai/api/mcp/auth_callback",
    )
    assert out["client_kind"] == "claude"
    assert out["redirect_host"] == "claude.ai"
    assert out["client_name"] == "Claude"
