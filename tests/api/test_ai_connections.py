"""Tests for GET /app/ai-connections and /app/claude-status status helper."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api.app_api import _grant_status
from app.api.auth import AuthUser


def test_grant_status_variants() -> None:
    assert _grant_status({"revoked_at": datetime.now(timezone.utc), "last_mcp_request": None}) == "revoked"
    assert _grant_status({"revoked_at": None, "last_mcp_request": None}) == "connected_unused"
    assert (
        _grant_status(
            {
                "revoked_at": None,
                "last_mcp_request": datetime(2026, 1, 1, tzinfo=timezone.utc),
            }
        )
        == "connected"
    )


@pytest.fixture
def auth_user() -> AuthUser:
    return AuthUser(user_id="11111111-1111-1111-1111-111111111111", email="a@b.c", raw_token="t")


def test_ai_connections_lists_grants(auth_user: AuthUser) -> None:
    from app.main import app

    created = datetime(2026, 3, 1, tzinfo=timezone.utc)
    last = datetime(2026, 3, 2, tzinfo=timezone.utc)
    row = {
        "client_id": "cid-1",
        "client_name": "Claude",
        "client_kind": "claude",
        "redirect_host": "claude.ai",
        "last_mcp_request": last,
        "revoked_at": None,
        "created_at": created,
    }
    mock_result = MagicMock()
    mock_result.mappings.return_value = [row]
    mock_conn = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.__exit__.return_value = None
    mock_conn.execute.return_value = mock_result
    mock_engine = MagicMock()
    mock_engine.connect.return_value = mock_conn

    async def _override_user():
        return auth_user

    from app.api.auth import get_current_user

    app.dependency_overrides[get_current_user] = _override_user
    try:
        with patch("app.api.app_api.get_engine", return_value=mock_engine):
            client = TestClient(app)
            resp = client.get("/app/ai-connections")
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    body = resp.json()
    assert len(body["connections"]) == 1
    conn = body["connections"][0]
    assert conn["client_kind"] == "claude"
    assert conn["status"] == "connected"
    assert conn["redirect_host"] == "claude.ai"
    assert conn["last_mcp_request"].startswith("2026-03-02")
