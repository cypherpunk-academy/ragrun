"""MCP mount path: Claude uses /mcp without trailing slash."""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def test_post_mcp_without_trailing_slash_not_404() -> None:
    client = TestClient(app)
    response = client.post("/mcp", json={})
    assert response.status_code != 404
