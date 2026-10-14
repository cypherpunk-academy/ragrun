"""Smoke tests for the OAuth consent page hosted by ragrun."""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def test_oauth_consent_serves_html(monkeypatch) -> None:
    from app import config

    monkeypatch.setattr(config.settings, "supabase_url", "https://example.supabase.co")
    monkeypatch.setattr(config.settings, "supabase_anon_key", "test-anon-key")

    client = TestClient(app)
    response = client.get("/oauth/consent")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    body = response.text
    assert "Philo" in body
    assert "https://example.supabase.co" in body
    assert "test-anon-key" in body
    assert "__RAGRUN_OAUTH_CONFIG__" not in body


def test_index_lists_oauth_consent() -> None:
    client = TestClient(app)
    payload = client.get("/").json()
    assert payload.get("oauth_consent") == "/oauth/consent"
