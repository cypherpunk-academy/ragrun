"""Tests for /app JWT auth."""
from __future__ import annotations

import time
from unittest.mock import patch

import jwt
import pytest
from fastapi import HTTPException

from app.api.auth import parse_bearer_token


@pytest.fixture
def jwt_secret() -> str:
    return "test-secret-for-ragrun-auth"


def _make_token(
    secret: str,
    *,
    sub: str = "user-1",
    expired: bool = False,
    client_id: str | None = None,
) -> str:
    now = int(time.time())
    payload = {
        "sub": sub,
        "aud": "authenticated",
        "exp": now - 10 if expired else now + 3600,
        "email": "user@example.com",
    }
    if client_id is not None:
        payload["client_id"] = client_id
    return jwt.encode(payload, secret, algorithm="HS256")


def test_parse_bearer_token_valid(jwt_secret: str) -> None:
    token = _make_token(jwt_secret)
    with patch("app.api.auth.settings") as mock_settings:
        mock_settings.supabase_url = ""
        mock_settings.supabase_jwt_secret = jwt_secret
        user = parse_bearer_token(token)
    assert user.user_id == "user-1"
    assert user.email == "user@example.com"
    assert user.raw_token == token
    assert user.client_id is None


def test_parse_bearer_token_with_oauth_client_id(jwt_secret: str) -> None:
    token = _make_token(jwt_secret, client_id="e87351ba-43e3-4ecb-a846-fb6b1291edfb")
    with patch("app.api.auth.settings") as mock_settings:
        mock_settings.supabase_url = ""
        mock_settings.supabase_jwt_secret = jwt_secret
        user = parse_bearer_token(token)
    assert user.client_id == "e87351ba-43e3-4ecb-a846-fb6b1291edfb"


def test_parse_bearer_token_expired(jwt_secret: str) -> None:
    token = _make_token(jwt_secret, expired=True)
    with patch("app.api.auth.settings") as mock_settings:
        mock_settings.supabase_url = ""
        mock_settings.supabase_jwt_secret = jwt_secret
        with pytest.raises(HTTPException) as exc:
            parse_bearer_token(token)
    assert exc.value.status_code == 401
