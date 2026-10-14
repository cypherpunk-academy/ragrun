"""Cloudflare Access headers for tunnel origins."""
from __future__ import annotations

from unittest.mock import patch

from app.infra.service_http_headers import merge_service_http_headers


def test_merge_service_http_headers_empty() -> None:
    with patch("app.infra.service_http_headers.settings") as mock_settings:
        mock_settings.cf_access_client_id = ""
        mock_settings.cf_access_client_secret = ""
        assert merge_service_http_headers({"api-key": "x"}) == {"api-key": "x"}


def test_merge_service_http_headers_cf_access() -> None:
    with patch("app.infra.service_http_headers.settings") as mock_settings:
        mock_settings.cf_access_client_id = "cid"
        mock_settings.cf_access_client_secret = "sec"
        out = merge_service_http_headers(None)
        assert out == {
            "CF-Access-Client-Id": "cid",
            "CF-Access-Client-Secret": "sec",
        }
