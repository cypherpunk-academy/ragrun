"""Smoke the live staging path: Cloudflare DNS, Railway, Supabase, Tuxai.

Client → staging.ragxxx.com → Railway ragrun-staging
      → Supabase (Postgres + JWT), direct
      → qdrant-staging.ragxxx.com and embed.ragxxx.com
        through Cloudflare Access and the tunnel to Tuxai

Run: pytest tests/staging
Secrets stay in .env.staging. This module never prints them.
"""
from __future__ import annotations

import socket
import subprocess

import httpx

from scripts.mcp_tests.mint import mint_user
from tests.staging.conftest import EMBED_ORIGIN, QDRANT_STAGING_ORIGIN, STAGING_ORIGIN, StagingEnv

# Title search scores are 0.5, 0.8, or 1.0. Hybrid RRF scores sit well below that.
_VECTOR_SCORE_CEILING = 0.4


def _records(name: str, rtype: str) -> list[str]:
    proc = subprocess.run(
        ["dig", "+short", name, rtype],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    if proc.returncode != 0:
        raise AssertionError(f"dig {name} {rtype} failed")
    return [line.strip().rstrip(".") for line in proc.stdout.splitlines() if line.strip()]


def _public_ipv4(records: list[str]) -> list[str]:
    out: list[str] = []
    for item in records:
        try:
            socket.inet_aton(item)
        except OSError:
            continue
        if item.startswith(("10.", "127.", "192.168.", "169.254.", "0.")):
            continue
        if item.startswith("172."):
            second = int(item.split(".")[1])
            if 16 <= second <= 31:
                continue
        out.append(item)
    return out


def _get(client: httpx.Client, url: str, headers: dict[str, str] | None = None) -> httpx.Response:
    try:
        return client.get(url, headers=headers)
    except httpx.HTTPError as exc:
        raise AssertionError(f"GET {url} failed: {type(exc).__name__}") from None


def _assert_reached_origin(response: httpx.Response, url: str) -> None:
    """401/403 HTML from Cloudflare Access means the service token never reached Tuxai."""
    if "cloudflare access" in response.text.lower()[:500]:
        raise AssertionError(
            f"{url} rejected the service token in .env.staging (HTTP {response.status_code}). "
            "Railway /healthz uses the token stored on ragrun-staging, which can differ."
        )


def test_dns_staging_has_public_a() -> None:
    addresses = _public_ipv4(_records("staging.ragxxx.com", "A"))
    assert addresses, "staging.ragxxx.com has no public A record"


def test_dns_tunnel_hostnames_are_not_fd10_only() -> None:
    for name in ("qdrant-staging.ragxxx.com", "embed.ragxxx.com"):
        v4 = _public_ipv4(_records(name, "A"))
        v6 = _records(name, "AAAA")
        assert v4, f"{name} has no public A (fd10-only tunnel DNS is not reachable from Railway)"
        assert not any(item.lower().startswith("fd10:") for item in v4)
        if v6 and not v4:
            raise AssertionError(f"{name} is AAAA-only")


def test_staging_index_is_railway(http: httpx.Client) -> None:
    response = _get(http, f"{STAGING_ORIGIN}/")
    assert response.status_code == 200, response.status_code
    body = response.json()
    assert body["service"] == "ragrun-api"
    assert body["environment"] == "staging"


def test_healthz_reaches_tuxai_through_the_tunnel(http: httpx.Client) -> None:
    response = _get(http, f"{STAGING_ORIGIN}/healthz")
    assert response.status_code == 200, response.status_code
    deps = response.json()["dependencies"]
    qdrant = deps["qdrant"]
    embed = deps["embedding_service"]
    assert qdrant["status"] == "healthy", qdrant
    assert embed["status"] == "healthy", embed
    assert qdrant["endpoint"] == f"{QDRANT_STAGING_ORIGIN}/healthz"
    assert embed["endpoint"] == f"{EMBED_ORIGIN}/api/v1/health/simple"


def test_app_health_reaches_supabase_postgres(http: httpx.Client) -> None:
    response = _get(http, f"{STAGING_ORIGIN}/app/health")
    assert response.status_code == 200, response.status_code
    body = response.json()
    assert body["online"] is True
    assert body["status"] == "ok"


def test_supabase_rest_is_awake(http: httpx.Client, staging_env: StagingEnv) -> None:
    response = _get(
        http,
        f"{staging_env.supabase_url}/rest/v1/",
        {
            "apikey": staging_env.require("RAGRUN_SUPABASE_ANON_KEY"),
            "Authorization": f"Bearer {staging_env.require('RAGRUN_SUPABASE_ANON_KEY')}",
        },
    )
    assert response.status_code < 500, response.status_code


def test_supabase_jwks_is_published(http: httpx.Client, staging_env: StagingEnv) -> None:
    response = _get(http, f"{staging_env.supabase_url}/auth/v1/.well-known/jwks.json")
    assert response.status_code == 200, response.status_code
    keys = response.json().get("keys")
    assert isinstance(keys, list) and keys


def test_sources_reject_missing_bearer(http: httpx.Client) -> None:
    response = _get(http, f"{STAGING_ORIGIN}/app/sources")
    assert response.status_code == 401, response.status_code


def test_tunnel_qdrant_staging_accepts_service_auth(http: httpx.Client, staging_env: StagingEnv) -> None:
    url = f"{QDRANT_STAGING_ORIGIN}/healthz"
    response = _get(http, url, staging_env.qdrant_headers())
    _assert_reached_origin(response, url)
    assert response.status_code == 200, response.status_code


def test_tunnel_embed_accepts_service_auth(http: httpx.Client, staging_env: StagingEnv) -> None:
    url = f"{EMBED_ORIGIN}/api/v1/health/simple"
    response = _get(http, url, staging_env.cf_headers())
    _assert_reached_origin(response, url)
    assert response.status_code == 200, response.status_code


def test_tunnel_rejects_missing_access_token(http: httpx.Client) -> None:
    response = _get(http, f"{EMBED_ORIGIN}/api/v1/health/simple")
    assert response.status_code in {401, 403, 302}, response.status_code


def test_qdrant_rejects_missing_api_key(http: httpx.Client, staging_env: StagingEnv) -> None:
    url = f"{QDRANT_STAGING_ORIGIN}/healthz"
    response = _get(http, url, staging_env.cf_headers())
    _assert_reached_origin(response, url)
    assert response.status_code == 401, response.status_code


def test_search_returns_a_vector_hit(http: httpx.Client) -> None:
    """One search: Railway embeds on Tuxai, then queries staging Qdrant."""
    try:
        response = http.post(
            f"{STAGING_ORIGIN}/app/search",
            json={"query": "Freiheit und Erkenntnis", "types": ["text"], "limit": 5},
        )
    except httpx.HTTPError as exc:
        raise AssertionError(f"POST /app/search failed: {type(exc).__name__}") from None
    assert response.status_code == 200, response.status_code
    results = response.json().get("results") or []
    assert results, "search returned no hits"
    vector_hits = [hit for hit in results if float(hit.get("score") or 1) < _VECTOR_SCORE_CEILING]
    assert vector_hits, "hits are title matches only; embed/qdrant did not contribute"
    assert vector_hits[0].get("chunk_id")


def _sources_as_anton(client: httpx.Client) -> httpx.Response:
    try:
        token = mint_user("a")
    except Exception as exc:
        raise AssertionError(f"could not mint staging user anton: {type(exc).__name__}") from None
    return _get(client, f"{STAGING_ORIGIN}/app/sources", {"Authorization": f"Bearer {token}"})


def test_sources_accept_staging_user_jwt(http: httpx.Client, staging_env: StagingEnv) -> None:
    _ = staging_env
    response = _sources_as_anton(http)
    assert response.status_code == 200, response.status_code
    sources = response.json().get("sources") or []
    assert sources, "staging catalog is empty"
    assert sources[0].get("source_id")
