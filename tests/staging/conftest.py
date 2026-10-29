"""Live smoke against ragrun staging. Stays out of a plain pytest run."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = ROOT / ".env.staging"

STAGING_ORIGIN = "https://staging.ragxxx.com"
QDRANT_STAGING_ORIGIN = "https://qdrant-staging.ragxxx.com"
EMBED_ORIGIN = "https://embed.ragxxx.com"
SUPABASE_PROJECT_HOST = "duhtxxbynkilpxdpcpsk.supabase.co"

_SECRET_KEYS = (
    "RAGRUN_SUPABASE_ANON_KEY",
    "RAGRUN_SUPABASE_SERVICE_ROLE_KEY",
    "SUPABASE_SERVICE_ROLE_KEY",
    "RAGRUN_QDRANT_API_KEY",
    "RAGRUN_CF_ACCESS_CLIENT_ID",
    "RAGRUN_CF_ACCESS_CLIENT_SECRET",
)


def _explicitly_invoked(config: pytest.Config) -> bool:
    if os.environ.get("STAGING_SMOKE") == "1":
        return True
    return any("staging" in str(arg) for arg in config.invocation_params.args)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    own = [item for item in items if _is_staging(item)]
    if not own or _explicitly_invoked(config):
        return
    skip = pytest.mark.skip(reason="staging smoke runs only via pytest tests/staging")
    for item in own:
        item.add_marker(skip)


def _is_staging(item: pytest.Item) -> bool:
    return "tests/staging" in str(getattr(item, "path", item.fspath)).replace("\\", "/")


def _parse_env_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise RuntimeError(f"missing {path.name}")
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        if not value or "<" in value:
            continue
        out[key.strip()] = value
    return out


@dataclass(repr=False)
class StagingEnv:
    """Public hosts plus secret lookups. ``repr`` hides values so pytest ``-l`` stays safe."""

    values: dict[str, str]

    def __repr__(self) -> str:
        return "StagingEnv(redacted)"

    def require(self, key: str) -> str:
        value = self.values.get(key, "").strip()
        if not value:
            raise RuntimeError(f"{key} missing in .env.staging")
        return value

    @property
    def supabase_url(self) -> str:
        return self.require("RAGRUN_SUPABASE_URL").rstrip("/")

    def cf_headers(self) -> dict[str, str]:
        return {
            "CF-Access-Client-Id": self.require("RAGRUN_CF_ACCESS_CLIENT_ID"),
            "CF-Access-Client-Secret": self.require("RAGRUN_CF_ACCESS_CLIENT_SECRET"),
        }

    def qdrant_headers(self) -> dict[str, str]:
        headers = self.cf_headers()
        headers["api-key"] = self.require("RAGRUN_QDRANT_API_KEY")
        return headers


@pytest.fixture(scope="session")
def staging_env() -> StagingEnv:
    """Always read .env.staging so a shell pointed at dev cannot redirect the suite."""
    env = StagingEnv(_parse_env_file(ENV_FILE))
    host = httpx.URL(env.supabase_url).host or ""
    if host != SUPABASE_PROJECT_HOST:
        raise RuntimeError("RAGRUN_SUPABASE_URL is not the staging Supabase project")
    for key in _SECRET_KEYS:
        env.require(key)
    # mint.py reads the process environment. Force the staging project.
    os.environ["RAGRUN_SUPABASE_URL"] = env.supabase_url
    os.environ["RAGRUN_SUPABASE_ANON_KEY"] = env.require("RAGRUN_SUPABASE_ANON_KEY")
    os.environ["RAGRUN_SUPABASE_SERVICE_ROLE_KEY"] = env.require("RAGRUN_SUPABASE_SERVICE_ROLE_KEY")
    os.environ["SUPABASE_SERVICE_ROLE_KEY"] = env.require("SUPABASE_SERVICE_ROLE_KEY")
    return env


@pytest.fixture(scope="session")
def http() -> httpx.Client:
    with httpx.Client(timeout=httpx.Timeout(60.0), follow_redirects=False) as client:
        yield client
