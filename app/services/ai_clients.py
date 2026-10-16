"""Map OAuth redirect hosts to AI client kinds (config/ai_clients.yaml)."""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "ai_clients.yaml"


@lru_cache(maxsize=1)
def _load_kinds() -> dict[str, list[str]]:
    """Return kind → list of host patterns."""
    try:
        raw = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8")) or {}
    except Exception:
        logger.warning("config/ai_clients.yaml not readable; all clients stay unknown")
        return {}
    kinds = raw.get("kinds") if isinstance(raw, dict) else None
    if not isinstance(kinds, dict):
        return {}
    out: dict[str, list[str]] = {}
    for kind, body in kinds.items():
        if not isinstance(kind, str) or not isinstance(body, dict):
            continue
        hosts = body.get("hosts") or []
        if not isinstance(hosts, list):
            continue
        cleaned = [str(h).strip().lower() for h in hosts if str(h).strip()]
        if cleaned:
            out[kind.strip().lower()] = cleaned
    return out


def clear_ai_clients_cache() -> None:
    _load_kinds.cache_clear()


def hosts_from_redirect_uris(raw: str | None) -> list[str]:
    """Parse comma/semicolon-separated redirect_uris into hostnames."""
    if not raw or not str(raw).strip():
        return []
    hosts: list[str] = []
    for part in str(raw).replace(";", ",").split(","):
        uri = part.strip()
        if not uri:
            continue
        try:
            host = urlparse(uri).hostname
        except Exception:
            host = None
        if host:
            hosts.append(host.lower())
    return hosts


def pick_redirect_host(hosts: list[str]) -> str | None:
    """Prefer a public host over localhost for client_kind mapping."""
    if not hosts:
        return None
    for host in hosts:
        if host in {"localhost", "127.0.0.1", "::1"}:
            continue
        if host.endswith(".local"):
            continue
        return host
    return hosts[0]


def resolve_client_kind(host: str | None) -> str:
    """Map a redirect hostname to a configured kind, else ``unknown``."""
    if not host:
        return "unknown"
    needle = host.strip().lower()
    for kind, patterns in _load_kinds().items():
        for pattern in patterns:
            if needle == pattern or needle.endswith("." + pattern):
                return kind
    return "unknown"


def classify_oauth_client(
    *,
    client_name: str | None,
    redirect_uris: str | None,
) -> dict[str, Any]:
    """Derive redirect_host and client_kind from DCR / oauth_clients row fields."""
    hosts = hosts_from_redirect_uris(redirect_uris)
    redirect_host = pick_redirect_host(hosts)
    return {
        "client_name": (client_name or "").strip() or None,
        "redirect_host": redirect_host,
        "client_kind": resolve_client_kind(redirect_host),
        "hosts": hosts,
    }
