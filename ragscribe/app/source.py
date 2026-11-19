from __future__ import annotations

from urllib.parse import urlparse

SOURCES = ("youtube", "tiktok", "facebook", "other")


def _host_is(host: str, domain: str) -> bool:
    return host == domain or host.endswith("." + domain)


def detect_source(url: str) -> str:
    host = (urlparse(url).hostname or "").lower().rstrip(".")
    if _host_is(host, "youtube.com") or _host_is(host, "youtu.be"):
        return "youtube"
    if _host_is(host, "tiktok.com"):
        return "tiktok"
    if (
        _host_is(host, "facebook.com")
        or _host_is(host, "fb.com")
        or _host_is(host, "fb.watch")
    ):
        return "facebook"
    return "other"
