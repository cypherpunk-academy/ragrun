from __future__ import annotations

from urllib.parse import urlparse


def detect_source(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    if host.endswith("youtube.com") or host == "youtu.be" or host.endswith(".youtube.com"):
        return "youtube"
    if (
        host.endswith("tiktok.com")
        or host == "vm.tiktok.com"
        or host.endswith(".tiktok.com")
    ):
        return "tiktok"
    return "other"
