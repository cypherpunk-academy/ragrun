"""Temporary debug instrumentation (session 50ceb2). Remove after verification."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

_SESSION = "50ceb2"
_INGEST_URLS = (
    "http://127.0.0.1:7480/ingest/f96b38f1-0577-4277-afab-70a8601f20d7",
    "http://host.docker.internal:7480/ingest/f96b38f1-0577-4277-afab-70a8601f20d7",
)
_LOG_PATH = "/Users/michael/Reniets/Ai/ragrun/ragkeep/.cursor/debug-50ceb2.log"


def agent_log(
    *,
    location: str,
    message: str,
    data: dict | None = None,
    hypothesis_id: str = "",
    run_id: str = "pre-fix",
) -> None:
    payload = {
        "sessionId": _SESSION,
        "location": location,
        "message": message,
        "data": data or {},
        "hypothesisId": hypothesis_id,
        "runId": run_id,
        "timestamp": int(time.time() * 1000),
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "X-Debug-Session-Id": _SESSION,
    }
    for url in _INGEST_URLS:
        try:
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            urllib.request.urlopen(req, timeout=1.5)
            return
        except (urllib.error.URLError, TimeoutError, OSError):
            continue
    try:
        with open(_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except OSError:
        pass
