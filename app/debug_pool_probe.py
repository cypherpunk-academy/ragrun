"""Temporary agent-debug probes for Supabase connection pool investigation."""
from __future__ import annotations

import json
import time
from typing import Any

_LOG_PATH = "/Users/michael/Reniets/Ai/ragrun/ragkeep/.cursor/debug-05acdb.log"
_SESSION_ID = "05acdb"


def agent_log(
    location: str,
    message: str,
    data: dict[str, Any],
    *,
    hypothesis_id: str,
    run_id: str = "pre-fix",
) -> None:
    # region agent log
    try:
        payload = {
            "sessionId": _SESSION_ID,
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with open(_LOG_PATH, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str) + "\n")
    except Exception:
        pass
    # endregion


def pool_snapshot() -> dict[str, Any]:
    """SQLAlchemy pool counters for sync (NullPool) and async engines."""
    out: dict[str, Any] = {}
    try:
        from app.db.async_session import get_async_engine

        async_pool = get_async_engine().pool
        out["async"] = {
            "pool_class": type(async_pool).__name__,
            "size": async_pool.size(),
            "checked_out": async_pool.checkedout(),
            "overflow": async_pool.overflow(),
            "checked_in": async_pool.checkedin(),
        }
    except Exception as exc:
        out["async_error"] = repr(exc)
    try:
        from app.db.session import get_engine

        sync_pool = get_engine().pool
        out["sync"] = {"pool_class": type(sync_pool).__name__}
        if hasattr(sync_pool, "checkedout"):
            out["sync"]["checked_out"] = sync_pool.checkedout()
    except Exception as exc:
        out["sync_error"] = repr(exc)
    return out
