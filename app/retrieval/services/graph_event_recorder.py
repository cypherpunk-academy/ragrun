"""No-op stub — event_metadata/event_content tables have been removed."""
from __future__ import annotations

from typing import Any


class GraphEventRecorder:
    async def record_event(self, **_: Any) -> None:
        pass
