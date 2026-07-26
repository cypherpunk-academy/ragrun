"""No-op stubs — event_metadata/event_content tables have been removed."""
from __future__ import annotations

from typing import Any


class EventRecorder:
    async def record_metadata_only(self, **_: Any) -> None:
        pass

    async def record_event(self, **_: Any) -> None:
        pass


def enqueue_record_metadata_only(recorder: EventRecorder, **_: Any) -> None:
    pass


def enqueue_record_event(recorder: EventRecorder, **_: Any) -> None:
    pass
