from __future__ import annotations

from typing import Any, Callable

from .base import InputHandler, ScribeRequest
from .video_url import VideoUrlInput

ProgressCallback = Callable[[dict[str, Any]], None]

_REGISTRY: dict[str, InputHandler] = {
    VideoUrlInput.name: VideoUrlInput(),
}


def list_inputs() -> list[str]:
    return sorted(_REGISTRY.keys())


def get_input(name: str) -> InputHandler | None:
    return _REGISTRY.get(name)


def run_input(
    name: str,
    request: ScribeRequest,
    *,
    progress_cb: ProgressCallback | None = None,
) -> str:
    handler = get_input(name)
    if handler is None:
        raise KeyError(name)
    return handler.run(request, progress_cb=progress_cb)
