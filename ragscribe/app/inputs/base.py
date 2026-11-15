from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol


ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass
class ScribeRequest:
    url: str
    language: Optional[str] = None
    cookies_from_browser: Optional[str] = None


class InputHandler(Protocol):
    name: str

    def run(
        self,
        request: ScribeRequest,
        *,
        progress_cb: ProgressCallback | None = None,
    ) -> str: ...
