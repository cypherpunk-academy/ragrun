from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional

import tqdm as tqdm_mod

from .config import whisper_device, whisper_download_root, whisper_model_name

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass
class WhisperRuntime:
    model: Any
    model_name: str
    device: str
    ready: bool = False
    lock: threading.Lock = threading.Lock()


runtime = WhisperRuntime(model=None, model_name="", device="")


def load_model() -> None:
    import whisper

    name = whisper_model_name()
    device = whisper_device()
    download_root = whisper_download_root()
    logger.info("Loading Whisper model %s on %s …", name, device)
    kwargs: dict[str, Any] = {"device": device}
    if download_root:
        kwargs["download_root"] = download_root
    model = whisper.load_model(name, **kwargs)
    runtime.model = model
    runtime.model_name = name
    runtime.device = device
    runtime.ready = True
    logger.info("Whisper model %s ready", name)


class _ProgressTqdm(tqdm_mod.tqdm):
    """Whisper's transcribe() updates tqdm with mel frames; report seconds."""

    _frames_per_second = 100  # whisper.audio.FRAMES_PER_SECOND

    def __init__(self, *args: Any, progress_cb: ProgressCallback | None = None, **kwargs: Any):
        self._progress_cb = progress_cb
        super().__init__(*args, **kwargs)
        self._emit()

    def update(self, n: float | None = 1) -> bool | None:
        result = super().update(n)
        self._emit()
        return result

    def _emit(self) -> None:
        if not self._progress_cb:
            return
        total = float(self.total or 0)
        current = float(self.n or 0)
        duration_sec = total / self._frames_per_second if total > 0 else 0.0
        processed_sec = current / self._frames_per_second if total > 0 else 0.0
        pct = int(round(100 * current / total)) if total > 0 else 0
        self._progress_cb(
            {
                "type": "progress",
                "processed_sec": round(processed_sec, 1),
                "duration_sec": round(duration_sec, 1),
                "pct": max(0, min(100, pct)),
            }
        )


def audio_duration_sec(path: str) -> float:
    import whisper

    audio = whisper.load_audio(path)
    return float(len(audio) / whisper.audio.SAMPLE_RATE)


def transcribe_file(
    path: str,
    *,
    language: Optional[str] = None,
    progress_cb: ProgressCallback | None = None,
) -> str:
    if not runtime.ready or runtime.model is None:
        raise RuntimeError("Whisper model is not loaded")

    duration = audio_duration_sec(path)
    if progress_cb:
        progress_cb(
            {
                "type": "start",
                "duration_sec": round(duration, 1),
                "model": runtime.model_name,
                "device": runtime.device,
            }
        )

    # Whisper wires tqdm when verbose is False.
    original_tqdm = tqdm_mod.tqdm

    def factory(*args: Any, **kwargs: Any) -> _ProgressTqdm:
        kwargs.setdefault("disable", False)
        return _ProgressTqdm(*args, progress_cb=progress_cb, **kwargs)

    options: dict[str, Any] = {"verbose": False, "task": "transcribe"}
    if language:
        options["language"] = language.split("-")[0].lower()

    with runtime.lock:
        tqdm_mod.tqdm = factory  # type: ignore[assignment]
        try:
            result = runtime.model.transcribe(path, **options)
        finally:
            tqdm_mod.tqdm = original_tqdm  # type: ignore[assignment]

    text = str(result.get("text") or "").strip()
    if progress_cb:
        progress_cb(
            {
                "type": "done",
                "text": text,
                "duration_sec": round(duration, 1),
                "language": result.get("language"),
            }
        )
    return text
