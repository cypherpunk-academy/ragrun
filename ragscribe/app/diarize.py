"""WhisperX diarization. Models load on the first request and stay resident."""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

from .config import hf_token, whisper_device, whisper_model_name
from .transcribe import ProgressCallback, runtime
from .turns import segments_to_turn_cued_text

logger = logging.getLogger(__name__)

_probe_lock = threading.Lock()
_diarize_available: bool | None = None


class _Pipeline:
    asr: Any = None
    asr_name: str = ""
    asr_device: str = ""
    align_language: str = ""
    align_model: Any = None
    align_meta: Any = None
    diarizer: Any = None
    diarizer_device: str = ""


_pipeline = _Pipeline()


def diarize_available() -> bool:
    """True when whisperx imports and a Hugging Face token is set. Does not load models."""
    global _diarize_available
    if _diarize_available is not None:
        return _diarize_available
    with _probe_lock:
        if _diarize_available is not None:
            return _diarize_available
        if not hf_token():
            _diarize_available = False
            return False
        try:
            from whisperx.diarize import DiarizationPipeline  # noqa: F401
        except Exception:
            try:
                from whisperx import DiarizationPipeline  # type: ignore  # noqa: F401
            except Exception:
                logger.info("whisperx diarization pipeline not importable; diarize disabled")
                _diarize_available = False
                return False
        _diarize_available = True
        return True


def _compute_type(device: str) -> str:
    return "float16" if device == "cuda" else "int8"


def _batch_size(device: str) -> int:
    return 16 if device == "cuda" else 8


def _load_asr(name: str, device: str) -> Any:
    import whisperx

    if (
        _pipeline.asr is not None
        and _pipeline.asr_name == name
        and _pipeline.asr_device == device
    ):
        return _pipeline.asr
    logger.info("Loading WhisperX model %s on %s …", name, device)
    model = whisperx.load_model(name, device, compute_type=_compute_type(device))
    _pipeline.asr = model
    _pipeline.asr_name = name
    _pipeline.asr_device = device
    return model


def _load_align(language: str, device: str) -> tuple[Any, Any]:
    import whisperx

    if (
        _pipeline.align_model is not None
        and _pipeline.align_language == language
        and _pipeline.asr_device == device
    ):
        return _pipeline.align_model, _pipeline.align_meta
    logger.info("Loading WhisperX align model for %s …", language)
    model, meta = whisperx.load_align_model(language_code=language, device=device)
    _pipeline.align_model = model
    _pipeline.align_meta = meta
    _pipeline.align_language = language
    return model, meta


def _load_diarizer(device: str) -> Any:
    token = hf_token()
    if not token:
        raise RuntimeError("HF_TOKEN is not set")
    if _pipeline.diarizer is not None and _pipeline.diarizer_device == device:
        return _pipeline.diarizer

    try:
        from whisperx.diarize import DiarizationPipeline
    except ImportError:
        from whisperx import DiarizationPipeline  # type: ignore

    logger.info("Loading pyannote diarization pipeline on %s …", device)
    try:
        pipeline = DiarizationPipeline(token=token, device=device)
    except TypeError:
        pipeline = DiarizationPipeline(use_auth_token=token, device=device)
    _pipeline.diarizer = pipeline
    _pipeline.diarizer_device = device
    return pipeline


def _speaker_of(segment: dict[str, Any]) -> str:
    speaker = str(segment.get("speaker") or "").strip()
    if speaker:
        return speaker
    counts: dict[str, int] = {}
    for word in segment.get("words") or []:
        if not isinstance(word, dict):
            continue
        label = str(word.get("speaker") or "").strip()
        if label:
            counts[label] = counts.get(label, 0) + 1
    if not counts:
        return ""
    return max(counts, key=lambda label: counts[label])


def _stage(
    progress_cb: ProgressCallback | None,
    stage: str,
    pct: int,
    duration: float,
) -> None:
    if not progress_cb:
        return
    progress_cb({"type": "stage", "stage": stage, "pct": pct})
    progress_cb(
        {
            "type": "progress",
            "pct": pct,
            "processed_sec": 0,
            "duration_sec": round(duration, 1),
        }
    )


def transcribe_diarized(
    path: str,
    *,
    language: Optional[str] = None,
    progress_cb: ProgressCallback | None = None,
) -> str:
    if not diarize_available():
        raise RuntimeError("diarize unavailable")

    import whisperx

    device = whisper_device()
    name = whisper_model_name()
    lang = language.split("-")[0].lower() if language else None

    with runtime.lock:
        audio = whisperx.load_audio(path)
        duration = float(len(audio)) / 16000.0
        if progress_cb:
            progress_cb(
                {
                    "type": "start",
                    "duration_sec": round(duration, 1),
                    "model": name,
                    "device": device,
                }
            )
        _stage(progress_cb, "transcribe", 5, duration)

        asr = _load_asr(name, device)
        result = asr.transcribe(
            audio,
            batch_size=_batch_size(device),
            language=lang,
        )
        detected = str(result.get("language") or lang or "")
        _stage(progress_cb, "align", 45, duration)
        if detected:
            try:
                align_model, align_meta = _load_align(detected, device)
                result = whisperx.align(
                    result["segments"],
                    align_model,
                    align_meta,
                    audio,
                    device,
                    return_char_alignments=False,
                )
            except Exception:
                logger.exception("WhisperX align failed; continuing with raw segments")

        _stage(progress_cb, "diarize", 70, duration)
        diarizer = _load_diarizer(device)
        diarize_segments = diarizer(audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)

        segments = [
            {"text": str(seg.get("text") or ""), "speaker": _speaker_of(seg)}
            for seg in (result.get("segments") or [])
            if isinstance(seg, dict)
        ]
        text = segments_to_turn_cued_text(segments).strip()
        if not text:
            raise RuntimeError("WhisperX returned no text")
        speakers = {seg["speaker"] for seg in segments if seg.get("speaker")}
        logger.info(
            "WhisperX diarize: %s segments, %s speakers",
            len(segments),
            len(speakers),
        )
        if progress_cb:
            progress_cb(
                {
                    "type": "done",
                    "text": text,
                    "duration_sec": round(duration, 1),
                    "language": detected or None,
                }
            )
        return text
