from __future__ import annotations

import os


def whisper_model_name() -> str:
    return (os.environ.get("WHISPER_MODEL") or "medium").strip() or "medium"


def whisper_device() -> str:
    raw = (os.environ.get("WHISPER_DEVICE") or "").strip()
    if raw:
        return raw
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def whisper_download_root() -> str | None:
    raw = (os.environ.get("WHISPER_DOWNLOAD_ROOT") or "").strip()
    return raw or None


def hf_token() -> str | None:
    raw = (
        os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""
    ).strip()
    return raw or None
