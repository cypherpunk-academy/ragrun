from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from ..transcribe import transcribe_file
from .base import ProgressCallback, ScribeRequest

logger = logging.getLogger(__name__)

_PCT_RE = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*%")


def _validate_http_url(url: str) -> str:
    raw = (url or "").strip()
    if not raw:
        raise ValueError("url is required")
    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("url must be http or https")
    return raw


def _parse_yt_dlp_pct(line: str) -> int | None:
    matches = list(_PCT_RE.finditer(line))
    if not matches:
        return None
    n = float(matches[-1].group(1))
    if not (0 <= n <= 100):
        return None
    return int(round(n))


def download_audio(
    url: str,
    out_dir: Path,
    *,
    cookies_from_browser: Optional[str] = None,
    progress_cb: ProgressCallback | None = None,
) -> Path:
    out_template = str(out_dir / "audio.%(ext)s")
    cmd = ["yt-dlp"]
    if cookies_from_browser:
        cmd.extend(["--cookies-from-browser", cookies_from_browser])
    cmd.extend(
        [
            "-x",
            "--audio-format",
            "mp3",
            "--no-playlist",
            "-o",
            out_template,
            "--newline",
            url,
        ]
    )

    if progress_cb:
        progress_cb({"type": "fetch", "pct": 0})

    logger.info("yt-dlp download: %s", url)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    output_lines: list[str] = []
    for line in proc.stdout:
        output_lines.append(line)
        pct = _parse_yt_dlp_pct(line)
        if pct is not None and progress_cb:
            progress_cb({"type": "fetch", "pct": pct})
    code = proc.wait()
    if code != 0:
        detail = "".join(output_lines[-40:]).strip() or f"yt-dlp exit {code}"
        raise RuntimeError(f"yt-dlp failed: {detail}")

    files = sorted(out_dir.iterdir())
    audio = next(
        (p for p in files if p.suffix.lower() in {".mp3", ".m4a", ".wav", ".webm", ".opus"}),
        None,
    )
    if audio is None:
        raise RuntimeError(f"No audio file after yt-dlp in {out_dir}")

    if progress_cb:
        progress_cb({"type": "fetch", "pct": 100})
    return audio


class VideoUrlInput:
    name = "video_url"

    def run(
        self,
        request: ScribeRequest,
        *,
        progress_cb: ProgressCallback | None = None,
    ) -> str:
        url = _validate_http_url(request.url)
        tmp = Path(tempfile.mkdtemp(prefix="ragscribe-"))
        try:
            audio = download_audio(
                url,
                tmp,
                cookies_from_browser=request.cookies_from_browser,
                progress_cb=progress_cb,
            )
            return transcribe_file(
                str(audio),
                language=request.language,
                progress_cb=progress_cb,
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
