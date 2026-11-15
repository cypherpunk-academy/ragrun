from __future__ import annotations

import logging
import queue
import threading
import uuid
from typing import Any, Optional

from .inputs import run_input
from .inputs.base import ScribeRequest
from .qdrant_store import ScribeStore, get_store
from .source import detect_source
from .transcribe import runtime

logger = logging.getLogger(__name__)


class JobManager:
    def __init__(self, store: ScribeStore) -> None:
        self._store = store
        self._queue: queue.Queue[str] = queue.Queue()
        self._meta: dict[str, dict[str, Any]] = {}
        self._stop = threading.Event()
        self._worker = threading.Thread(
            target=self._run_worker, name="ragscribe-jobs", daemon=True
        )
        self._started = False
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._store.mark_interrupted()
            self._worker.start()
            self._started = True
            logger.info("Job worker started")

    def stop(self) -> None:
        self._stop.set()

    def enqueue(
        self,
        url: str,
        *,
        language: Optional[str] = None,
        cookies_from_browser: Optional[str] = None,
        diarize: bool = True,
    ) -> dict[str, Any]:
        from datetime import datetime, timezone

        job_id = str(uuid.uuid4())
        created = (
            datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        )
        payload = {
            "url": url.strip(),
            "source": detect_source(url),
            "status": "queued",
            "created_at": created,
            "duration_sec": None,
            "text": "",
            "error": None,
            "pct": 0,
        }
        with self._lock:
            self._meta[job_id] = {
                "language": language,
                "cookies_from_browser": cookies_from_browser,
                "diarize": diarize,
            }
        public = self._store.upsert(job_id, payload)
        self._queue.put(job_id)
        return public

    def _run_worker(self) -> None:
        while not self._stop.is_set():
            try:
                job_id = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._process(job_id)
            except Exception:
                logger.exception("Job worker crash on %s", job_id)
            finally:
                with self._lock:
                    self._meta.pop(job_id, None)
                self._queue.task_done()

    def _process(self, job_id: str) -> None:
        record = self._store.get(job_id)
        if record is None:
            return
        if record.get("status") == "error" and record.get("error") == "interrupted":
            return

        if not runtime.ready:
            self._store.patch(
                job_id,
                status="error",
                error="Whisper model not ready",
                pct=0,
            )
            return

        with self._lock:
            meta = dict(self._meta.get(job_id) or {})
        url = str(record.get("url") or "")
        language = meta.get("language")
        cookies = meta.get("cookies_from_browser")
        diarize = meta.get("diarize")
        if not isinstance(diarize, bool):
            diarize = True

        self._store.patch(job_id, status="processing", pct=0, error=None)

        def on_progress(event: dict[str, Any]) -> None:
            fields: dict[str, Any] = {}
            et = event.get("type")
            if et == "fetch":
                pct = event.get("pct")
                if pct is not None:
                    fields["pct"] = max(0, min(20, int(round(float(pct) * 0.2))))
            elif et == "start":
                if event.get("duration_sec") is not None:
                    fields["duration_sec"] = float(event["duration_sec"])
                fields["pct"] = 20
            elif et == "progress":
                pct = event.get("pct")
                if pct is not None:
                    fields["pct"] = 20 + max(
                        0, min(79, int(round(float(pct) * 0.79)))
                    )
                if event.get("duration_sec") is not None:
                    fields["duration_sec"] = float(event["duration_sec"])
            if fields:
                try:
                    self._store.patch(job_id, **fields)
                except Exception:
                    logger.exception("Failed to patch progress for %s", job_id)

        try:
            text = run_input(
                "video_url",
                ScribeRequest(
                    url=url,
                    language=language if isinstance(language, str) else None,
                    cookies_from_browser=(
                        cookies if isinstance(cookies, str) else None
                    ),
                    diarize=diarize,
                ),
                progress_cb=on_progress,
            )
            self._store.patch(
                job_id,
                status="ok",
                text=text or "",
                pct=100,
                error=None,
            )
        except Exception as exc:
            logger.exception("Job %s failed", job_id)
            self._store.patch(
                job_id,
                status="error",
                error=str(exc),
                pct=0,
            )


_manager: JobManager | None = None


def get_job_manager() -> JobManager:
    global _manager
    if _manager is None:
        _manager = JobManager(get_store())
    return _manager
