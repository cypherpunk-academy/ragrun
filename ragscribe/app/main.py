from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from contextlib import asynccontextmanager
from queue import Empty, Queue
from typing import Any, AsyncGenerator, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .diarize import diarize_available
from .inputs import get_input, list_inputs, run_input
from .inputs.base import ScribeRequest
from .jobs import get_job_manager
from .qdrant_store import get_store
from .transcribe import load_model, runtime

_log_level_str = (
    os.environ.get("WHISPER_LOG_LEVEL")
    or os.environ.get("LOG_LEVEL", "INFO")
)
_log_level = getattr(logging, _log_level_str.upper(), logging.INFO)
logging.basicConfig(
    level=_log_level,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("Starting ragscribe")
    try:
        load_model()
    except Exception:
        logger.exception("Failed to load Whisper model")
        raise
    try:
        get_store()
        get_job_manager().start()
    except Exception:
        logger.exception("Failed to start Qdrant job store")
        raise
    yield
    get_job_manager().stop()
    logger.info("Shutting down ragscribe")


app = FastAPI(
    title="ragscribe",
    description="Turn spoken video into text (model loaded once at startup)",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ScribeBody(BaseModel):
    input: str = Field(..., description="Registered input name, e.g. video_url")
    url: str
    language: Optional[str] = None
    cookies_from_browser: Optional[str] = None
    diarize: bool = True
    stream: bool = True


class JobCreateBody(BaseModel):
    url: str
    language: Optional[str] = None
    cookies_from_browser: Optional[str] = None
    diarize: bool = True


def _ndjson(event: dict[str, Any]) -> bytes:
    return (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")


@app.get("/health")
async def health():
    if not runtime.ready:
        return JSONResponse(
            status_code=503,
            content={"status": "loading", "ready": False},
        )
    return {
        "status": "ok",
        "ready": True,
        "model": runtime.model_name,
        "device": runtime.device,
        "diarize": diarize_available(),
        "inputs": list_inputs(),
    }


@app.get("/")
async def root():
    return {
        "service": "ragscribe",
        "ready": runtime.ready,
        "model": runtime.model_name or None,
        "docs": "/docs",
        "health": "/health",
        "inputs": "/api/v1/inputs",
        "scribe": "/api/v1/scribe",
        "jobs": "/api/v1/jobs",
    }


@app.get("/api/v1/inputs")
async def inputs():
    return {"inputs": list_inputs()}


@app.post("/api/v1/jobs")
async def create_job(body: JobCreateBody):
    if not runtime.ready:
        raise HTTPException(status_code=503, detail="Whisper model not ready")
    url = (body.url or "").strip()
    if not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="url must be http or https")
    try:
        return get_job_manager().enqueue(
            url,
            language=body.language,
            cookies_from_browser=body.cookies_from_browser,
            diarize=body.diarize,
        )
    except Exception as exc:
        logger.exception("create_job failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/v1/jobs")
async def list_jobs(
    source: Optional[str] = Query(None),
    q: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
):
    try:
        items = get_store().list_jobs(source=source, q=q, limit=limit)
    except Exception as exc:
        logger.exception("list_jobs failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"jobs": items}


@app.get("/api/v1/jobs/{job_id}")
async def get_job(job_id: str):
    item = get_store().get(job_id)
    if item is None:
        raise HTTPException(status_code=404, detail="job not found")
    return item


@app.delete("/api/v1/jobs/{job_id}")
async def delete_job(job_id: str):
    if not get_store().delete(job_id):
        raise HTTPException(status_code=404, detail="job not found")
    return {"ok": True, "id": job_id}


@app.post("/api/v1/scribe")
async def scribe(body: ScribeBody):
    if not runtime.ready:
        raise HTTPException(status_code=503, detail="Whisper model not ready")

    if get_input(body.input) is None:
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"unknown input: {body.input}",
                "inputs": list_inputs(),
            },
        )

    request = ScribeRequest(
        url=body.url,
        language=body.language,
        cookies_from_browser=body.cookies_from_browser,
        diarize=body.diarize,
    )

    if not body.stream:
        try:
            text = await asyncio.to_thread(
                run_input,
                body.input,
                request,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("scribe failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {
            "text": text,
            "model": runtime.model_name,
            "device": runtime.device,
        }

    return StreamingResponse(
        _stream_scribe(body.input, request),
        media_type="application/x-ndjson",
    )


async def _stream_scribe(
    input_name: str,
    request: ScribeRequest,
) -> AsyncGenerator[bytes, None]:
    events: Queue[dict[str, Any] | None] = Queue()

    def on_progress(event: dict[str, Any]) -> None:
        events.put(event)

    def worker() -> None:
        try:
            run_input(input_name, request, progress_cb=on_progress)
        except Exception as exc:
            events.put({"type": "error", "detail": str(exc)})
        finally:
            events.put(None)

    threading.Thread(target=worker, daemon=True).start()

    while True:
        try:
            item = await asyncio.to_thread(events.get, True, 0.5)
        except Empty:
            yield _ndjson({"type": "heartbeat"})
            continue
        if item is None:
            break
        yield _ndjson(item)
