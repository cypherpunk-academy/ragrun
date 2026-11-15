from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from contextlib import asynccontextmanager
from queue import Empty, Queue
from typing import Any, AsyncGenerator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .inputs import get_input, list_inputs, run_input
from .inputs.base import ScribeRequest
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
    yield
    logger.info("Shutting down ragscribe")


app = FastAPI(
    title="ragscribe",
    description="Turn spoken video into text (model loaded once at startup)",
    version="1.0.0",
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
    stream: bool = True


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
    }


@app.get("/api/v1/inputs")
async def inputs():
    return {"inputs": list_inputs()}


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
