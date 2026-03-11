"""Entry point for the ragrun FastAPI application."""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .api import rag as rag_router
from .retrieval.api import router as retrieval_router
from .core.providers import (
    get_deepseek_reasoner_client,
)
from .services.cleanup_service import run_cleanup_loop

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    cleanup_task = asyncio.create_task(run_cleanup_loop())
    try:
        yield
    finally:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            logger.info("event_content cleanup loop stopped")


app = FastAPI(
    title="ragrun API",
    version="0.3.0",
    summary="LangChain + Qdrant orchestration service",
    lifespan=lifespan,
)


async def _probe(client: httpx.AsyncClient, url: str) -> Dict[str, Any]:
    """Probe a downstream dependency and normalize the response."""

    try:
        response = await client.get(url)
        response.raise_for_status()
        payload: Any
        try:
            payload = response.json()
        except ValueError:
            payload = {"message": response.text}
        return {"status": "healthy", "endpoint": url, "details": payload}
    except Exception as exc:  # pragma: no cover - best effort health probe
        return {"status": "unhealthy", "endpoint": url, "error": str(exc)}


@app.get("/", tags=["meta"])
def index() -> Dict[str, Any]:
    """Basic service descriptor."""

    return {
        "service": "ragrun-api",
        "environment": settings.app_env,
        "docs": "/docs",
        "health": "/healthz",
    }


@app.get("/healthz", tags=["meta"])
async def healthz() -> Dict[str, Any]:
    """Aggregate health check for primary dependencies."""

    qdrant_health_url = f"{settings.qdrant_url.rstrip('/')}/healthz"
    embeddings_health_url = f"{settings.embeddings_base_url.rstrip('/')}/api/v1/health/simple"
    langfuse_health_url = f"{settings.langfuse_host.rstrip('/')}/api/public/health"

    async with httpx.AsyncClient(timeout=httpx.Timeout(3.0)) as client:
        probes = await asyncio.gather(
            _probe(client, qdrant_health_url),
            _probe(client, embeddings_health_url),
            _probe(client, langfuse_health_url),
        )

    deepseek_probe: Dict[str, Any] | None = None
    if settings.deepseek_model_probe and settings.deepseek_api_key:
        try:
            reasoner = get_deepseek_reasoner_client()
            models = await reasoner.list_models()
            deepseek_probe = {
                "status": "healthy",
                "models_count": len(models),
                "configured_reasoner": reasoner.model,
                "configured_chat": settings.deepseek_chat_model,
            }
            if reasoner.model and models and reasoner.model not in models:
                deepseek_probe["warning"] = "configured reasoner model not reported by /models"
        except Exception as exc:  # pragma: no cover - best effort
            deepseek_probe = {"status": "unhealthy", "error": str(exc)}

    return {
        "status": "ok",
        "environment": settings.app_env,
        "dependencies": {
            "qdrant": probes[0],
            "embedding_service": probes[1],
            "langfuse": probes[2],
            "deepseek": deepseek_probe or {"status": "disabled"},
        },
    }


_cors_origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

app.include_router(rag_router.router, prefix="/api/v1")
app.include_router(retrieval_router, prefix="/api/v1")
