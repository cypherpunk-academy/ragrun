"""Entry point for the ragrun FastAPI application."""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager, suppress
from typing import Any, AsyncGenerator, Dict

# Configure root logger early so all app loggers emit at INFO by default.
# Reads RAGRUN_LOG_LEVEL first (consistent with the RAGRUN_ settings prefix),
# then falls back to LOG_LEVEL, then defaults to INFO.
_log_level_str = (
    os.environ.get("RAGRUN_LOG_LEVEL")
    or os.environ.get("LOG_LEVEL", "INFO")
)
_log_level = getattr(logging, _log_level_str.upper(), logging.INFO)
logging.basicConfig(
    level=_log_level,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)
# httpcore emits very chatty DEBUG lines (connect_tcp, send_request_headers, …)
# that are rarely useful; keep it at WARNING regardless of the root level.
logging.getLogger("httpcore").setLevel(logging.WARNING)

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from fastapi import Depends

from .config import settings
from .api import app_api as app_api_router
from .api import rag as rag_router
from .api import admin as admin_router
from .api.internal_auth import require_internal_key
from .api.limiter import limiter
from .api.chat import router as chat_router
from .api.action_prompt import router as action_prompt_router
from .api.problem_solver import router as problem_solver_router
from .retrieval.api import router as retrieval_router
from .retrieval.graphs.assistant_chat_graph import build_chat_graph
from .core.providers import (
    get_deepseek_reasoner_client,
)
from .db.session import get_engine
from .services.app_talks_repository import PostgresTalksRepository
from .services.pricing_service import update_pricing
from .tools.index import register_all_tools

logger = logging.getLogger(__name__)

_TALK_CLEANUP_INTERVAL_SECONDS = 24 * 60 * 60
_TALK_CLEANUP_RETENTION_DAYS = 7


async def _talk_cleanup_loop() -> None:
    """Welle 5a — loescht unpinned Talks (und via ON DELETE CASCADE ihre Turns)

    aelter als 7 Tage. In-Process-Task statt externem Scheduler (kein neues
    Deployment-Artefakt fuer eine einzelne taegliche Aufraeum-Query).
    """
    talks = PostgresTalksRepository(get_engine())
    while True:
        try:
            await asyncio.sleep(_TALK_CLEANUP_INTERVAL_SECONDS)
            deleted = await talks.delete_stale_unpinned(older_than_days=_TALK_CLEANUP_RETENTION_DAYS)
            if deleted:
                logger.info("Talk cleanup: removed %d stale unpinned talks", deleted)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Talk cleanup loop failed; will retry after next interval")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Chat-Graph (Iter. 1: MemorySaver im RAM)
    # Iter. 2: MemorySaver ersetzen durch AsyncPostgresSaver:
    #   from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    #   checkpointer = AsyncPostgresSaver.from_conn_string(str(settings.postgres_dsn))
    #   await checkpointer.setup()
    #   app.state.chat_graph = build_chat_graph(checkpointer=checkpointer)
    app.state.chat_graph = build_chat_graph()
    logger.info("Chat graph initialized (MemorySaver)")

    app.state.app_tool_registry = register_all_tools()
    logger.info("App tool registry initialized")

    await update_pricing()

    cleanup_task = asyncio.create_task(_talk_cleanup_loop())
    logger.info("Talk cleanup background task started (every %ds)", _TALK_CLEANUP_INTERVAL_SECONDS)

    yield

    cleanup_task.cancel()
    with suppress(asyncio.CancelledError):
        await cleanup_task


app = FastAPI(
    title="ragrun API",
    version="0.3.0",
    summary="LangChain + Qdrant orchestration service",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)


async def _probe(client: httpx.AsyncClient, url: str, headers: dict | None = None) -> Dict[str, Any]:
    """Probe a downstream dependency and normalize the response."""

    try:
        response = await client.get(url, headers=headers or {})
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

    qdrant_health_url = f"{str(settings.qdrant_url).rstrip('/')}/healthz"
    embeddings_health_url = f"{str(settings.embeddings_base_url).rstrip('/')}/api/v1/health/simple"
    langfuse_health_url = f"{str(settings.langfuse_host).rstrip('/')}/api/public/health" if settings.langfuse_host else None
    qdrant_headers = {"api-key": settings.qdrant_api_key} if settings.qdrant_api_key else {}

    async with httpx.AsyncClient(timeout=httpx.Timeout(3.0)) as client:
        probe_coros = [
            _probe(client, qdrant_health_url, headers=qdrant_headers),
            _probe(client, embeddings_health_url),
        ]
        if langfuse_health_url:
            probe_coros.append(_probe(client, langfuse_health_url))
        probes = await asyncio.gather(*probe_coros)

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
            "langfuse": probes[2] if langfuse_health_url else {"status": "disabled"},
            "deepseek": deepseek_probe or {"status": "disabled"},
        },
    }


_cors_origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_methods=["GET", "POST", "PATCH"],
        allow_headers=["*"],
    )

app.include_router(app_api_router.router)
app.include_router(rag_router.router, prefix="/api/v1", dependencies=[Depends(require_internal_key)])
app.include_router(admin_router.router, prefix="/api/v1", dependencies=[Depends(require_internal_key)])
app.include_router(retrieval_router, prefix="/api/v1")
app.include_router(chat_router, prefix="/api/v1")
app.include_router(action_prompt_router, prefix="/api/v1")
app.include_router(problem_solver_router, prefix="/api/v1")
