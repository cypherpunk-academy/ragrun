"""ragapp-facing /app/* API (JWT-protected except health and personalities)."""
from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from pydantic import BaseModel, Field

from sqlalchemy import text

from app.api.auth import AuthUser, get_current_user
from app.config import settings
from app.core.providers import get_sync_engine
from app.db.session import get_engine
from app.retrieval.services.action_prompt_service import list_actions, load_assistant_name
from app.services.app_catalog_repository import PostgresCatalogRepository
from app.services.app_chat_service import send_app_chat, summarize_app_talk
from app.services.app_search_service import app_search
from app.services.app_sync_service import pull_changes, push_changes
from app.services.app_talks_repository import PostgresTalksRepository

router = APIRouter(prefix="/app", tags=["app"])


def _catalog() -> PostgresCatalogRepository:
    return PostgresCatalogRepository(get_engine())


def _talks() -> PostgresTalksRepository:
    return PostgresTalksRepository(get_engine())


class SearchRequest(BaseModel):
    query: str
    types: list[str] | None = None
    limit: int = Field(20, ge=1, le=50)
    collection: str | None = None


class SearchResponse(BaseModel):
    results: list[dict[str, Any]]


class SourcesResponse(BaseModel):
    sources: list[dict[str, Any]]


class SegmentsResponse(BaseModel):
    segments: list[dict[str, Any]]


class ChunkTextResponse(BaseModel):
    chunk_id: str
    source_id: str | None = None
    text: str | None = None
    snippet: str | None = None


class PersonalityItem(BaseModel):
    slug: str
    display_name: str
    avatar_url: str | None = None


class PersonalitiesResponse(BaseModel):
    personalities: list[PersonalityItem]


class ChatContextIds(BaseModel):
    paragraph_id: str | None = None
    source_id: str | None = None
    segment_id: str | None = None
    note_id: str | None = None


class ChatRequest(BaseModel):
    message: str
    personality: str
    talk_id: str | None = None
    context_mode: str | None = None
    context_ids: ChatContextIds | None = None


class ChatResponse(BaseModel):
    talk_id: str
    turn_id: str
    reply: str


class ChatSummarizeResponse(BaseModel):
    summary: str


class SyncPullRequest(BaseModel):
    last_pulled_at: int | None = None
    schema_version: int | None = None


class SyncPushRequest(BaseModel):
    changes: dict[str, Any]
    last_pulled_at: int | None = None


@router.get("/health")
async def app_health() -> dict[str, Any]:
    """Lightweight health for ragapp (no JWT)."""
    try:
        engine = get_sync_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False
    return {
        "status": "ok" if db_ok else "degraded",
        "online": db_ok,
    }


@router.get("/personalities", response_model=PersonalitiesResponse)
async def app_personalities() -> PersonalitiesResponse:
    """List chat personalities (no JWT — public catalogue)."""
    assistant_name = load_assistant_name(settings.app_default_assistant_slug)
    actions = list_actions(assistant_name=assistant_name)
    personalities = [
        PersonalityItem(slug=str(a["id"]), display_name=str(a.get("label") or a["id"]))
        for a in actions
        if a.get("category") == "primary"
    ]
    return PersonalitiesResponse(personalities=personalities)


@router.post("/search", response_model=SearchResponse)
async def app_search_endpoint(
    body: SearchRequest,
    _user: Annotated[AuthUser, Depends(get_current_user)],
) -> SearchResponse:
    results = await app_search(
        query=body.query,
        types=body.types,
        limit=body.limit,
        collection=body.collection,
        engine=get_engine(),
    )
    return SearchResponse(results=results)


@router.get("/sources", response_model=SourcesResponse)
async def app_sources(
    _user: Annotated[AuthUser, Depends(get_current_user)],
) -> SourcesResponse:
    sources = await _catalog().list_sources()
    return SourcesResponse(sources=sources)


@router.get("/sources/{source_id}/segments", response_model=SegmentsResponse)
async def app_segments(
    source_id: str,
    _user: Annotated[AuthUser, Depends(get_current_user)],
) -> SegmentsResponse:
    segments = await _catalog().list_segments(source_id)
    return SegmentsResponse(segments=segments)


@router.get("/chunks/{chunk_id}", response_model=ChunkTextResponse)
async def app_chunk(
    chunk_id: str,
    _user: Annotated[AuthUser, Depends(get_current_user)],
    source_id: str | None = Query(None),
) -> ChunkTextResponse:
    row = await _catalog().get_chunk_text(chunk_id, source_id=source_id)
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="chunk not found")
    return ChunkTextResponse(**row)


@router.post("/chat", response_model=ChatResponse)
async def app_chat(
    body: ChatRequest,
    user: Annotated[AuthUser, Depends(get_current_user)],
) -> ChatResponse:
    try:
        result = await send_app_chat(
            _talks(),
            user_id=user.user_id,
            user_name=user.email or user.user_id,
            message=body.message,
            personality=body.personality,
            talk_id=body.talk_id,
            context_mode=body.context_mode,
            context_ids=body.context_ids.model_dump() if body.context_ids else None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return ChatResponse(**result)


@router.post("/chat/{talk_id}/summarize", response_model=ChatSummarizeResponse)
async def app_summarize(
    talk_id: str,
    _user: Annotated[AuthUser, Depends(get_current_user)],
) -> ChatSummarizeResponse:
    try:
        summary = await summarize_app_talk(_talks(), talk_id=talk_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return ChatSummarizeResponse(summary=summary)


@router.post("/sync/pull")
async def app_sync_pull(
    body: SyncPullRequest,
    user: Annotated[AuthUser, Depends(get_current_user)],
) -> dict[str, Any]:
    return await pull_changes(
        user,
        last_pulled_at=body.last_pulled_at,
        schema_version=body.schema_version,
    )


@router.post("/sync/push", status_code=status.HTTP_204_NO_CONTENT, response_class=Response)
async def app_sync_push(
    body: SyncPushRequest,
    user: Annotated[AuthUser, Depends(get_current_user)],
) -> Response:
    await push_changes(
        user,
        changes=body.changes,
        last_pulled_at=body.last_pulled_at,
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)
