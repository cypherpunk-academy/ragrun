"""ragapp-facing /app/* API (JWT-protected except health and personalities)."""
from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from sqlalchemy import text

from app.api.auth import AuthUser, get_current_user
from app.config import settings
from app.core.providers import get_sync_engine
from app.db.session import get_engine
from app.retrieval.services.action_prompt_service import list_actions, load_assistant_name
from app.services.app_catalog_repository import PostgresCatalogRepository
from app.services.app_chat_service import compress_app_talk, send_app_chat, summarize_app_talk
from app.services.app_chat_stream_service import stream_app_chat
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
    mode: str | None = None
    model: str | None = None
    context_mode: str | None = None
    context_ids: ChatContextIds | None = None
    context_paragraph_text: str | None = None
    linked_document_id: str | None = None
    document_outline: dict[str, Any] | None = None
    linked_document_content: str | None = None


class ChatResponse(BaseModel):
    talk_id: str
    turn_id: str
    reply: str


class ChatSummarizeResponse(BaseModel):
    summary: str


class ChatCompressResponse(BaseModel):
    summary: str
    compressed_up_to_turn_index: int


class TalkSettingsUpdate(BaseModel):
    pinned: bool | None = None
    mode: str | None = None


class TalksListResponse(BaseModel):
    talks: list[dict[str, Any]]


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


@router.get("/chat/talks", response_model=TalksListResponse)
async def app_list_talks(
    user: Annotated[AuthUser, Depends(get_current_user)],
    q: str | None = Query(None, description="Textsuche in Titel/Zusammenfassung"),
    paragraph_id: str | None = Query(None, description="Absatz-UUID; wenn gesetzt: nur Talks dieses Absatzes"),
    pinned_only: bool = Query(True, description="Nur angepinnte Talks (nur ohne paragraph_id wirksam)"),
    limit: int = Query(30, ge=1, le=100),
) -> TalksListResponse:
    """Kontextbasierte Gesprächssuche für den CHAT-Tab.

    - Mit paragraph_id: alle Talks für diesen Absatz (unabhängig von pinned).
    - Ohne paragraph_id: allgemeine Talks; default pinned_only=true.
    """
    talks = await _talks().search_talks(
        user.user_id,
        q=q or None,
        paragraph_id=paragraph_id or None,
        pinned_only=pinned_only,
        limit=limit,
    )
    return TalksListResponse(talks=talks)


@router.post("/chat", response_model=ChatResponse)
async def app_chat(
    body: ChatRequest,
    request: Request,
    user: Annotated[AuthUser, Depends(get_current_user)],
) -> ChatResponse:
    try:
        result = await send_app_chat(
            request.app.state.chat_graph,
            _talks(),
            user_id=user.user_id,
            user_name=user.email or user.user_id,
            message=body.message,
            personality=body.personality,
            talk_id=body.talk_id,
            mode=body.mode,
            model=body.model,
            context_mode=body.context_mode,
            context_ids=body.context_ids.model_dump() if body.context_ids else None,
            context_paragraph_text=body.context_paragraph_text,
            linked_document_id=body.linked_document_id,
            document_outline=body.document_outline,
            linked_document_content=body.linked_document_content,
            app_tool_registry=request.app.state.app_tool_registry,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return ChatResponse(**result)


@router.post("/chat/stream")
async def app_chat_stream(
    body: ChatRequest,
    request: Request,
    user: Annotated[AuthUser, Depends(get_current_user)],
) -> EventSourceResponse:
    """SSE-Streaming-Variante von /app/chat (Contract §3/§5/§6).

    Validierung vor Stream-Start (SSE-Response committet sofort Status 200 —
    Fehler danach koennen nur noch als `error`-Event, nicht als HTTP-Status,
    gemeldet werden).
    """
    if not body.message.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="message must not be empty")
    if not body.personality.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="personality must not be empty")

    generator = stream_app_chat(
        request.app.state.chat_graph,
        _talks(),
        user_id=user.user_id,
        user_name=user.email or user.user_id,
        message=body.message,
        personality=body.personality,
        talk_id=body.talk_id,
        mode=body.mode,
        model=body.model,
        context_mode=body.context_mode,
        context_ids=body.context_ids.model_dump() if body.context_ids else None,
        context_paragraph_text=body.context_paragraph_text,
        linked_document_id=body.linked_document_id,
        document_outline=body.document_outline,
        linked_document_content=body.linked_document_content,
        app_tool_registry=request.app.state.app_tool_registry,
    )
    return EventSourceResponse(generator)


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


@router.post("/chat/{talk_id}/compress", response_model=ChatCompressResponse)
async def app_compress(
    talk_id: str,
    _user: Annotated[AuthUser, Depends(get_current_user)],
) -> ChatCompressResponse:
    try:
        result = await compress_app_talk(_talks(), talk_id=talk_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return ChatCompressResponse(**result)


@router.patch("/chat/{talk_id}", status_code=status.HTTP_204_NO_CONTENT, response_class=Response)
async def app_update_talk(
    talk_id: str,
    body: TalkSettingsUpdate,
    _user: Annotated[AuthUser, Depends(get_current_user)],
) -> Response:
    """Welle 5a/5c — pinned & mode sind nicht Teil des generischen Sync-Pfads

    (`push_changes` verarbeitet nur `notes`/`bookmarks`), daher eigener Endpoint.
    """
    await _talks().set_talk_settings(talk_id, pinned=body.pinned, mode=body.mode)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete("/chat/{talk_id}/turns", status_code=status.HTTP_204_NO_CONTENT, response_class=Response)
async def app_delete_turns(
    talk_id: str,
    _user: Annotated[AuthUser, Depends(get_current_user)],
    after_index: int = Query(..., ge=0),
) -> Response:
    """Welle 5d — Turn-Aktionen (Bearbeiten/Wiederholen/Truncate).

    Loescht alle Turns ab `after_index` server-seitig, damit `load_talk_turns`
    (naechster Chat-Aufruf) den bearbeiteten/verworfenen Verlauf nicht mehr sieht.
    """
    await _talks().delete_turns_from(talk_id, after_index)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


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
