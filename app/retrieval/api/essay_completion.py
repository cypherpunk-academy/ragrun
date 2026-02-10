"""API router for essay:completion graph."""
from __future__ import annotations

import logging
import traceback
from functools import lru_cache
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.retrieval.models import EssayCompletionResult
from app.retrieval.services.essay_completion_service import EssayCompletionService
from app.retrieval.services.providers import get_deepseek_chat, get_embedding_client, get_qdrant_client

logger = logging.getLogger(__name__)

router = APIRouter()


class EssayCompletionRequest(BaseModel):
    assistant: str = Field(..., min_length=1)
    essay_slug: str = Field(..., min_length=1)
    essay_title: str = Field(..., min_length=1)
    mood_index: int = Field(..., ge=1, le=7)
    mood_name: Optional[str] = None
    current_text: Optional[str] = None
    current_header: Optional[str] = None
    previous_parts: Optional[str] = None
    k: int = Field(5, ge=1, le=50)
    verbose: bool = False
    retries: int = Field(3, ge=0, le=10)
    force: bool = False


class EssayCompletionResponse(BaseModel):
    assistant: str
    essay_slug: str
    essay_title: str
    mood_index: int
    mood_name: str
    header: str
    draft_header: str
    draft_text: str
    verification_report: str
    revised_header: str
    revised_text: str
    verify_refs: List[str]
    all_books_refs: List[str]
    references: Optional[List[dict[str, Any]]] = None
    graph_event_id: Optional[str] = None


@lru_cache(maxsize=1)
def get_service() -> EssayCompletionService:
    return EssayCompletionService(
        embedding_client=get_embedding_client(),
        qdrant_client=get_qdrant_client(),
        chat_client=get_deepseek_chat(),
    )


@router.post(
    "/graphs/essay-completion",
    response_model=EssayCompletionResponse,
    status_code=status.HTTP_200_OK,
)
async def essay_completion(
    request: EssayCompletionRequest,
    service: EssayCompletionService = Depends(get_service),
) -> EssayCompletionResponse:
    assistant = request.assistant.strip()
    essay_slug = request.essay_slug.strip()
    essay_title = request.essay_title.strip()

    if not assistant:
        raise HTTPException(status_code=400, detail="assistant is required")
    if not essay_slug:
        raise HTTPException(status_code=400, detail="essay_slug is required")
    if not essay_title:
        raise HTTPException(status_code=400, detail="essay_title is required")
    if int(request.mood_index) >= 2 and (
        request.previous_parts is None or not request.previous_parts.strip()
    ):
        raise HTTPException(
            status_code=400,
            detail="previous_parts is required for mood_index >= 2 (no filesystem fallback)",
        )

    try:
        result: EssayCompletionResult = await service.complete(
            assistant=assistant,
            essay_slug=essay_slug,
            essay_title=essay_title,
            mood_index=int(request.mood_index),
            mood_name=request.mood_name,
            current_text=request.current_text,
            current_header=request.current_header,
            previous_parts=request.previous_parts,
            k=int(request.k),
            verbose=bool(request.verbose),
            llm_retries=int(request.retries),
            force=bool(request.force),
        )
    except ValueError as exc:
        logger.error("ValueError in essay_completion: %s", exc, exc_info=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        error_msg = str(exc) or type(exc).__name__
        error_traceback = traceback.format_exc()
        logger.error(
            "essay-completion failed for assistant=%s, essay_slug=%s, mood_index=%s: %s\n%s",
            assistant,
            essay_slug,
            request.mood_index,
            error_msg,
            error_traceback,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"essay-completion failed: {error_msg}",
        ) from exc

    return EssayCompletionResponse(
        assistant=result.assistant,
        essay_slug=result.essay_slug,
        essay_title=result.essay_title,
        mood_index=result.mood_index,
        mood_name=result.mood_name,
        header=result.header,
        draft_header=result.draft_header,
        draft_text=result.draft_text,
        verification_report=result.verification_report,
        revised_header=result.revised_header,
        revised_text=result.revised_text,
        verify_refs=result.verify_refs,
        all_books_refs=result.all_books_refs,
        references=result.references,
        graph_event_id=result.graph_event_id,
    )
