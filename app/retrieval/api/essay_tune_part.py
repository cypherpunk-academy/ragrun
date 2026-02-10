"""API router for essay tune part graph."""
from __future__ import annotations

import logging
import traceback
from functools import lru_cache
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.retrieval.models import EssayCompletionResult
from app.retrieval.services.providers import get_deepseek_chat, get_embedding_client, get_qdrant_client
from app.infra.deepseek_client import DeepSeekClient
from app.infra.embedding_client import EmbeddingClient
from app.infra.qdrant_client import QdrantClient
from app.retrieval.graphs.essay_finetune import run_essay_tune_part_graph

logger = logging.getLogger(__name__)

router = APIRouter()


class EssayTunePartRequest(BaseModel):
    assistant: str = Field(..., min_length=1)
    essay_slug: str = Field(..., min_length=1)
    essay_title: str = Field(..., min_length=1)
    mood_index: int = Field(..., ge=1, le=7)
    mood_name: Optional[str] = None
    current_text: str = Field(..., min_length=1)
    current_header: Optional[str] = None
    previous_parts: Optional[str] = None
    modifications: str = Field(..., min_length=1)
    k: int = Field(5, ge=1, le=50)
    verbose: bool = False
    retries: int = Field(3, ge=0, le=10)


class EssayTunePartResponse(BaseModel):
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
    graph_event_id: Optional[str] = None


class EssayTunePartService:
    def __init__(
        self,
        *,
        embedding_client: EmbeddingClient,
        qdrant_client: QdrantClient,
        chat_client: DeepSeekClient,
    ) -> None:
        self.embedding_client = embedding_client
        self.qdrant_client = qdrant_client
        self.chat_client = chat_client

    async def tune(
        self,
        *,
        assistant: str,
        essay_slug: str,
        essay_title: str,
        mood_index: int,
        mood_name: str | None,
        current_text: str,
        current_header: str | None,
        previous_parts: str | None,
        modifications: str,
        k: int = 5,
        verbose: bool = False,
        llm_retries: int = 3,
    ) -> EssayCompletionResult:
        return await run_essay_tune_part_graph(
            assistant=assistant,
            essay_slug=essay_slug,
            essay_title=essay_title,
            mood_index=mood_index,
            mood_name=mood_name,
            current_text=current_text,
            current_header=current_header,
            previous_parts=previous_parts,
            modifications=modifications,
            k=k,
            embedding_client=self.embedding_client,
            qdrant_client=self.qdrant_client,
            chat_client=self.chat_client,
            verbose=verbose,
            llm_retries=llm_retries,
        )


@lru_cache(maxsize=1)
def get_service() -> EssayTunePartService:
    return EssayTunePartService(
        embedding_client=get_embedding_client(),
        qdrant_client=get_qdrant_client(),
        chat_client=get_deepseek_chat(),
    )


@router.post(
    "/graphs/essay-tune-part",
    response_model=EssayTunePartResponse,
    status_code=status.HTTP_200_OK,
)
async def essay_tune_part(
    request: EssayTunePartRequest,
    service: EssayTunePartService = Depends(get_service),
) -> EssayTunePartResponse:
    assistant = request.assistant.strip()
    essay_slug = request.essay_slug.strip()
    essay_title = request.essay_title.strip()
    current_text = request.current_text.strip()
    modifications = request.modifications.strip()

    if not assistant:
        raise HTTPException(status_code=400, detail="assistant is required")
    if not essay_slug:
        raise HTTPException(status_code=400, detail="essay_slug is required")
    if not essay_title:
        raise HTTPException(status_code=400, detail="essay_title is required")
    if not current_text:
        raise HTTPException(status_code=400, detail="current_text is required")
    if not modifications:
        raise HTTPException(status_code=400, detail="modifications is required")

    try:
        result: EssayCompletionResult = await service.tune(
            assistant=assistant,
            essay_slug=essay_slug,
            essay_title=essay_title,
            mood_index=int(request.mood_index),
            mood_name=request.mood_name,
            current_text=current_text,
            current_header=request.current_header,
            previous_parts=request.previous_parts,
            modifications=modifications,
            k=int(request.k),
            verbose=bool(request.verbose),
            llm_retries=int(request.retries),
        )
    except ValueError as exc:
        logger.error("ValueError in essay_tune_part: %s", exc, exc_info=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        error_msg = str(exc) or type(exc).__name__
        error_traceback = traceback.format_exc()
        logger.error(
            "essay-tune-part failed for assistant=%s, essay_slug=%s, mood_index=%s: %s\n%s",
            assistant,
            essay_slug,
            request.mood_index,
            error_msg,
            error_traceback,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"essay-tune-part failed: {error_msg}",
        ) from exc

    return EssayTunePartResponse(
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
        graph_event_id=result.graph_event_id,
    )
