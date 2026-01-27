"""API router for essay:create graph."""
from __future__ import annotations

from functools import lru_cache
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.retrieval.models import EssayCreateResult, EssayCreateStepInfo
from app.retrieval.services.essay_create_service import EssayCreateService
from app.retrieval.services.providers import get_deepseek_chat, get_embedding_client, get_qdrant_client

router = APIRouter()


class EssayCreateRequest(BaseModel):
    assistant: str = Field(..., min_length=1)
    essay_slug: str = Field(..., min_length=1)
    essay_title: str = Field(..., min_length=1)
    pitch_steps: List[str] = Field(..., min_items=7, max_items=7)
    k: int = Field(5, ge=1, le=50)
    verbose: bool = False


class EssayCreateStepInfoDTO(BaseModel):
    step: int
    prompt_file: str


class EssayCreateResponse(BaseModel):
    assistant: str
    essay_slug: str
    essay_title: str
    final_text: str
    steps: List[EssayCreateStepInfoDTO]


@lru_cache(maxsize=1)
def get_service() -> EssayCreateService:
    return EssayCreateService(
        embedding_client=get_embedding_client(),
        qdrant_client=get_qdrant_client(),
        chat_client=get_deepseek_chat(),
    )


@router.post(
    "/graphs/essay-create",
    response_model=EssayCreateResponse,
    status_code=status.HTTP_200_OK,
)
async def essay_create(
    request: EssayCreateRequest,
    service: EssayCreateService = Depends(get_service),
) -> EssayCreateResponse:
    assistant = request.assistant.strip()
    essay_slug = request.essay_slug.strip()
    essay_title = request.essay_title.strip()
    pitch_steps = [str(x or "").strip() for x in request.pitch_steps]

    if not assistant:
        raise HTTPException(status_code=400, detail="assistant is required")
    if not essay_slug:
        raise HTTPException(status_code=400, detail="essay_slug is required")
    if not essay_title:
        raise HTTPException(status_code=400, detail="essay_title is required")
    if len(pitch_steps) != 7:
        raise HTTPException(status_code=400, detail="pitch_steps must have exactly 7 entries")

    try:
        result: EssayCreateResult = await service.create(
            assistant=assistant,
            essay_slug=essay_slug,
            essay_title=essay_title,
            pitch_steps=pitch_steps,
            k=int(request.k),
            verbose=bool(request.verbose),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"essay-create failed: {exc}",
        ) from exc

    return EssayCreateResponse(
        assistant=result.assistant,
        essay_slug=result.essay_slug,
        essay_title=result.essay_title,
        final_text=result.final_text,
        steps=[
            EssayCreateStepInfoDTO(step=s.step, prompt_file=s.prompt_file)
            for s in (result.steps or [])
        ],
    )

