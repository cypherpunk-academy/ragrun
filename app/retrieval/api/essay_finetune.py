"""API router for essay:finetune graph."""
from __future__ import annotations

from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.retrieval.models import EssayFinetuneResult
from app.retrieval.services.essay_finetune_service import EssayFinetuneService
from app.retrieval.services.providers import get_deepseek_chat, get_embedding_client, get_qdrant_client

router = APIRouter()


class EssayFinetuneRequest(BaseModel):
    assistant: str = Field(..., min_length=1)
    essay_slug: str = Field(..., min_length=1)
    essay_title: str = Field(..., min_length=1)
    essay_text: str = Field(..., min_length=1)
    instruction: str = Field(..., min_length=1)
    k: int = Field(5, ge=1, le=50)
    verbose: bool = False


class EssayFinetuneResponse(BaseModel):
    assistant: str
    essay_slug: str
    essay_title: str
    revised_text: str


@lru_cache(maxsize=1)
def get_service() -> EssayFinetuneService:
    return EssayFinetuneService(
        embedding_client=get_embedding_client(),
        qdrant_client=get_qdrant_client(),
        chat_client=get_deepseek_chat(),
    )


@router.post(
    "/graphs/essay-finetune",
    response_model=EssayFinetuneResponse,
    status_code=status.HTTP_200_OK,
)
async def essay_finetune(
    request: EssayFinetuneRequest,
    service: EssayFinetuneService = Depends(get_service),
) -> EssayFinetuneResponse:
    assistant = request.assistant.strip()
    essay_slug = request.essay_slug.strip()
    essay_title = request.essay_title.strip()
    essay_text = request.essay_text.strip()
    instruction = request.instruction.strip()

    if not assistant:
        raise HTTPException(status_code=400, detail="assistant is required")
    if not essay_slug:
        raise HTTPException(status_code=400, detail="essay_slug is required")
    if not essay_title:
        raise HTTPException(status_code=400, detail="essay_title is required")
    if not essay_text:
        raise HTTPException(status_code=400, detail="essay_text is required")
    if not instruction:
        raise HTTPException(status_code=400, detail="instruction is required")

    try:
        result: EssayFinetuneResult = await service.finetune(
            assistant=assistant,
            essay_slug=essay_slug,
            essay_title=essay_title,
            essay_text=essay_text,
            instruction=instruction,
            k=int(request.k),
            verbose=bool(request.verbose),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"essay-finetune failed: {exc}",
        ) from exc

    return EssayFinetuneResponse(
        assistant=result.assistant,
        essay_slug=result.essay_slug,
        essay_title=result.essay_title,
        revised_text=result.revised_text,
    )

