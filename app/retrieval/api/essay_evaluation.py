"""API router for essay:evaluation graph."""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.retrieval.models import EssayEvaluationResult
from app.retrieval.services.essay_evaluation_service import EssayEvaluationService
from app.retrieval.services.providers import get_deepseek_chat

router = APIRouter()


class EssayEvaluationRequest(BaseModel):
    assistant: str = Field(..., min_length=1)
    essay_slug: str = Field(..., min_length=1)
    essay_title: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    mood_index: Optional[int] = Field(None, ge=1, le=7)
    mood_name: Optional[str] = None
    verbose: bool = False
    retries: int = Field(3, ge=0, le=10)


class EssayEvaluationResponse(BaseModel):
    assistant: str
    essay_slug: str
    essay_title: str
    mood_index: Optional[int]
    mood_name: Optional[str]
    overall_score: int
    criteria_scores: Dict[str, int]
    issues: List[str]
    instruction: str
    graph_event_id: Optional[str] = None


@lru_cache(maxsize=1)
def get_service() -> EssayEvaluationService:
    return EssayEvaluationService(chat_client=get_deepseek_chat())


@router.post(
    "/graphs/essay-evaluation",
    response_model=EssayEvaluationResponse,
    status_code=status.HTTP_200_OK,
)
async def essay_evaluation(
    request: EssayEvaluationRequest,
    service: EssayEvaluationService = Depends(get_service),
) -> EssayEvaluationResponse:
    assistant = request.assistant.strip()
    essay_slug = request.essay_slug.strip()
    essay_title = request.essay_title.strip()
    text = request.text.strip()

    if not assistant:
        raise HTTPException(status_code=400, detail="assistant is required")
    if not essay_slug:
        raise HTTPException(status_code=400, detail="essay_slug is required")
    if not essay_title:
        raise HTTPException(status_code=400, detail="essay_title is required")
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    try:
        result: EssayEvaluationResult = await service.evaluate(
            assistant=assistant,
            essay_slug=essay_slug,
            essay_title=essay_title,
            text=text,
            mood_index=request.mood_index,
            mood_name=request.mood_name,
            verbose=bool(request.verbose),
            llm_retries=int(request.retries),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"essay-evaluation failed: {exc}",
        ) from exc

    return EssayEvaluationResponse(
        assistant=result.assistant,
        essay_slug=result.essay_slug,
        essay_title=result.essay_title,
        mood_index=result.mood_index,
        mood_name=result.mood_name,
        overall_score=result.overall_score,
        criteria_scores=dict(result.criteria_scores),
        issues=list(result.issues or []),
        instruction=result.instruction,
        graph_event_id=result.graph_event_id,
    )
