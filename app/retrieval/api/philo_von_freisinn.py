"""Agent router for philo-von-freisinn retrieval endpoints."""
from __future__ import annotations

from functools import lru_cache
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.retrieval.services.concept_explain_service import ConceptExplainService
from app.retrieval.services.providers import (
    get_deepseek_client,
    get_embedding_client,
    get_qdrant_client,
)
from app.retrieval.telemetry import retrieval_telemetry
from app.retrieval.models import ConceptExplainResult


router = APIRouter(tags=["concept-explain"])


class ConceptExplainRequest(BaseModel):
    concept: str = Field(..., description="Begriff oder kurze Frage zu genau einem Begriff")
    book_id: Optional[str] = Field(None, description="Optionaler Buchbezug (Metadaten)")
    k: int = Field(10, ge=1, le=20, description="Anzahl primärer Treffer (default 10)")


class RetrievedItem(BaseModel):
    text: str
    score: float
    payload: dict[str, Any]


class ConceptExplainResponse(BaseModel):
    concept: str
    answer: str
    retrieved: List[RetrievedItem]
    expanded: List[RetrievedItem]


class AssistantChatRequest(BaseModel):
    prompt: str = Field(..., description="Freitext-Prompt an den Assistenten")


class AssistantChatResponse(BaseModel):
    branch: str
    answer: str
    concept: Optional[str] = None
    retrieved: Optional[List[RetrievedItem]] = None
    expanded: Optional[List[RetrievedItem]] = None


@lru_cache(maxsize=1)
def get_concept_explain_service() -> ConceptExplainService:
    return ConceptExplainService(
        embedding_client=get_embedding_client(),
        qdrant_client=get_qdrant_client(),
        deepseek_client=get_deepseek_client(),
        collection="philo-von-freisinn",
        k=10,
    )


def _map(items):
    out: List[RetrievedItem] = []
    for it in items:
        out.append(RetrievedItem(text=it.text, score=it.score, payload=dict(it.payload)))
    return out


@router.post(
    "/retrieval/concept-explain",
    response_model=ConceptExplainResponse,
    status_code=status.HTTP_200_OK,
)
async def concept_explain(
    request: ConceptExplainRequest,
    service: ConceptExplainService = Depends(get_concept_explain_service),
) -> ConceptExplainResponse:
    concept = (request.concept or "").strip()
    if not concept:
        raise HTTPException(status_code=400, detail="concept is required")

    try:
        result: ConceptExplainResult = await service.explain(concept)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"concept explanation failed: {exc}",
        ) from exc

    await retrieval_telemetry.record_retrieval(
        trace_id=None,
        agent="philo-von-freisinn",
        branch="concept-explain",
        concept=result.concept,
        retrieved=len(result.retrieved),
        expanded=len(result.expanded),
        metadata={"book_id": request.book_id} if request.book_id else None,
    )

    return ConceptExplainResponse(
        concept=result.concept,
        answer=result.answer,
        retrieved=_map(result.retrieved),
        expanded=_map(result.expanded),
    )


def _looks_like_single_concept(prompt: str) -> Optional[str]:
    cleaned = (prompt or "").strip()
    if not cleaned:
        return None
    tokens = cleaned.split()
    if len(tokens) <= 8 and " und " not in cleaned.lower() and " oder " not in cleaned.lower():
        return cleaned
    return None


@router.post(
    "/chat",
    response_model=AssistantChatResponse,
    status_code=status.HTTP_200_OK,
)
async def assistant_chat(
    request: AssistantChatRequest,
    service: ConceptExplainService = Depends(get_concept_explain_service),
) -> AssistantChatResponse:
    prompt = (request.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")

    concept = _looks_like_single_concept(prompt)
    if concept:
        result: ConceptExplainResult = await service.explain(concept)
        return AssistantChatResponse(
            branch="concept-explain",
            concept=concept,
            answer=result.answer,
            retrieved=_map(result.retrieved),
            expanded=_map(result.expanded),
        )

    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="standard RAG branch not yet implemented; concept branch only",
    )


