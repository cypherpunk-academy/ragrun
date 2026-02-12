"""API router for authentic concept explanation (Steiner-first)."""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.retrieval.models import AuthenticConceptExplainResult
from app.retrieval.services.authentic_concept_explain_service import AuthenticConceptExplainService
from app.retrieval.services.providers import (
    get_deepseek_chat,
    get_embedding_client,
    get_qdrant_client,
)


router = APIRouter(tags=["authentic-concept-explain"])


class AuthenticConceptExplainRequest(BaseModel):
    concept: str = Field(..., min_length=1)
    verbose: bool = False
    retries: int = Field(3, ge=0, le=10, description="LLM retries per step (default 3)")


class AuthenticConceptExplainResponse(BaseModel):
    concept: str
    steiner_prior_text: str
    verify_refs: List[str]
    verification_report: str
    lexicon_entry: str
    references: Optional[List[Dict[str, Any]]] = None
    graph_event_id: Optional[str] = None


@lru_cache(maxsize=1)
def get_service() -> AuthenticConceptExplainService:
    return AuthenticConceptExplainService(
        embedding_client=get_embedding_client(),
        qdrant_client=get_qdrant_client(),
        chat_client=get_deepseek_chat(),
        collection="philo-von-freisinn",
    )


@router.post(
    "/graphs/authentic-concept-explain",
    response_model=AuthenticConceptExplainResponse,
    status_code=status.HTTP_200_OK,
)
async def authentic_concept_explain(
    request: AuthenticConceptExplainRequest,
    service: AuthenticConceptExplainService = Depends(get_service),
) -> AuthenticConceptExplainResponse:
    concept = (request.concept or "").strip()
    if not concept:
        raise HTTPException(status_code=400, detail="concept is required")

    try:
        result: AuthenticConceptExplainResult = await service.explain(
            concept=concept, verbose=bool(request.verbose), llm_retries=int(request.retries)
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"authentic concept explain failed: {exc}",
        ) from exc

    return AuthenticConceptExplainResponse(
        concept=result.concept,
        steiner_prior_text=result.steiner_prior_text,
        verify_refs=result.verify_refs,
        verification_report=result.verification_report,
        lexicon_entry=result.lexicon_entry,
        references=result.references,
        graph_event_id=result.graph_event_id,
    )

