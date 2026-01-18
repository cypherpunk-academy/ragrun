"""API router for concept-explain-worldviews graph."""
from __future__ import annotations

from functools import lru_cache
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.retrieval.models import ConceptExplainWorldviewsResult, WorldviewAnswer
from app.retrieval.services.concept_explain_worldviews_service import ConceptExplainWorldviewsService
from app.retrieval.services.providers import (
    get_deepseek_chat,
    get_deepseek_reasoner,
    get_embedding_client,
    get_qdrant_client,
)
from app.shared.constants import ALLOWED_WORLDVIEWS

def _validate_worldviews(worldviews: List[str]) -> List[str]:
    cleaned: List[str] = []
    for w in worldviews:
        if not isinstance(w, str):
            raise HTTPException(status_code=400, detail="worldviews entries must be strings")
        w_clean = w.strip()
        if not w_clean:
            raise HTTPException(status_code=400, detail="worldviews entries must be non-empty")
        if w_clean not in ALLOWED_WORLDVIEWS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unknown worldview '{w_clean}'. Allowed values: {', '.join(sorted(ALLOWED_WORLDVIEWS))}"
                ),
            )
        cleaned.append(w_clean)
    return cleaned

router = APIRouter()


class WorldviewAnswerDTO(BaseModel):
    worldview: str
    main_points: str
    how_details: str
    context1_refs: List[str]
    context2_refs: List[str]
    sufficiency: str
    errors: List[str] | None = None


class ConceptExplainWorldviewsResponse(BaseModel):
    concept: str
    concept_explanation: str
    worldviews: List[WorldviewAnswerDTO]
    context_refs: List[str]
    graph_event_id: str | None = None


class ConceptExplainWorldviewsRequest(BaseModel):
    concept: str = Field(..., min_length=1)
    worldviews: List[str] = Field(..., min_items=1)
    verbose: bool = False


@lru_cache(maxsize=1)
def get_service() -> ConceptExplainWorldviewsService:
    return ConceptExplainWorldviewsService(
        embedding_client=get_embedding_client(),
        qdrant_client=get_qdrant_client(),
        reasoning_client=get_deepseek_reasoner(),
        chat_client=get_deepseek_chat(),
        collection="philo-von-freisinn",
        max_concurrency=4,
    )


@router.post(
    "/graphs/concept-explain-worldviews",
    response_model=ConceptExplainWorldviewsResponse,
    status_code=status.HTTP_200_OK,
)
async def concept_explain_worldviews(
    request: ConceptExplainWorldviewsRequest,
    service: ConceptExplainWorldviewsService = Depends(get_service),
) -> ConceptExplainWorldviewsResponse:
    concept = request.concept.strip()
    if not concept:
        raise HTTPException(status_code=400, detail="concept is required")
    if not request.worldviews:
        raise HTTPException(status_code=400, detail="worldviews must not be empty")
    worldviews = _validate_worldviews(request.worldviews)

    try:
        result: ConceptExplainWorldviewsResult = await service.explain(
            concept=concept, worldviews=worldviews, verbose=request.verbose
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"concept explain worldviews failed: {exc}",
        ) from exc

    return ConceptExplainWorldviewsResponse(
        concept=result.concept,
        concept_explanation=result.concept_explanation,
        worldviews=[
            WorldviewAnswerDTO(
                worldview=w.worldview,
                main_points=w.main_points,
                how_details=w.how_details,
                context1_refs=w.context1_refs,
                context2_refs=w.context2_refs,
                sufficiency=w.sufficiency,
                errors=w.errors,
            )
            for w in result.worldviews
        ],
        context_refs=result.context_refs,
        graph_event_id=result.graph_event_id,
    )
