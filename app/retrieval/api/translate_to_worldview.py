"""API router for translate-to-worldview (Sigrid per-worldview prompts)."""

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.retrieval.models import TranslateToWorldviewResult, WorldviewAnswer
from app.retrieval.services.providers import (
    get_deepseek_chat,
    get_embedding_client,
    get_qdrant_client,
)
from app.retrieval.services.translate_to_worldview_service import TranslateToWorldviewService
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


router = APIRouter(tags=["translate-to-worldview"])


class WorldviewAnswerDTO(BaseModel):
    worldview: str
    main_points: str
    how_details: str
    context1_refs: List[str]
    context2_refs: List[str]
    sufficiency: str
    errors: List[str] | None = None


class TranslateToWorldviewRequest(BaseModel):
    text: str = Field(..., min_length=1)
    worldviews: List[str] = Field(..., min_items=1)
    concept: Optional[str] = None
    verbose: bool = False
    retries: int = Field(3, ge=0, le=10, description="LLM retries per step (default 3)")
    max_concurrency: int = Field(4, ge=1, le=16)
    hybrid: bool | None = None


class TranslateToWorldviewResponse(BaseModel):
    input_text: str
    worldviews: List[WorldviewAnswerDTO]
    graph_event_id: Optional[str] = None


@lru_cache(maxsize=1)
def get_service() -> TranslateToWorldviewService:
    return TranslateToWorldviewService(
        embedding_client=get_embedding_client(),
        qdrant_client=get_qdrant_client(),
        chat_client=get_deepseek_chat(),
        collection="sigrid-von-gleich",
        max_concurrency=4,
    )


@router.post(
    "/graphs/translate-to-worldview",
    response_model=TranslateToWorldviewResponse,
    status_code=status.HTTP_200_OK,
)
async def translate_to_worldview(
    request: TranslateToWorldviewRequest,
    service: TranslateToWorldviewService = Depends(get_service),
) -> TranslateToWorldviewResponse:
    text = (request.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    if not request.worldviews:
        raise HTTPException(status_code=400, detail="worldviews must not be empty")
    worldviews = _validate_worldviews(request.worldviews)

    try:
        result: TranslateToWorldviewResult = await service.translate(
            text=text,
            worldviews=worldviews,
            concept=(request.concept or "").strip() or None,
            verbose=bool(request.verbose),
            llm_retries=int(request.retries),
        )
    except ValueError as exc:
        # Used for prompt-missing fail-fast and other user-facing validation.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"translate to worldview failed: {exc}",
        ) from exc

    return TranslateToWorldviewResponse(
        input_text=result.input_text,
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
        graph_event_id=result.graph_event_id,
    )

