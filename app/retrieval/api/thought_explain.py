"""API router: thought-explain (LangGraph)."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.retrieval.models import ThoughtExplainResult
from app.retrieval.services.providers import (
    get_deepseek_chat,
    get_embedding_client,
    get_qdrant_client,
)
from app.retrieval.services.thought_explain_service import ThoughtExplainService

router = APIRouter(tags=["thought-explain"])
logger = logging.getLogger(__name__)


def _bad_request(code: str, message: str) -> HTTPException:
    logger.warning("thought-explain 400 %s: %s", code, message)
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail={"code": code, "message": message},
    )


class ThoughtExplainRequest(BaseModel):
    thought_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    quote: str = Field(..., min_length=1)
    theme: str = ""
    alignment: str = ""
    writing_style: str = ""
    collection: Optional[str] = None
    references: List[Dict[str, Any]] = Field(default_factory=list)
    verbose: bool = False
    retries: int = Field(3, ge=0, le=10)


class ThoughtExplainResponse(BaseModel):
    thought_id: str
    explanation: str
    word_count: int
    references: Optional[List[Dict[str, Any]]] = None
    graph_event_id: Optional[str] = None


@router.post(
    "/graphs/thought-explain",
    response_model=ThoughtExplainResponse,
    status_code=status.HTTP_200_OK,
)
async def thought_explain(request: ThoughtExplainRequest) -> ThoughtExplainResponse:
    collection = (request.collection or "").strip() or "philo-von-freisinn"
    thought_id = (request.thought_id or "").strip()
    text = (request.text or "").strip()
    quote = (request.quote or "").strip()
    if not thought_id:
        raise _bad_request("MISSING_THOUGHT_ID", "thought_id fehlt oder ist leer.")
    if not text:
        raise _bad_request("MISSING_TEXT", "text fehlt oder ist leer.")
    if not quote:
        raise _bad_request("MISSING_QUOTE", "quote fehlt oder ist leer.")

    service = ThoughtExplainService(
        embedding_client=get_embedding_client(),
        qdrant_client=get_qdrant_client(),
        chat_client=get_deepseek_chat(),
        collection=collection,
    )
    try:
        result: ThoughtExplainResult = await service.explain(
            thought_id=thought_id,
            text=text,
            quote=quote,
            theme=(request.theme or "").strip(),
            alignment=(request.alignment or "").strip(),
            writing_style=(request.writing_style or "").strip(),
            references=list(request.references or []),
            verbose=bool(request.verbose),
            llm_retries=int(request.retries),
        )
    except ValueError as exc:
        raise _bad_request("THOUGHT_EXPLAIN_VALIDATION", str(exc)) from exc
    except Exception as exc:
        logger.exception("thought-explain failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={"code": "THOUGHT_EXPLAIN_FAILED", "message": str(exc)},
        ) from exc

    return ThoughtExplainResponse(
        thought_id=result.thought_id,
        explanation=result.explanation,
        word_count=result.word_count,
        references=list(result.references) if result.references else None,
        graph_event_id=result.graph_event_id,
    )
