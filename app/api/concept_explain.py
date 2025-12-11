"""Endpoint for concept explanations (specialized, deterministic)."""
from __future__ import annotations

from functools import lru_cache
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.config import settings
from app.services.concept_explain_service import ConceptExplainResult, ConceptExplainService
from app.services.deepseek_client import DeepSeekClient
from app.services.embedding_client import EmbeddingClient
from app.services.qdrant_client import QdrantClient


router = APIRouter(prefix="/agent/philo-von-freisinn", tags=["concept-explain"])


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


def _build_service() -> ConceptExplainService:
    if not settings.deepseek_api_key:
        raise RuntimeError("RAGRUN_DEEPSEEK_API_KEY is required for concept explanations")
    embedding_client = EmbeddingClient(settings.embeddings_base_url, batch_size=32)
    qdrant_client = QdrantClient(settings.qdrant_url, api_key=settings.qdrant_api_key)
    deepseek_client = DeepSeekClient(settings.deepseek_api_key)
    return ConceptExplainService(
        embedding_client=embedding_client,
        qdrant_client=qdrant_client,
        deepseek_client=deepseek_client,
        collection="philo-von-freisinn",
        k=10,
    )


@lru_cache(maxsize=1)
def get_concept_explain_service() -> ConceptExplainService:
    return _build_service()


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

    def _map(items):
        out: List[RetrievedItem] = []
        for it in items:
            out.append(RetrievedItem(text=it.text, score=it.score, payload=dict(it.payload)))
        return out

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
        def _map(items):
            out: List[RetrievedItem] = []
            for it in items:
                out.append(RetrievedItem(text=it.text, score=it.score, payload=dict(it.payload)))
            return out

        return AssistantChatResponse(
            branch="concept-explain",
            concept=concept,
            answer=result.answer,
            retrieved=_map(result.retrieved),
            expanded=_map(result.expanded),
        )

    # Fallback not yet implemented – reserve for standard RAG branch
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="standard RAG branch not yet implemented; concept branch only",
    )
