"""API router: typology explain (structured classifikation systems, RAG-grounded)."""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.retrieval.models import TypologyExplainResult
from app.retrieval.services.typology_explain_service import TypologyExplainService
from app.retrieval.services.providers import (
    get_deepseek_chat,
    get_embedding_client,
    get_qdrant_client,
)


router = APIRouter(tags=["typology-explain"])
logger = logging.getLogger(__name__)


def _bad_request(code: str, message: str) -> HTTPException:
    """400 with JSON body detail: { code, message } for clients and logs."""
    logger.warning("typology-explain 400 %s: %s", code, message)
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail={"code": code, "message": message},
    )


class TypologySourceFilter(BaseModel):
    """Filter für Quellbücher, aus denen die Typologie-Infos stammen."""

    source_ids: List[str] = Field(
        default_factory=list,
        description="UUIDs aus book-chunks.jsonl metadata.source_id; werden von ragprep aus books aufgelöst.",
    )
    chunk_types: List[str] = Field(default_factory=lambda: ["book", "secondary_book"])


class TypologyExplainRequest(BaseModel):
    name: str = Field(..., min_length=1)
    aliases: List[str] = Field(default_factory=list)
    known_members: List[str] = Field(
        default_factory=list,
        description="Bekannte Mitglieder der Typologie (aus typologies-to-explain.jsonl `list`). Dienen als Grundlage für die LLM-Erklärung.",
    )
    source_chunks: List[str] = Field(default_factory=list)
    source_filter: TypologySourceFilter | None = None
    verbose: bool = False
    retries: int = Field(3, ge=0, le=10)
    run_verify: bool = True


class TypologyExplainResponse(BaseModel):
    name: str
    members: List[str]
    explanation: str
    aliases: List[str]
    references: Optional[List[Dict[str, Any]]] = None
    graph_event_id: Optional[str] = None


@lru_cache(maxsize=1)
def get_service() -> TypologyExplainService:
    return TypologyExplainService(
        embedding_client=get_embedding_client(),
        qdrant_client=get_qdrant_client(),
        chat_client=get_deepseek_chat(),
        collection="philo-von-freisinn",
    )


@router.post(
    "/graphs/typology-explain",
    response_model=TypologyExplainResponse,
    status_code=status.HTTP_200_OK,
)
async def typology_explain(
    request: TypologyExplainRequest,
    service: TypologyExplainService = Depends(get_service),
) -> TypologyExplainResponse:
    name = (request.name or "").strip()
    if not name:
        raise _bad_request(
            "MISSING_NAME",
            "Parameter 'name' fehlt oder ist leer; er muss den Typologie-Namen enthalten.",
        )

    sf = request.source_filter
    if sf is None:
        raise _bad_request(
            "MISSING_SOURCE_FILTER",
            "source_filter fehlt. Bitte source_filter mit mindestens source_ids "
            "(UUIDs wie rag_chunks.source_id / book-manifest book-id) oder books (Legacy: source_title-Präfixe) senden.",
        )
    sid = [s.strip() for s in (sf.source_ids or []) if isinstance(s, str) and s.strip()]
    if not sid:
        raise _bad_request(
            "EMPTY_SOURCE_IDS",
            "source_filter.source_ids ist leer. "
            "Bitte UUIDs aus book-chunks.jsonl metadata.source_id übergeben.",
        )

    try:
        result: TypologyExplainResult = await service.explain(
            name=name,
            aliases=list(request.aliases or []),
            known_members=list(request.known_members or []),
            source_ids=sid,
            source_chunk_types=list(sf.chunk_types or ["book", "secondary_book"]),
            source_chunks=list(request.source_chunks or []),
            verbose=bool(request.verbose),
            llm_retries=int(request.retries),
            run_verify=bool(request.run_verify),
        )
    except ValueError as exc:
        msg = str(exc).strip() or "Unbekannter Validierungsfehler in der Typologie-Kette."
        raise _bad_request("TYPOLOGY_CHAIN_VALIDATION", msg) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"typology explain failed: {exc}",
        ) from exc

    return TypologyExplainResponse(
        name=result.name,
        members=result.members,
        explanation=result.explanation,
        aliases=result.aliases,
        references=result.references,
        graph_event_id=result.graph_event_id,
    )
