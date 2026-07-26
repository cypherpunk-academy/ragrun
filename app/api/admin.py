"""Admin endpoints for dashboard stats and talk management."""
from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import and_, func, select, update

from ..core.providers import get_sync_engine
from ..db.tables import (
    rag_chunks_table,
    rag_references_table,
    rag_talks_table,
    rag_turns_table,
    rag_usage_table,
    users_table,
    vector_chunks_table,
)

router = APIRouter(prefix="/admin", tags=["admin"])


class CollectionStat(BaseModel):
    name: str
    count: int


class UsageByModel(BaseModel):
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    call_count: int


class StatusCount(BaseModel):
    status: str
    count: int


class UsageTimeseriesPoint(BaseModel):
    day: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class AdminStatsResponse(BaseModel):
    rag_chunks_total: int
    rag_chunks_embedded: int
    rag_chunks_deprecated: int
    rag_chunks_by_partition: List[CollectionStat]
    vector_chunks_total: int
    vector_chunks_by_collection: List[CollectionStat]
    rag_talks_total: int
    rag_talks_by_status: List[StatusCount]
    rag_turns_total: int
    avg_turns_per_talk: float
    rag_references_total: int
    avg_refs_per_turn: float
    rag_usage_total_calls: int
    rag_usage_total_tokens: int
    rag_usage_by_model: List[UsageByModel]
    rag_usage_timeseries: List[UsageTimeseriesPoint]


class TalkSummary(BaseModel):
    talk_id: str
    collection: str
    title: str
    user_name: str
    publishing_status: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    turn_count: int = 0


class AdminTalksResponse(BaseModel):
    total: int
    items: List[TalkSummary]


class TalkReference(BaseModel):
    ref_id: str
    ref_index: int
    chunk_id: Optional[str] = None
    relevance: Optional[float] = None
    source_title: Optional[str] = None
    segment_title: Optional[str] = None


class TalkUsage(BaseModel):
    id: int
    model: Optional[str] = None
    provider: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    created_at: Optional[str] = None


class TalkTurn(BaseModel):
    turn_id: str
    turn_index: int
    user_message: str
    assistant_message: str
    updated_at: Optional[str] = None
    references: List[TalkReference]
    usage: List[TalkUsage]


class TalkDetails(BaseModel):
    talk_id: str
    collection: str
    title: str
    user_name: str
    publishing_status: str
    summary: Optional[str] = None
    turns: List[TalkTurn]


class UpdateTurnRequest(BaseModel):
    user_message: str = Field(..., min_length=1)


class UpdateTalkStatusRequest(BaseModel):
    publishing_status: str = Field(..., min_length=1)


class UpsertUserRequest(BaseModel):
    github_id: str
    github_login: str
    email: Optional[str] = None
    name: Optional[str] = None
    avatar_url: Optional[str] = None


class UpsertUserResponse(BaseModel):
    user_id: str
    github_login: str
    role: str


@router.get("/collections", response_model=List[CollectionStat])
async def list_collections() -> List[CollectionStat]:
    engine = get_sync_engine()

    def _fetch() -> List[CollectionStat]:
        with engine.begin() as conn:
            rows = conn.execute(
                select(
                    vector_chunks_table.c.collection.label("name"),
                    func.count().label("count"),
                )
                .group_by(vector_chunks_table.c.collection)
                .order_by(func.count().desc())
            ).mappings()
            return [CollectionStat(name=str(r["name"]), count=int(r["count"])) for r in rows]

    return await asyncio.to_thread(_fetch)


@router.get("/stats", response_model=AdminStatsResponse)
async def get_admin_stats() -> AdminStatsResponse:
    engine = get_sync_engine()

    def _fetch() -> AdminStatsResponse:
        with engine.begin() as conn:
            rag_chunks_total = int(conn.execute(select(func.count()).select_from(rag_chunks_table)).scalar() or 0)
            rag_chunks_embedded = int(
                conn.execute(
                    select(func.count()).select_from(rag_chunks_table).where(rag_chunks_table.c.embedded_at.is_not(None))
                ).scalar()
                or 0
            )
            rag_chunks_deprecated = int(
                conn.execute(
                    select(func.count()).select_from(rag_chunks_table).where(rag_chunks_table.c.deprecated_at.is_not(None))
                ).scalar()
                or 0
            )
            rag_chunks_by_partition_rows = conn.execute(
                select(rag_chunks_table.c.rag_partition, func.count().label("count"))
                .group_by(rag_chunks_table.c.rag_partition)
                .order_by(func.count().desc())
            ).all()

            vector_chunks_total = int(
                conn.execute(select(func.count()).select_from(vector_chunks_table)).scalar() or 0
            )
            vector_chunks_by_collection_rows = conn.execute(
                select(vector_chunks_table.c.collection, func.count().label("count"))
                .group_by(vector_chunks_table.c.collection)
                .order_by(func.count().desc())
            ).all()

            rag_talks_total = int(conn.execute(select(func.count()).select_from(rag_talks_table)).scalar() or 0)
            talks_status_rows = conn.execute(
                select(rag_talks_table.c.publishing_status, func.count().label("count"))
                .group_by(rag_talks_table.c.publishing_status)
            ).all()

            rag_turns_total = int(conn.execute(select(func.count()).select_from(rag_turns_table)).scalar() or 0)
            rag_references_total = int(conn.execute(select(func.count()).select_from(rag_references_table)).scalar() or 0)
            rag_usage_total_calls = int(conn.execute(select(func.count()).select_from(rag_usage_table)).scalar() or 0)
            rag_usage_total_tokens = int(
                conn.execute(select(func.coalesce(func.sum(rag_usage_table.c.total_tokens), 0))).scalar() or 0
            )

            usage_by_model_rows = conn.execute(
                select(
                    func.coalesce(rag_usage_table.c.model, "unknown").label("model"),
                    func.coalesce(func.sum(rag_usage_table.c.prompt_tokens), 0).label("prompt_tokens"),
                    func.coalesce(func.sum(rag_usage_table.c.completion_tokens), 0).label("completion_tokens"),
                    func.coalesce(func.sum(rag_usage_table.c.total_tokens), 0).label("total_tokens"),
                    func.count().label("call_count"),
                )
                .group_by(rag_usage_table.c.model)
                .order_by(func.coalesce(func.sum(rag_usage_table.c.total_tokens), 0).desc())
            ).all()

            usage_ts_rows = conn.execute(
                select(
                    func.date_trunc("day", rag_usage_table.c.created_at).label("day"),
                    func.coalesce(func.sum(rag_usage_table.c.prompt_tokens), 0).label("prompt_tokens"),
                    func.coalesce(func.sum(rag_usage_table.c.completion_tokens), 0).label("completion_tokens"),
                    func.coalesce(func.sum(rag_usage_table.c.total_tokens), 0).label("total_tokens"),
                )
                .group_by(func.date_trunc("day", rag_usage_table.c.created_at))
                .order_by(func.date_trunc("day", rag_usage_table.c.created_at).desc())
                .limit(30)
            ).all()

            avg_turns_per_talk = float(rag_turns_total / rag_talks_total) if rag_talks_total else 0.0
            avg_refs_per_turn = float(rag_references_total / rag_turns_total) if rag_turns_total else 0.0

            return AdminStatsResponse(
                rag_chunks_total=rag_chunks_total,
                rag_chunks_embedded=rag_chunks_embedded,
                rag_chunks_deprecated=rag_chunks_deprecated,
                rag_chunks_by_partition=[
                    CollectionStat(name=str(name), count=int(count)) for name, count in rag_chunks_by_partition_rows
                ],
                vector_chunks_total=vector_chunks_total,
                vector_chunks_by_collection=[
                    CollectionStat(name=str(name), count=int(count)) for name, count in vector_chunks_by_collection_rows
                ],
                rag_talks_total=rag_talks_total,
                rag_talks_by_status=[StatusCount(status=str(s), count=int(c)) for s, c in talks_status_rows],
                rag_turns_total=rag_turns_total,
                avg_turns_per_talk=avg_turns_per_talk,
                rag_references_total=rag_references_total,
                avg_refs_per_turn=avg_refs_per_turn,
                rag_usage_total_calls=rag_usage_total_calls,
                rag_usage_total_tokens=rag_usage_total_tokens,
                rag_usage_by_model=[
                    UsageByModel(
                        model=str(model),
                        prompt_tokens=int(prompt_tokens),
                        completion_tokens=int(completion_tokens),
                        total_tokens=int(total_tokens),
                        call_count=int(call_count),
                    )
                    for model, prompt_tokens, completion_tokens, total_tokens, call_count in usage_by_model_rows
                ],
                rag_usage_timeseries=[
                    UsageTimeseriesPoint(
                        day=day.date().isoformat() if isinstance(day, datetime) else str(day),
                        prompt_tokens=int(prompt_tokens),
                        completion_tokens=int(completion_tokens),
                        total_tokens=int(total_tokens),
                    )
                    for day, prompt_tokens, completion_tokens, total_tokens in usage_ts_rows
                ],
            )

    return await asyncio.to_thread(_fetch)


@router.get("/talks", response_model=AdminTalksResponse)
async def list_talks(
    collection: Optional[str] = Query(None),
    q: Optional[str] = Query(None),
    statuses: Optional[str] = Query(None, description="Comma-separated statuses"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> AdminTalksResponse:
    engine = get_sync_engine()

    def _fetch() -> AdminTalksResponse:
        with engine.begin() as conn:
            filters: list[Any] = []
            if collection:
                filters.append(rag_talks_table.c.collection == collection)

            status_values: list[str] = []
            if statuses:
                status_values = [s.strip() for s in statuses.split(",") if s.strip()]
                if status_values:
                    filters.append(rag_talks_table.c.publishing_status.in_(status_values))

            if q:
                needle = f"%{q.strip()}%"
                filters.append(
                    (rag_talks_table.c.title.ilike(needle))
                    | (rag_talks_table.c.user_name.ilike(needle))
                )

            where_clause = and_(*filters) if filters else None
            count_stmt = select(func.count()).select_from(rag_talks_table)
            if where_clause is not None:
                count_stmt = count_stmt.where(where_clause)
            total = int(conn.execute(count_stmt).scalar() or 0)

            stmt = (
                select(
                    rag_talks_table.c.talk_id,
                    rag_talks_table.c.collection,
                    rag_talks_table.c.title,
                    rag_talks_table.c.user_name,
                    rag_talks_table.c.publishing_status,
                    rag_talks_table.c.created_at,
                    rag_talks_table.c.updated_at,
                    func.count(rag_turns_table.c.turn_id).label("turn_count"),
                )
                .select_from(rag_talks_table.outerjoin(rag_turns_table, rag_turns_table.c.talk_id == rag_talks_table.c.talk_id))
                .group_by(
                    rag_talks_table.c.talk_id,
                    rag_talks_table.c.collection,
                    rag_talks_table.c.title,
                    rag_talks_table.c.user_name,
                    rag_talks_table.c.publishing_status,
                    rag_talks_table.c.created_at,
                    rag_talks_table.c.updated_at,
                )
                .order_by(rag_talks_table.c.updated_at.desc())
                .limit(limit)
                .offset(offset)
            )
            if where_clause is not None:
                stmt = stmt.where(where_clause)
            rows = conn.execute(stmt).all()

            return AdminTalksResponse(
                total=total,
                items=[
                    TalkSummary(
                        talk_id=str(row.talk_id),
                        collection=str(row.collection),
                        title=str(row.title),
                        user_name=str(row.user_name or ""),
                        publishing_status=str(row.publishing_status),
                        created_at=row.created_at.isoformat() if row.created_at else None,
                        updated_at=row.updated_at.isoformat() if row.updated_at else None,
                        turn_count=int(row.turn_count or 0),
                    )
                    for row in rows
                ],
            )

    return await asyncio.to_thread(_fetch)


@router.get("/talks/{talk_id}", response_model=TalkDetails)
async def get_talk_details(talk_id: str) -> TalkDetails:
    engine = get_sync_engine()

    def _fetch() -> TalkDetails:
        with engine.begin() as conn:
            talk = conn.execute(
                select(
                    rag_talks_table.c.talk_id,
                    rag_talks_table.c.collection,
                    rag_talks_table.c.title,
                    rag_talks_table.c.user_name,
                    rag_talks_table.c.publishing_status,
                    rag_talks_table.c.summary,
                ).where(rag_talks_table.c.talk_id == talk_id)
            ).first()
            if talk is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Talk not found")

            turns = conn.execute(
                select(
                    rag_turns_table.c.turn_id,
                    rag_turns_table.c.turn_index,
                    rag_turns_table.c.user_message,
                    rag_turns_table.c.assistant_message,
                    rag_turns_table.c.updated_at,
                )
                .where(rag_turns_table.c.talk_id == talk_id)
                .order_by(rag_turns_table.c.turn_index.asc())
            ).all()

            turn_ids = [str(t.turn_id) for t in turns]
            refs_by_turn: dict[str, list[TalkReference]] = defaultdict(list)
            usage_by_turn: dict[str, list[TalkUsage]] = defaultdict(list)

            if turn_ids:
                refs = conn.execute(
                    select(
                        rag_references_table.c.ref_id,
                        rag_references_table.c.turn_id,
                        rag_references_table.c.ref_index,
                        rag_references_table.c.chunk_id,
                        rag_references_table.c.relevance,
                        rag_references_table.c.source_title,
                        rag_references_table.c.segment_title,
                    )
                    .where(rag_references_table.c.turn_id.in_(turn_ids))
                    .order_by(rag_references_table.c.turn_id.asc(), rag_references_table.c.ref_index.asc())
                ).all()
                for ref in refs:
                    refs_by_turn[str(ref.turn_id)].append(
                        TalkReference(
                            ref_id=str(ref.ref_id),
                            ref_index=int(ref.ref_index),
                            chunk_id=ref.chunk_id,
                            relevance=float(ref.relevance) if ref.relevance is not None else None,
                            source_title=ref.source_title,
                            segment_title=ref.segment_title,
                        )
                    )

                usage_rows = conn.execute(
                    select(
                        rag_usage_table.c.id,
                        rag_usage_table.c.turn_id,
                        rag_usage_table.c.model,
                        rag_usage_table.c.provider,
                        rag_usage_table.c.prompt_tokens,
                        rag_usage_table.c.completion_tokens,
                        rag_usage_table.c.total_tokens,
                        rag_usage_table.c.created_at,
                    )
                    .where(rag_usage_table.c.turn_id.in_(turn_ids))
                    .order_by(rag_usage_table.c.created_at.asc())
                ).all()
                for usage in usage_rows:
                    usage_by_turn[str(usage.turn_id)].append(
                        TalkUsage(
                            id=int(usage.id),
                            model=usage.model,
                            provider=str(usage.provider),
                            prompt_tokens=usage.prompt_tokens,
                            completion_tokens=usage.completion_tokens,
                            total_tokens=usage.total_tokens,
                            created_at=usage.created_at.isoformat() if usage.created_at else None,
                        )
                    )

            return TalkDetails(
                talk_id=str(talk.talk_id),
                collection=str(talk.collection),
                title=str(talk.title),
                user_name=str(talk.user_name or ""),
                publishing_status=str(talk.publishing_status),
                summary=talk.summary,
                turns=[
                    TalkTurn(
                        turn_id=str(turn.turn_id),
                        turn_index=int(turn.turn_index),
                        user_message=str(turn.user_message),
                        assistant_message=str(turn.assistant_message),
                        updated_at=turn.updated_at.isoformat() if turn.updated_at else None,
                        references=refs_by_turn.get(str(turn.turn_id), []),
                        usage=usage_by_turn.get(str(turn.turn_id), []),
                    )
                    for turn in turns
                ],
            )

    return await asyncio.to_thread(_fetch)


@router.patch("/turns/{turn_id}")
async def update_turn(turn_id: str, request: UpdateTurnRequest) -> Dict[str, Any]:
    engine = get_sync_engine()

    def _write() -> Dict[str, Any]:
        with engine.begin() as conn:
            result = conn.execute(
                update(rag_turns_table)
                .where(rag_turns_table.c.turn_id == turn_id)
                .values(user_message=request.user_message, updated_at=func.now())
            )
            if result.rowcount == 0:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Turn not found")
            return {"updated": True, "turn_id": turn_id}

    return await asyncio.to_thread(_write)


@router.patch("/talks/{talk_id}")
async def update_talk_status(talk_id: str, request: UpdateTalkStatusRequest) -> Dict[str, Any]:
    allowed = {"draft", "published", "personal", "bug"}
    next_status = request.publishing_status.strip().lower()
    if next_status not in allowed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid status: {next_status}. Allowed: {', '.join(sorted(allowed))}",
        )

    engine = get_sync_engine()

    def _write() -> Dict[str, Any]:
        with engine.begin() as conn:
            result = conn.execute(
                update(rag_talks_table)
                .where(rag_talks_table.c.talk_id == talk_id)
                .values(publishing_status=next_status, updated_at=func.now())
            )
            if result.rowcount == 0:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Talk not found")
            return {"updated": True, "talk_id": talk_id, "publishing_status": next_status}

    return await asyncio.to_thread(_write)


@router.post("/users/upsert", response_model=UpsertUserResponse)
async def upsert_user(request: UpsertUserRequest) -> UpsertUserResponse:
    engine = get_sync_engine()

    def _write() -> UpsertUserResponse:
        with engine.begin() as conn:
            existing = conn.execute(
                select(users_table.c.user_id, users_table.c.role).where(users_table.c.github_id == request.github_id)
            ).first()
            if existing:
                conn.execute(
                    update(users_table)
                    .where(users_table.c.github_id == request.github_id)
                    .values(
                        github_login=request.github_login,
                        email=request.email,
                        name=request.name,
                        avatar_url=request.avatar_url,
                        updated_at=func.now(),
                    )
                )
                return UpsertUserResponse(
                    user_id=str(existing.user_id),
                    github_login=request.github_login,
                    role=str(existing.role),
                )

            created = conn.execute(
                users_table.insert()
                .values(
                    github_id=request.github_id,
                    github_login=request.github_login,
                    email=request.email,
                    name=request.name,
                    avatar_url=request.avatar_url,
                )
                .returning(users_table.c.user_id, users_table.c.role)
            ).first()
            if created is None:
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create user")
            return UpsertUserResponse(
                user_id=str(created.user_id),
                github_login=request.github_login,
                role=str(created.role),
            )

    return await asyncio.to_thread(_write)
