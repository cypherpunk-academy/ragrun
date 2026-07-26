"""Talk persistence for /app/chat."""
from __future__ import annotations

import asyncio
import json
from typing import Any, Sequence

from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.db.ports import TalksPort


class PostgresTalksRepository(TalksPort):
    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    async def create_talk_turn(
        self,
        *,
        user_id: str,
        user_name: str,
        collection: str,
        personality: str,
        title: str,
        user_message: str,
        assistant_message: str,
        talk_id: str | None = None,
        kontext_meta: dict[str, Any] | None = None,
        usage: dict[str, Any] | None = None,
        chunk_index_map: list[dict[str, Any]] | None = None,
    ) -> dict[str, str]:
        km = kontext_meta or {}
        kontext_source_id = km.get("sourceId") or km.get("source_id")
        kontext_paragraph_id = km.get("segmentId") or km.get("paragraph_id")
        kontext_paragraph = km.get("paragraphHint") or km.get("paragraph_hint")

        def _write() -> dict[str, str]:
            with self._engine.begin() as conn:
                talk_id_val: str | None
                if talk_id:
                    existing = conn.execute(
                        text("SELECT talk_id::text FROM rag_talks WHERE talk_id = :talk_id"),
                        {"talk_id": talk_id},
                    ).first()
                    talk_id_val = str(existing[0]) if existing else None
                else:
                    talk_id_val = None

                if not talk_id_val:
                    row = conn.execute(
                        text(
                            """
                            INSERT INTO rag_talks
                              (talk_id, collection, user_id, user_name, title, personality,
                               kontext_source_id, kontext_paragraph_id, kontext_paragraph,
                               publishing_status, updated_at)
                            VALUES
                              (COALESCE(CAST(:talk_id AS uuid), gen_random_uuid()),
                               :collection, :user_id, :user_name, :title, :personality,
                               :kontext_source_id, :kontext_paragraph_id, :kontext_paragraph,
                               'personal', now())
                            RETURNING talk_id::text
                            """
                        ),
                        {
                            "talk_id": talk_id,
                            "collection": collection,
                            "user_id": user_id,
                            "user_name": user_name,
                            "title": title[:500] if title else user_message[:120],
                            "personality": personality,
                            "kontext_source_id": kontext_source_id,
                            "kontext_paragraph_id": kontext_paragraph_id,
                            "kontext_paragraph": kontext_paragraph,
                        },
                    ).first()
                    talk_id_val = str(row[0])
                    turn_index = 0
                else:
                    max_idx = conn.execute(
                        text(
                            "SELECT COALESCE(MAX(turn_index), -1) FROM rag_turns WHERE talk_id = :talk_id"
                        ),
                        {"talk_id": talk_id_val},
                    ).scalar()
                    turn_index = int(max_idx) + 1  # COALESCE garantiert non-NULL; kein `or -1` (0 wäre falsy)
                    conn.execute(
                        text("UPDATE rag_talks SET updated_at = now() WHERE talk_id = :talk_id"),
                        {"talk_id": talk_id_val},
                    )

                turn_row = conn.execute(
                    text(
                        """
                        INSERT INTO rag_turns
                          (talk_id, turn_index, personality, user_message, assistant_message,
                           usage, collection, kontext_meta, chunk_index_map)
                        VALUES
                          (:talk_id, :turn_index, :personality, :user_message, :assistant_message,
                           CAST(:usage AS jsonb), :collection, CAST(:kontext_meta AS jsonb),
                           CAST(:chunk_index_map AS jsonb))
                        RETURNING turn_id::text
                        """
                    ),
                    {
                        "talk_id": talk_id_val,
                        "turn_index": turn_index,
                        "personality": personality,
                        "user_message": user_message,
                        "assistant_message": assistant_message,
                        "usage": json.dumps(usage) if usage else None,
                        "collection": collection,
                        "kontext_meta": json.dumps(kontext_meta) if kontext_meta else None,
                        "chunk_index_map": json.dumps(chunk_index_map) if chunk_index_map else None,
                    },
                ).first()
                turn_id_val = str(turn_row[0])

            return {"talk_id": talk_id_val, "turn_id": turn_id_val}

        return await asyncio.to_thread(_write)

    async def load_talk_turns(self, talk_id: str) -> Sequence[dict[str, Any]]:
        def _read() -> list[dict[str, Any]]:
            with self._engine.connect() as conn:
                rows = conn.execute(
                    text(
                        """
                        SELECT turn_index, user_message, assistant_message
                        FROM rag_turns
                        WHERE talk_id = :talk_id
                        ORDER BY turn_index ASC
                        """
                    ),
                    {"talk_id": talk_id},
                ).mappings().all()
            return [dict(r) for r in rows]

        return await asyncio.to_thread(_read)

    async def set_talk_settings(
        self, talk_id: str, *, pinned: bool | None = None, mode: str | None = None
    ) -> None:
        set_clauses: list[str] = []
        params: dict[str, Any] = {"talk_id": talk_id}
        if pinned is not None:
            set_clauses.append("pinned = :pinned")
            params["pinned"] = pinned
        if mode is not None:
            set_clauses.append("mode = :mode")
            params["mode"] = mode
        if not set_clauses:
            return
        set_clauses.append("updated_at = now()")

        def _write() -> None:
            with self._engine.begin() as conn:
                conn.execute(
                    text(f"UPDATE rag_talks SET {', '.join(set_clauses)} WHERE talk_id = :talk_id"),
                    params,
                )

        await asyncio.to_thread(_write)

    async def delete_stale_unpinned(self, *, older_than_days: int = 7) -> int:
        def _write() -> int:
            with self._engine.begin() as conn:
                result = conn.execute(
                    text(
                        """
                        DELETE FROM rag_talks
                        WHERE pinned = false
                          AND updated_at < now() - make_interval(days => :older_than_days)
                        """
                    ),
                    {"older_than_days": older_than_days},
                )
                return result.rowcount or 0

        return await asyncio.to_thread(_write)

    async def delete_turns_from(self, talk_id: str, after_index: int) -> int:
        """Welle 5d — loescht alle Turns mit turn_index >= after_index (Bearbeiten/Wiederholen)."""

        def _write() -> int:
            with self._engine.begin() as conn:
                result = conn.execute(
                    text(
                        "DELETE FROM rag_turns WHERE talk_id = :talk_id AND turn_index >= :after_index"
                    ),
                    {"talk_id": talk_id, "after_index": after_index},
                )
                return result.rowcount or 0

        return await asyncio.to_thread(_write)

    async def save_talk_summary(self, talk_id: str, summary: str) -> None:
        def _write() -> None:
            with self._engine.begin() as conn:
                conn.execute(
                    text(
                        "UPDATE rag_talks SET summary = :summary, updated_at = now() WHERE talk_id = :talk_id"
                    ),
                    {"talk_id": talk_id, "summary": summary},
                )

        await asyncio.to_thread(_write)

    async def save_compressed_summary(
        self, talk_id: str, summary: str, up_to_turn_index: int
    ) -> None:
        def _write() -> None:
            with self._engine.begin() as conn:
                conn.execute(
                    text(
                        """
                        UPDATE rag_talks
                        SET compressed_summary = :summary,
                            compressed_up_to_turn_index = :up_to_turn_index,
                            updated_at = now()
                        WHERE talk_id = :talk_id
                        """
                    ),
                    {
                        "talk_id": talk_id,
                        "summary": summary,
                        "up_to_turn_index": up_to_turn_index,
                    },
                )

        await asyncio.to_thread(_write)

    async def search_talks(
        self,
        user_id: str,
        *,
        q: str | None = None,
        paragraph_id: str | None = None,
        pinned_only: bool = True,
        limit: int = 30,
    ) -> list[dict[str, Any]]:
        def _read() -> list[dict[str, Any]]:
            conditions = ["user_id = :user_id"]
            params: dict[str, Any] = {"user_id": user_id, "limit": max(1, min(limit, 100))}

            if paragraph_id:
                conditions.append("kontext_paragraph_id = :paragraph_id")
                params["paragraph_id"] = paragraph_id
            else:
                conditions.append("kontext_paragraph_id IS NULL")
                if pinned_only:
                    conditions.append("pinned = true")

            if q:
                conditions.append("(title ILIKE :q OR summary ILIKE :q)")
                params["q"] = f"%{q}%"

            where = " AND ".join(conditions)
            with self._engine.connect() as conn:
                rows = conn.execute(
                    text(
                        f"""
                        SELECT talk_id::text, title, pinned, mode, updated_at, summary,
                               kontext_paragraph_id, kontext_source_id, kontext_paragraph
                        FROM rag_talks
                        WHERE {where}
                        ORDER BY pinned DESC, updated_at DESC
                        LIMIT :limit
                        """
                    ),
                    params,
                ).mappings().all()
            return [dict(r) for r in rows]

        return await asyncio.to_thread(_read)

    async def save_turn_references(
        self, turn_id: str, references: Sequence[dict[str, Any]]
    ) -> None:
        if not references:
            return

        def _write() -> None:
            chunk_index_map: list[dict[str, Any]] = []
            with self._engine.begin() as conn:
                for idx, ref in enumerate(references):
                    # Zitat-[N] im Antworttext ist 1-basiert; ref.get("index") bevorzugt.
                    raw_index = ref.get("index")
                    ref_index = (
                        int(raw_index)
                        if isinstance(raw_index, (int, float)) and int(raw_index) >= 1
                        else idx + 1
                    )
                    chunk_id = ref.get("chunk_id")
                    conn.execute(
                        text(
                            """
                            INSERT INTO rag_references
                              (turn_id, ref_index, chunk_id, relevance, source_title, segment_title)
                            VALUES
                              (:turn_id, :ref_index, :chunk_id, :relevance, :source_title, :segment_title)
                            """
                        ),
                        {
                            "turn_id": turn_id,
                            "ref_index": ref_index,
                            "chunk_id": chunk_id,
                            "relevance": ref.get("relevance") if ref.get("relevance") is not None else ref.get("score"),
                            "source_title": ref.get("source_title"),
                            "segment_title": ref.get("segment_title"),
                        },
                    )
                    # Volle Citation-Payload für Sync → KI-Treffer-Karten + Navigation
                    entry: dict[str, Any] = {
                        "index": ref_index,
                        "chunk_id": chunk_id,
                    }
                    for key in (
                        "text",
                        "source_id",
                        "source_title",
                        "segment_title",
                        "segment_index",
                        "lecture_date",
                        "chunk_type",
                        "source_type",
                        "author",
                        "book_title",
                        "venue",
                        "vortragstitel",
                        "paragraph_id",
                        "parent_id",
                        "score",
                    ):
                        val = ref.get(key)
                        if val is not None and val != "":
                            entry[key] = val
                    chunk_index_map.append(entry)

                conn.execute(
                    text(
                        """
                        UPDATE rag_turns
                        SET chunk_index_map = CAST(:chunk_index_map AS jsonb),
                            updated_at = now()
                        WHERE turn_id = :turn_id
                        """
                    ),
                    {
                        "turn_id": turn_id,
                        "chunk_index_map": json.dumps(chunk_index_map),
                    },
                )

        await asyncio.to_thread(_write)
