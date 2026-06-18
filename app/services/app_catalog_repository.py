"""Postgres-backed catalogue reads for /app/*."""
from __future__ import annotations

import asyncio
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.db.ports import CatalogPort


class PostgresCatalogRepository(CatalogPort):
    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    async def list_sources(self) -> list[dict[str, Any]]:
        def _read() -> list[dict[str, Any]]:
            with self._engine.connect() as conn:
                rows = conn.execute(
                    text(
                        """
                        SELECT id::text AS source_id, title, author, language,
                               COALESCE(is_primary, false) AS is_primary, sort_order
                        FROM rag_sources
                        ORDER BY sort_order ASC, title ASC
                        """
                    )
                ).mappings().all()
            out: list[dict[str, Any]] = []
            for row in rows:
                title = str(row["title"] or "")
                author = str(row["author"] or "")
                display = f"{author}: {title}" if author and title else title or author
                out.append(
                    {
                        "source_id": row["source_id"],
                        "display_name": display,
                        "source_type": "book",
                    }
                )
            return out

        return await asyncio.to_thread(_read)

    async def list_segments(self, source_id: str) -> list[dict[str, Any]]:
        sid = source_id.strip()
        if not sid:
            return []

        def _read() -> list[dict[str, Any]]:
            with self._engine.connect() as conn:
                rows = conn.execute(
                    text(
                        """
                        SELECT segment_index, segment_title, COUNT(*) AS paragraph_count
                        FROM rag_paragraphs
                        WHERE source_id = :source_id
                          AND deprecated_at IS NULL
                        GROUP BY segment_index, segment_title
                        ORDER BY segment_index ASC
                        """
                    ),
                    {"source_id": sid},
                ).mappings().all()
            segments: list[dict[str, Any]] = []
            for row in rows:
                idx = int(row["segment_index"])
                segments.append(
                    {
                        "segment_id": str(idx),
                        "segment_index": idx,
                        "title": str(row["segment_title"] or ""),
                    }
                )
            return segments

        return await asyncio.to_thread(_read)

    async def get_chunk_text(self, chunk_id: str, *, source_id: str | None = None) -> dict[str, Any] | None:
        cid = chunk_id.strip()
        if not cid:
            return None
        partition_cid = cid.split(":", 1)
        bare_id = partition_cid[-1] if partition_cid else cid

        def _read() -> dict[str, Any] | None:
            with self._engine.connect() as conn:
                params: dict[str, Any] = {"chunk_id": bare_id, "full_id": cid}
                source_clause = ""
                if source_id and source_id.strip():
                    params["source_id"] = source_id.strip()
                    source_clause = "AND source_id = :source_id"
                row = conn.execute(
                    text(
                        f"""
                        SELECT chunk_id, source_id, text
                        FROM rag_chunks
                        WHERE deprecated_at IS NULL
                          AND (chunk_id = :chunk_id OR (rag_partition || ':' || chunk_id) = :full_id)
                          {source_clause}
                        ORDER BY updated_at DESC
                        LIMIT 1
                        """
                    ),
                    params,
                ).mappings().first()
            if not row:
                return None
            text_val = str(row["text"] or "")
            return {
                "chunk_id": str(row["chunk_id"]),
                "source_id": str(row["source_id"] or ""),
                "text": text_val,
                "snippet": text_val[:280],
            }

        return await asyncio.to_thread(_read)
