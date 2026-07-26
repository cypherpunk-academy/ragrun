"""Logical partitions for the rag_chunks primary store (Postgres).

Qdrant collection names stay on vector_chunks / Qdrant only. rag_partition answers:
\"In which logical partition of rag_chunks should this row live?\"
"""
from __future__ import annotations

# Reserved partition for shared corpus rows (book + secondary_book body text).
RAG_PARTITION_SHARED = "__shared__"

# chunk_type values stored under RAG_PARTITION_SHARED (deterministic store rule).
CHUNK_TYPES_RAG_SHARED = frozenset({"book", "secondary_book"})

__all__ = ["RAG_PARTITION_SHARED", "CHUNK_TYPES_RAG_SHARED"]
