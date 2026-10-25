"""Shared corpus search rules (MCP plan §3, E7, E10)."""
from __future__ import annotations

from typing import Any

ORIGINAL_HIT_CHUNK_TYPES = frozenset({"book", "secondary_book", "quote", "curated_text"})
AUGMENT_CHUNK_TYPES = frozenset({"begriff", "typology", "chapter_summary"})

_CHUNK_RANK = {"book": 3, "secondary_book": 2, "quote": 1, "curated_text": 0}


def chunk_types_for_title_search(resolved_chunk_types: list[str]) -> list[str]:
    """E7: title ILIKE only on original source types present in this search."""
    return [t for t in resolved_chunk_types if t in ORIGINAL_HIT_CHUNK_TYPES]


def apply_quote_hit_policy(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """E10: verified quotes with paragraph_id only; dedupe by paragraph_id."""
    filtered: list[dict[str, Any]] = []
    for r in results:
        ct = str(r.get("chunk_type") or "")
        if ct == "quote":
            if not r.get("quote_verified"):
                continue
            if not r.get("paragraph_id"):
                continue
        filtered.append(r)

    by_para: dict[str, dict[str, Any]] = {}
    without_para: list[dict[str, Any]] = []
    for r in filtered:
        pid = r.get("paragraph_id")
        ct = str(r.get("chunk_type") or "")
        if not pid or ct not in ("book", "secondary_book", "quote"):
            without_para.append(r)
            continue
        key = str(pid)
        prev = by_para.get(key)
        if prev is None or _prefer_hit(r, prev):
            by_para[key] = r

    merged = without_para + list(by_para.values())
    merged.sort(key=lambda x: float(x.get("score", 0)), reverse=True)
    return merged


def _prefer_hit(a: dict[str, Any], b: dict[str, Any]) -> bool:
    rank_a = _CHUNK_RANK.get(str(a.get("chunk_type") or ""), 0)
    rank_b = _CHUNK_RANK.get(str(b.get("chunk_type") or ""), 0)
    if rank_a != rank_b:
        return rank_a > rank_b
    return float(a.get("score", 0)) >= float(b.get("score", 0))


def is_original_hit(chunk_type: str | None) -> bool:
    return str(chunk_type or "") in ORIGINAL_HIT_CHUNK_TYPES


def is_augment_hit(chunk_type: str | None) -> bool:
    return str(chunk_type or "") in AUGMENT_CHUNK_TYPES
