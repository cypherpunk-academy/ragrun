#!/usr/bin/env python3
"""Idempotent text index on rudolf-steiner-ga payload field ``vortragstitel``.

The GA title path reads ``rag_lecture_catalog.vortragstitel`` and does not need
this index. Create it when the catalog lookup is unavailable and title matches
should fall back to a Qdrant text filter (tokenizer ``word``, lowercase).

  .venv/bin/python scripts/ensure_ga_vortragstitel_index.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

COLLECTION = "rudolf-steiner-ga"
FIELD = "vortragstitel"


async def main() -> None:
    from app.core.providers import get_qdrant_client

    client = get_qdrant_client()
    info = await client.get_collection_info(COLLECTION)
    schema = {}
    if isinstance(info, dict):
        raw = info.get("payload_schema")
        if isinstance(raw, dict):
            schema = raw
    existing = schema.get(FIELD)
    print(f"{FIELD} payload schema before: {existing!r}")
    await client.ensure_text_index(
        COLLECTION,
        field_name=FIELD,
        tokenizer="word",
        lowercase=True,
    )
    print(f"text index ensured on {COLLECTION}.{FIELD}")


if __name__ == "__main__":
    asyncio.run(main())
