#!/usr/bin/env python3
"""Migration script: backfill BM25 sparse vectors for all existing Qdrant points.

Usage:
    python scripts/add_sparse_vectors.py [--collection NAME] [--batch-size N] [--dry-run]

Run from the ragrun project root with the venv active.

What it does:
1. Lists all collections (or a single named one).
2. For each collection, scrolls all points page by page.
3. Skips points that already have 'text-sparse' vector data.
4. Computes BM25 sparse vectors from each point's 'text' payload field.
5. Calls PUT /collections/{name}/points/vectors to add the sparse vectors.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from typing import List

import httpx

# Make sure the ragrun package is on sys.path when run from project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("add_sparse_vectors")


QDRANT_URL = os.environ.get("RAGRUN_QDRANT_URL", "http://localhost:6333")
VECTOR_NAME = "text-sparse"
DEFAULT_BATCH = 256
DEFAULT_EXCLUDED_COLLECTIONS = {"demo-books"}


async def get_collections(client: httpx.AsyncClient) -> list[str]:
    resp = await client.get(f"{QDRANT_URL}/collections")
    resp.raise_for_status()
    return [c["name"] for c in resp.json()["result"]["collections"]]


async def has_sparse_config(client: httpx.AsyncClient, collection: str) -> bool:
    resp = await client.get(f"{QDRANT_URL}/collections/{collection}")
    resp.raise_for_status()
    result = resp.json().get("result", {}) or {}
    params = result.get("config", {}).get("params", {}) if isinstance(result, dict) else {}
    sparse_vectors = params.get("sparse_vectors", {}) if isinstance(params, dict) else {}
    return isinstance(sparse_vectors, dict) and VECTOR_NAME in sparse_vectors


async def scroll_page(
    client: httpx.AsyncClient,
    collection: str,
    *,
    offset: object | None,
    limit: int,
) -> tuple[list[dict], object | None]:
    body: dict = {
        "limit": limit,
        "with_payload": True,
        "with_vector": [VECTOR_NAME],  # only fetch the sparse slot to detect existing data
    }
    if offset is not None:
        body["offset"] = offset

    resp = await client.post(
        f"{QDRANT_URL}/collections/{collection}/points/scroll",
        json=body,
        timeout=60.0,
    )
    resp.raise_for_status()
    result = resp.json()["result"]
    return result.get("points", []), result.get("next_page_offset")


async def update_vectors_batch(
    client: httpx.AsyncClient,
    collection: str,
    points: list[dict],
    *,
    dry_run: bool,
) -> int:
    if not points or dry_run:
        return len(points)

    body = {"points": points}
    resp = await client.put(
        f"{QDRANT_URL}/collections/{collection}/points/vectors?wait=true",
        json=body,
        timeout=120.0,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"update_vectors failed ({resp.status_code}): {resp.text}")
    return len(points)


async def migrate_collection(
    client: httpx.AsyncClient,
    embedder,
    collection: str,
    *,
    batch_size: int,
    dry_run: bool,
) -> None:
    if not await has_sparse_config(client, collection):
        logger.warning(
            "[%s] missing sparse slot '%s'; cannot backfill on existing collection in Qdrant 1.11. "
            "Recreate collection with sparse_vectors at creation time.",
            collection,
            VECTOR_NAME,
        )
        return

    total_processed = 0
    total_skipped = 0
    total_updated = 0
    offset: object | None = None
    page = 0

    while True:
        page += 1
        points, next_offset = await scroll_page(
            client, collection, offset=offset, limit=batch_size
        )

        if not points:
            break

        # Collect texts for points that don't already have sparse vectors
        to_embed: list[tuple[str, str]] = []  # (point_id, text)
        for pt in points:
            # If the point already has the sparse vector, skip it
            vectors = pt.get("vector") or {}
            if isinstance(vectors, dict) and VECTOR_NAME in vectors:
                total_skipped += 1
                continue

            payload = pt.get("payload") or {}
            text = payload.get("text") or ""
            if not isinstance(text, str) or not text.strip():
                total_skipped += 1
                continue

            to_embed.append((pt["id"], text))

        if to_embed:
            texts = [t for _, t in to_embed]
            sparse_vecs = embedder.embed_batch(texts)

            update_points = []
            for (pid, _), sv in zip(to_embed, sparse_vecs):
                update_points.append({
                    "id": pid,
                    "vector": {VECTOR_NAME: sv},
                })

            n = await update_vectors_batch(client, collection, update_points, dry_run=dry_run)
            total_updated += n

        total_processed += len(points)
        logger.info(
            "[%s] page %d: processed=%d updated=%d skipped=%d (offset=%s)",
            collection, page, total_processed, total_updated, total_skipped,
            "done" if next_offset is None else "…",
        )

        if next_offset is None:
            break
        offset = next_offset

    logger.info(
        "[%s] DONE — processed=%d updated=%d skipped=%d%s",
        collection, total_processed, total_updated, total_skipped,
        " (dry-run, no writes)" if dry_run else "",
    )


async def main(args: argparse.Namespace) -> None:
    # Import here so the script only needs the ragrun venv
    from app.infra.sparse_embedder import SparseEmbedder

    logger.info("Loading BM25 model (first run downloads ~5 MB)…")
    embedder = SparseEmbedder()
    # Trigger model load once before the async loop
    _ = embedder.embed_query("warmup")
    logger.info("BM25 model ready")

    async with httpx.AsyncClient(timeout=30.0) as client:
        if args.collection:
            collections = [args.collection]
        else:
            collections = await get_collections(client)
            logger.info("Collections: %s", collections)
            collections = [c for c in collections if c not in DEFAULT_EXCLUDED_COLLECTIONS]
            logger.info("After exclusions: %s", collections)

        for coll in collections:
            logger.info("=== Migrating collection: %s ===", coll)
            try:
                await migrate_collection(
                    client,
                    embedder,
                    coll,
                    batch_size=args.batch_size,
                    dry_run=args.dry_run,
                )
            except Exception:
                logger.exception("Failed to migrate collection %s", coll)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill BM25 sparse vectors into Qdrant")
    parser.add_argument("--collection", help="Migrate only this collection (default: all)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH, metavar="N")
    parser.add_argument("--dry-run", action="store_true", help="Scroll and embed but don't write")
    args = parser.parse_args()

    asyncio.run(main(args))
