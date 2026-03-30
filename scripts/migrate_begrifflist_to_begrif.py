"""Migrate Qdrant payload: rename chunk_type 'begrifflist' to 'begriff'.

Run from the ragrun project root:
    python scripts/migrate_begrifflist_to_begrif.py [--dry-run] [--collections col1 col2 ...]

Requires QDRANT_URL and optionally QDRANT_API_KEY environment variables,
or falls back to app.config settings when run inside the project.
"""
from __future__ import annotations

import asyncio
import argparse
import os
import sys

# Allow running from project root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings
from app.infra.qdrant_client import QdrantClient

OLD_CHUNK_TYPE = "begriff_list"
NEW_CHUNK_TYPE = "begriff"
DEFAULT_COLLECTIONS = ["philo-von-freisinn"]


async def migrate_collection(
    client: QdrantClient,
    collection: str,
    *,
    dry_run: bool = False,
) -> int:
    """Scroll all points with chunk_type == OLD_CHUNK_TYPE and update them.

    Returns the number of points updated (or that would be updated in dry-run mode).
    """
    filter_ = {
        "must": [
            {
                "key": "chunk_type",
                "match": {"value": OLD_CHUNK_TYPE},
            }
        ]
    }

    points = await client.scroll_all_points(
        collection,
        filter_=filter_,
        with_payload=True,
        with_vectors=False,
    )

    if not points:
        print(f"  {collection}: no points with chunk_type='{OLD_CHUNK_TYPE}' found")
        return 0

    print(f"  {collection}: found {len(points)} point(s) with chunk_type='{OLD_CHUNK_TYPE}'")

    if dry_run:
        print(f"  {collection}: dry-run – skipping update")
        return len(points)

    updates = [
        {"id": p["id"], "payload": {"chunk_type": NEW_CHUNK_TYPE}}
        for p in points
    ]
    await client.set_payload(collection, updates)
    print(f"  {collection}: updated {len(points)} point(s) -> chunk_type='{NEW_CHUNK_TYPE}'")
    return len(points)


async def main(collections: list[str], *, dry_run: bool) -> None:
    qdrant_url = os.environ.get("QDRANT_URL", str(settings.qdrant_url))
    qdrant_api_key = os.environ.get("QDRANT_API_KEY", settings.qdrant_api_key)

    client = QdrantClient(qdrant_url, api_key=qdrant_api_key, timeout=60.0)

    print(f"Connecting to Qdrant at {qdrant_url}")
    print(f"Migrating chunk_type: '{OLD_CHUNK_TYPE}' -> '{NEW_CHUNK_TYPE}'")
    if dry_run:
        print("DRY-RUN mode – no changes will be written")
    print()

    total_updated = 0
    for collection in collections:
        updated = await migrate_collection(client, collection, dry_run=dry_run)
        total_updated += updated

    print()
    action = "would update" if dry_run else "updated"
    print(f"Done. Total points {action}: {total_updated}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate Qdrant chunk_type begrifflist -> begriff")
    parser.add_argument(
        "--collections",
        nargs="+",
        default=DEFAULT_COLLECTIONS,
        metavar="COLLECTION",
        help=f"Qdrant collection names to migrate (default: {DEFAULT_COLLECTIONS})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report counts without writing any changes",
    )
    args = parser.parse_args()
    asyncio.run(main(args.collections, dry_run=args.dry_run))
