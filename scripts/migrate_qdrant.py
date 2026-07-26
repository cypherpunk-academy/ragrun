#!/usr/bin/env python3
"""
Migrate a Qdrant collection from one instance to another via point-by-point copy.
Avoids snapshot format incompatibilities between Qdrant versions.

Usage:
    python3 scripts/migrate_qdrant.py
"""
import json
import sys
import time

import requests

SRC_URL = "http://localhost:6333"
SRC_KEY = None  # local, no auth

DST_URL = "https://d6407d98-9790-4cc1-a97f-f3b21e7ccff8.eu-central-1-0.aws.cloud.qdrant.io:6333"
DST_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6ZTAyNDRlYTEtYzk3My00ZjE3LWJlY2EtYWJjZGJiNmQ1ZTAyIn0.0o9tn4CQvozGsJgaKR48RsrKgd2UJZ6xrK1yvIrykoE"

COLLECTION = "philo-von-freisinn"
BATCH_SIZE = 100


def src_headers():
    return {}


def dst_headers():
    return {"api-key": DST_KEY, "Content-Type": "application/json"}


def get_collection_info():
    r = requests.get(f"{SRC_URL}/collections/{COLLECTION}", headers=src_headers())
    r.raise_for_status()
    return r.json()["result"]


def create_collection(info):
    """Create collection on destination with same config as source."""
    cfg = info["config"]
    params = cfg["params"]

    # Build vectors config
    vectors = params.get("vectors", {})
    sparse_vectors = params.get("sparse_vectors", {})

    # Map on_disk → mmap for v1.18+ compatibility
    def remap_storage(d):
        if isinstance(d, dict):
            return {k: ("mmap" if v == "on_disk" else remap_storage(v))
                    for k, v in d.items()}
        return d

    vectors = remap_storage(vectors)

    body = {
        "vectors": vectors,
    }
    if sparse_vectors:
        body["sparse_vectors"] = remap_storage(sparse_vectors)

    # HNSW config
    hnsw = cfg.get("hnsw_config", {})
    if hnsw:
        body["hnsw_config"] = {k: v for k, v in hnsw.items()
                               if k not in ("full_scan_threshold",)}

    # Check if collection already exists on dst
    r = requests.get(f"{DST_URL}/collections/{COLLECTION}", headers=dst_headers())
    if r.status_code == 200:
        existing = r.json()["result"]
        existing_count = existing.get("points_count", 0)
        print(f"Deleting existing collection ({existing_count} points) and recreating...")
        requests.delete(f"{DST_URL}/collections/{COLLECTION}", headers=dst_headers())
        time.sleep(2)

    print(f"Creating collection on destination...")
    r = requests.put(
        f"{DST_URL}/collections/{COLLECTION}",
        headers=dst_headers(),
        json=body,
    )
    if not r.ok:
        print(f"ERROR creating collection: {r.status_code} {r.text}")
        sys.exit(1)
    print(f"  Created: {r.json()}")


def scroll_points(offset=None):
    body = {
        "limit": BATCH_SIZE,
        "with_payload": True,
        "with_vector": True,
    }
    if offset:
        body["offset"] = offset

    r = requests.post(
        f"{SRC_URL}/collections/{COLLECTION}/points/scroll",
        headers=src_headers(),
        json=body,
    )
    r.raise_for_status()
    result = r.json()["result"]
    return result["points"], result.get("next_page_offset")


def upsert_points(points):
    body = {"points": points}
    for attempt in range(5):
        r = requests.put(
            f"{DST_URL}/collections/{COLLECTION}/points?wait=false",
            headers=dst_headers(),
            json=body,
            timeout=60,
        )
        if r.ok:
            return
        if r.status_code in (429, 502, 503, 504):
            wait = 5 * (attempt + 1)
            print(f"  Retrying ({r.status_code}) in {wait}s...")
            time.sleep(wait)
            continue
        print(f"ERROR upserting batch: {r.status_code} {r.text[:200]}")
        sys.exit(1)
    print(f"ERROR: 5 retries exhausted")
    sys.exit(1)


def main():
    print(f"Source: {SRC_URL}")
    print(f"Destination: {DST_URL}")
    print(f"Collection: {COLLECTION}")
    print()

    info = get_collection_info()
    total = info.get("points_count", "?")
    print(f"Source points: {total}")

    create_collection(info)

    print(f"\nMigrating points (batch size {BATCH_SIZE})...")
    offset = None
    migrated = 0
    start = time.time()

    while True:
        points, next_offset = scroll_points(offset)
        if not points:
            break

        upsert_points(points)
        migrated += len(points)

        elapsed = time.time() - start
        rate = migrated / elapsed if elapsed > 0 else 0
        remaining = (total - migrated) / rate if rate > 0 and isinstance(total, int) else "?"
        print(f"  {migrated}/{total} points  ({rate:.0f} pts/s, ~{remaining:.0f}s remaining)" if isinstance(remaining, float) else f"  {migrated} points")

        if next_offset is None:
            break
        offset = next_offset

    print(f"\nDone. Migrated {migrated} points in {time.time()-start:.1f}s")

    # Verify
    time.sleep(3)
    r = requests.get(f"{DST_URL}/collections/{COLLECTION}", headers=dst_headers())
    dst_count = r.json()["result"].get("points_count", "?")
    print(f"Destination now has: {dst_count} points")
    if dst_count == total:
        print("Migration successful!")
    else:
        print(f"WARNING: count mismatch (src={total}, dst={dst_count}) — indexing may still be in progress")


if __name__ == "__main__":
    main()
