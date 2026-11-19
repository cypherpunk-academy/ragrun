#!/usr/bin/env python3
"""
Point-by-point copy of one Qdrant collection between two instances.
Use this when snapshot formats differ between Qdrant versions.

Qdrant Cloud is retired. Staging and production both use the Tuxai Qdrant
at qdrant.ragxxx.com. The usual copy path is deploy/tuxai/copy_collection.sh.
This script deletes and recreates the destination collection, so the target
URL has no default.

Usage:
    export QDRANT_SOURCE_URL=http://localhost:6333
    export QDRANT_TARGET_URL=http://127.0.0.1:6335
    # optional, for a Tuxai tunnel target:
    # export QDRANT_TARGET_API_KEY=...
    # export RAGRUN_CF_ACCESS_CLIENT_ID=...
    # export RAGRUN_CF_ACCESS_CLIENT_SECRET=...
    python3 scripts/migrate_qdrant.py philo-von-freisinn
"""
import os
import sys
import time

import requests

BATCH_SIZE = 100

SRC_URL = ""
DST_URL = ""
COLLECTION = ""
SRC_KEY = ""
DST_KEY = ""


def _env(name):
    return (os.environ.get(name) or "").strip()


def _headers(url, api_key):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["api-key"] = api_key
    if url.startswith("https://"):
        client_id = _env("RAGRUN_CF_ACCESS_CLIENT_ID")
        client_secret = _env("RAGRUN_CF_ACCESS_CLIENT_SECRET")
        if client_id and client_secret:
            headers["CF-Access-Client-Id"] = client_id
            headers["CF-Access-Client-Secret"] = client_secret
    return headers


def configure():
    source = (_env("QDRANT_SOURCE_URL") or "http://localhost:6333").rstrip("/")
    target = _env("QDRANT_TARGET_URL").rstrip("/")
    if not target:
        print("ERROR: QDRANT_TARGET_URL is required", file=sys.stderr)
        sys.exit(1)
    for label, url in (("source", source), ("target", target)):
        if "cloud.qdrant.io" in url:
            print(f"ERROR: {label} still points at Qdrant Cloud: {url}", file=sys.stderr)
            sys.exit(1)
    if len(sys.argv) != 2 or not sys.argv[1].strip():
        print("Usage: python3 scripts/migrate_qdrant.py COLLECTION", file=sys.stderr)
        sys.exit(1)
    source_key = _env("QDRANT_SOURCE_API_KEY")
    target_key = _env("QDRANT_TARGET_API_KEY") or _env("TUXAI_QDRANT_API_KEY")
    return source, target, sys.argv[1].strip(), source_key, target_key


def src_headers():
    return _headers(SRC_URL, SRC_KEY)


def dst_headers():
    return _headers(DST_URL, DST_KEY)


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
    global SRC_URL, DST_URL, COLLECTION, SRC_KEY, DST_KEY
    SRC_URL, DST_URL, COLLECTION, SRC_KEY, DST_KEY = configure()

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
