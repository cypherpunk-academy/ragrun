#!/usr/bin/env python3
"""Compare HF serverless embeddings vs local personal-embeddings-service.

Usage (from ragrun root, venv active):

    export HF_TOKEN=hf_...
    export RAGRUN_EMBEDDINGS_BASE_URL=http://localhost:8001   # optional local
    python scripts/compare_hf_embeddings.py

Exit 0 when cosine similarity >= --min-cosine (default 0.98) and dim == 1024.
Without a reachable local service, only HF shape/dim checks run (exit 0 with warning).
"""
from __future__ import annotations

import argparse
import asyncio
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx

from app.infra.hf_embedding import (
    DEFAULT_HF_MODEL,
    HuggingFaceEmbeddingBackend,
)


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (na * nb)


async def _local_embed(base_url: str, texts: list[str], model: str) -> list[list[float]]:
    url = f"{base_url.rstrip('/')}/api/v1/embeddings"
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=5.0)) as client:
        response = await client.post(url, json={"texts": texts, "model": model})
        response.raise_for_status()
        data = response.json()
        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list):
            raise RuntimeError("local embeddings malformed")
        return embeddings


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=os.environ.get("RAGRUN_EMBEDDINGS_HF_MODEL", DEFAULT_HF_MODEL),
    )
    parser.add_argument(
        "--local-url",
        default=os.environ.get("RAGRUN_EMBEDDINGS_BASE_URL", "http://localhost:8001"),
    )
    parser.add_argument("--min-cosine", type=float, default=0.98)
    parser.add_argument(
        "--text",
        action="append",
        dest="texts",
        help="Text to embed (repeatable). Default: two e5-prefixed samples.",
    )
    args = parser.parse_args()

    token = (
        os.environ.get("RAGRUN_HF_TOKEN")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or ""
    )
    if not token:
        print("ERROR: set HF_TOKEN or RAGRUN_HF_TOKEN", file=sys.stderr)
        return 2

    texts = args.texts or [
        "query: Was ist Freiheit bei Steiner?",
        "passage: Die Freiheit des Menschen liegt im Denken.",
    ]

    hf = HuggingFaceEmbeddingBackend(token=token, model=args.model)
    print(f"HF model={args.model} texts={len(texts)}")
    hf_vecs = await hf.embed_texts(texts)
    for i, v in enumerate(hf_vecs):
        print(f"  HF[{i}] dim={len(v)} l2={math.sqrt(sum(x*x for x in v)):.6f}")

    if any(len(v) != 1024 for v in hf_vecs):
        print("ERROR: expected dimension 1024 for multilingual-e5-large", file=sys.stderr)
        return 1

    try:
        local_vecs = await _local_embed(args.local_url, texts, args.model)
    except Exception as exc:
        print(f"WARNING: local embedder unavailable ({exc}); HF-only checks passed")
        return 0

    if len(local_vecs) != len(hf_vecs):
        print("ERROR: local/HF batch size mismatch", file=sys.stderr)
        return 1

    ok = True
    for i, (hv, lv) in enumerate(zip(hf_vecs, local_vecs)):
        if len(lv) != len(hv):
            print(f"ERROR: dim mismatch at {i}: local={len(lv)} hf={len(hv)}", file=sys.stderr)
            ok = False
            continue
        sim = _cosine(hv, lv)
        print(f"  cosine[{i}]={sim:.6f} (min={args.min_cosine})")
        if sim < args.min_cosine:
            ok = False

    if not ok:
        print("ERROR: compatibility check failed", file=sys.stderr)
        return 1
    print("OK: HF embeddings compatible with local service")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
