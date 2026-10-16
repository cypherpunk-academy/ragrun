#!/usr/bin/env python3
"""Measure retrieval pipeline latency (embed, Qdrant, hybrid, search_ga path).

Run once per environment (fresh process so Settings pick up env):

  set -a && source .env.dev && set +a
  .venv/bin/python scripts/benchmark_retrieval_latency.py --label dev-local

  set -a && source .env.staging && set +a
  .venv/bin/python scripts/benchmark_retrieval_latency.py --label tuxai-from-mac

Inside Railway (same SOCKS as production MCP):

  railway ssh -e staging -s ragrun-staging
  python scripts/benchmark_retrieval_latency.py --label railway-staging
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

QUERY = "GA 5 Rudolf Steiner Geisteswissenschaft"
COLLECTION = "rudolf-steiner-ga"
CHUNK_SUMMARY = ["chapter_summary"]
CHUNK_GA_TEXT = ["ga_text"]
RUNS = 5


def _ms(seconds: float) -> float:
    return round(seconds * 1000, 1)


async def _timed(name: str, coro_factory: Callable[[], Any], samples: dict[str, list[float]]) -> None:
    t0 = time.perf_counter()
    await coro_factory()
    samples[name].append(time.perf_counter() - t0)


async def _timed_sync(name: str, fn: Callable[[], Any], samples: dict[str, list[float]]) -> None:
    t0 = time.perf_counter()
    fn()
    samples[name].append(time.perf_counter() - t0)


def _summarize(samples: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for key, vals in samples.items():
        if not vals:
            continue
        s = sorted(vals)
        out[key] = {
            "min_ms": _ms(s[0]),
            "median_ms": _ms(statistics.median(s)),
            "p95_ms": _ms(s[min(len(s) - 1, int(len(s) * 0.95))]),
            "max_ms": _ms(s[-1]),
        }
    return out


async def _http_health(label: str, url: str, samples: dict[str, list[float]]) -> None:
    async def once() -> None:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(url)
            r.raise_for_status()

    key = f"health_{label}"
    samples[key] = []
    for _ in range(3):
        await _timed(key, once, samples)


async def run_benchmark(label: str) -> dict[str, Any]:
    from app.config import settings
    from app.core.providers import get_embedding_client, get_qdrant_client, get_sparse_embedder
    from app.retrieval.utils.retrievers import dense_retrieve, hybrid_retrieve, sparse_retrieve

    samples: dict[str, list[float]] = {}

    qdrant_base = str(settings.qdrant_url).rstrip("/")
    embed_base = str(settings.embeddings_base_url).rstrip("/")

    await _http_health("qdrant", f"{qdrant_base}/healthz", samples)
    await _http_health("embeddings", f"{embed_base}/api/v1/health/simple", samples)

    emb = get_embedding_client()
    qdrant = get_qdrant_client()
    sparse = get_sparse_embedder()
    prefix = settings.embedding_prefix_query or "query: "

    # Warm BM25 once (not counted in hybrid runs if we warm before loop)
    samples["bm25_embed_warm"] = []
    await _timed_sync("bm25_embed_warm", lambda: sparse.embed_query(QUERY), samples)

    for i in range(RUNS):
        async def embed_once() -> None:
            await emb.embed_texts([f"{prefix}{QUERY}"])

        samples.setdefault("embed_query", [])
        await _timed("embed_query", embed_once, samples)

        async def dense_once() -> None:
            await dense_retrieve(
                query=QUERY,
                k=8,
                worldview=None,
                book_types=CHUNK_GA_TEXT,
                collection=COLLECTION,
                embedding_client=emb,
                qdrant_client=qdrant,
                query_prefix=prefix,
            )

        samples.setdefault("dense_qdrant", [])
        await _timed("dense_qdrant", dense_once, samples)

        async def sparse_once() -> None:
            await sparse_retrieve(
                query=QUERY,
                k=8,
                worldview=None,
                book_types=CHUNK_GA_TEXT,
                collection=COLLECTION,
                qdrant_client=qdrant,
                sparse_embedder=sparse,
            )

        samples.setdefault("sparse_qdrant", [])
        await _timed("sparse_qdrant", sparse_once, samples)

        async def hybrid_once() -> None:
            await hybrid_retrieve(
                query=QUERY,
                k_dense=8,
                k_sparse=8,
                k_fused=8,
                worldview=None,
                book_types=CHUNK_SUMMARY,
                collection=COLLECTION,
                embedding_client=emb,
                qdrant_client=qdrant,
                sparse_embedder=sparse,
                force_sparse=settings.use_hybrid_retrieval,
                query_prefix=prefix,
            )

        samples.setdefault("hybrid_ga_summaries", [])
        await _timed("hybrid_ga_summaries", hybrid_once, samples)

    # search_ga_summaries-shaped path (includes optional postgres ga check)
    from app.mcp_server.tools import search_ga_summaries

    samples["search_ga_summaries"] = []
    for _ in range(RUNS):
        async def ga_sum_once() -> None:
            await search_ga_summaries(QUERY, ga="5", limit=5)

        await _timed("search_ga_summaries", ga_sum_once, samples)

    grouped: dict[str, list[float]] = {
        "embed_query": samples.get("embed_query", []),
        "dense_qdrant": samples.get("dense_qdrant", []),
        "sparse_qdrant": samples.get("sparse_qdrant", []),
        "hybrid_ga_summaries": samples.get("hybrid_ga_summaries", []),
        "search_ga_summaries": samples.get("search_ga_summaries", []),
    }

    return {
        "label": label,
        "app_env": settings.app_env,
        "qdrant_url": qdrant_base,
        "embeddings_url": embed_base,
        "use_hybrid": settings.use_hybrid_retrieval,
        "health": _summarize({k: v for k, v in samples.items() if k.startswith("health_")}),
        "bm25_warm_ms": _ms(samples["bm25_embed_warm"][0]) if samples.get("bm25_embed_warm") else None,
        "steps": _summarize(grouped),
        "proxy_env": {
            "ALL_PROXY": (__import__("os").environ.get("ALL_PROXY") or "")[:80],
            "NO_PROXY": (__import__("os").environ.get("NO_PROXY") or "")[:120],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrieval latency benchmark")
    parser.add_argument("--label", required=True, help="Run name (dev-local, railway-staging, …)")
    args = parser.parse_args()
    report = asyncio.run(run_benchmark(args.label))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
