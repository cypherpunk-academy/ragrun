#!/usr/bin/env python3
"""Compare GA search ranking variants on evaluate/ga_ranking_testkorpus.yaml.

Calls ``_search_ga_collection`` directly (no MCP). Needs RAGRUN_QDRANT_URL and
RAGRUN_QDRANT_API_KEY, normally via ``.env`` in the repo root.

Variants:
  baseline                  previous hybrid_retrieve, k = limit
  +kandidaten+gruppierung   deep candidates, fusion, grouping, volume cap
  +titel                    plus title signal and catalog title path
  +rerank                   plus cross-encoder (fail-open if the service is down)

Writes a JSON report under evaluate/results/ and prints a summary table.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os

os.chdir(ROOT)

import yaml  # noqa: E402

VARIANTS = (
    ("baseline", "baseline"),
    ("candidates", "+kandidaten+gruppierung"),
    ("title", "+titel"),
    ("rerank", "+rerank"),
)


def _year(value: object) -> int | None:
    text = str(value or "").strip()
    if len(text) >= 4 and text[:4].isdigit():
        year = int(text[:4])
        if 1800 <= year <= 2100:
            return year
    parts = text.split(".")
    if len(parts) == 3 and parts[-1].isdigit():
        year = int(parts[-1])
        if 1800 <= year <= 2100:
            return year
    return None


def _ga_of(hit: dict[str, Any]) -> str:
    return str(hit.get("ga") or "").strip()


def _match_index(
    hit: dict[str, Any],
    expected: list[dict[str, Any]],
    remaining: set[int],
) -> int | None:
    lecture_id = str(hit.get("lecture_id") or "").strip()
    ga = _ga_of(hit)
    for idx in sorted(remaining):
        item = expected[idx]
        want = str(item.get("lecture_id") or "").strip()
        if want and want == lecture_id:
            return idx
    for idx in sorted(remaining):
        item = expected[idx]
        if str(item.get("lecture_id") or "").strip():
            continue
        want_ga = str(item.get("ga") or "").strip()
        if want_ga and want_ga == ga:
            return idx
    return None


def _grades(
    hits: list[dict[str, Any]],
    expected: list[dict[str, Any]],
    k: int,
) -> tuple[list[int], list[float]]:
    remaining = set(range(len(expected)))
    found: list[int] = []
    grades: list[float] = []
    for hit in hits[:k]:
        idx = _match_index(hit, expected, remaining)
        if idx is None:
            grades.append(0.0)
            continue
        remaining.remove(idx)
        found.append(idx)
        grades.append(float(expected[idx].get("relevance") or 0))
    return found, grades


def _dcg(grades: list[float], k: int) -> float:
    total = 0.0
    for i, grade in enumerate(grades[:k]):
        total += (2**grade - 1) / math.log2(i + 2)
    return total


def _ndcg(grades: list[float], ideal: list[float], k: int) -> float:
    idcg = _dcg(sorted(ideal, reverse=True), k)
    if idcg <= 0:
        return 0.0
    return _dcg(grades, k) / idcg


def _recall(found: list[int], expected: list[dict[str, Any]], relevance: set[int]) -> float | None:
    wanted = [i for i, item in enumerate(expected) if int(item.get("relevance") or 0) in relevance]
    if not wanted:
        return None
    hit = sum(1 for i in found if i in set(wanted))
    return hit / len(wanted)


def _mrr(hits: list[dict[str, Any]], expected: list[dict[str, Any]]) -> float | None:
    haupt = {i for i, item in enumerate(expected) if int(item.get("relevance") or 0) == 2}
    if not haupt:
        return None
    remaining = set(range(len(expected)))
    for rank, hit in enumerate(hits, start=1):
        idx = _match_index(hit, expected, remaining)
        if idx is None:
            continue
        remaining.remove(idx)
        if idx in haupt:
            return 1.0 / rank
    return 0.0


def _spread(hits: list[dict[str, Any]]) -> dict[str, float]:
    lectures: set[str] = set()
    volumes: set[str] = set()
    years: list[int] = []
    for hit in hits[:10]:
        key = str(hit.get("lecture_id") or hit.get("segment_id") or hit.get("chunk_id") or "").strip()
        if key:
            lectures.add(key)
        ga = _ga_of(hit)
        if ga:
            volumes.add(ga)
        year = _year(hit.get("lecture_date"))
        if year is not None:
            years.append(year)
    span = float(max(years) - min(years)) if len(years) >= 2 else 0.0
    return {"lectures": float(len(lectures)), "volumes": float(len(volumes)), "year_span": span}


def _accept(hits: list[dict[str, Any]], rule: dict[str, Any] | None) -> dict[str, bool] | None:
    if not rule:
        return None
    out: dict[str, bool] = {}
    volumes = rule.get("any_ga_in_top5") or []
    if volumes:
        wanted = {str(v) for v in volumes}
        top = hits[:5]
        out["any_ga_in_top5"] = any(
            part.strip() in wanted
            for hit in top
            for part in _ga_of(hit).split(",")
            if part.strip()
        )
    lecture = str(rule.get("lecture_in_top10") or "").strip()
    if lecture:
        out["lecture_in_top10"] = any(str(hit.get("lecture_id") or "") == lecture for hit in hits[:10])
    return out


def _case_metrics(hits: list[dict[str, Any]], case: dict[str, Any]) -> dict[str, Any]:
    expected = list(case.get("expected") or [])
    found10, grades10 = _grades(hits, expected, 10)
    found5, _grades5 = _grades(hits, expected, 5)
    ideal = [float(item.get("relevance") or 0) for item in expected]
    negative = not expected
    metrics: dict[str, Any] = {
        "negative": negative,
        "recall_at_5_haupt": None if negative else _recall(found5, expected, {2}),
        "recall_at_10_haupt": None if negative else _recall(found10, expected, {2}),
        "recall_at_5_all": None if negative else _recall(found5, expected, {1, 2}),
        "recall_at_10_all": None if negative else _recall(found10, expected, {1, 2}),
        "mrr_haupt": None if negative else _mrr(hits, expected),
        "ndcg_at_10": None if negative else _ndcg(grades10, ideal, 10),
        "expected_in_top10": 0 if negative else len(found10),
        **_spread(hits),
        "accept": _accept(hits, case.get("accept")),
        "top_score": round(float(hits[0]["score"]), 4) if hits else None,
        "hits": [
            {
                "ga": hit.get("ga"),
                "lecture_id": hit.get("lecture_id"),
                "segment_id": hit.get("segment_id"),
                "vortragstitel": hit.get("vortragstitel"),
                "lecture_date": hit.get("lecture_date"),
                "score": hit.get("score"),
                "lecture_hits": hit.get("lecture_hits"),
            }
            for hit in hits[:10]
        ],
    }
    return metrics


def _mean(values: list[float | None]) -> float | None:
    nums = [v for v in values if v is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def _latency(samples_s: list[float]) -> dict[str, float]:
    if not samples_s:
        return {"median_ms": 0.0, "p95_ms": 0.0}
    ordered = sorted(samples_s)
    p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
    return {
        "median_ms": round(statistics.median(ordered) * 1000, 1),
        "p95_ms": round(p95 * 1000, 1),
    }


def _summarize(rows: list[dict[str, Any]], samples_s: list[float]) -> dict[str, Any]:
    keys = [
        "recall_at_5_haupt",
        "recall_at_10_haupt",
        "recall_at_5_all",
        "recall_at_10_all",
        "mrr_haupt",
        "ndcg_at_10",
        "lectures",
        "volumes",
        "year_span",
    ]
    summary = {key: _mean([row.get(key) for row in rows]) for key in keys}
    summary.update(_latency(samples_s))
    summary["cases"] = len(rows)
    return summary


async def _run_case(case: dict[str, Any], variant: str) -> tuple[dict[str, Any], float]:
    from app.mcp_server.tools import _search_ga_collection

    started = time.perf_counter()
    result = await _search_ga_collection(
        query=str(case["query"]),
        chunk_types=list(case.get("chunk_types") or ["ga_text"]),
        ga=case.get("ga"),
        lecture_id=case.get("lecture_id"),
        limit=int(case.get("limit") or 10),
        full_text=False,
        variant=variant,
    )
    elapsed = time.perf_counter() - started
    if isinstance(result, dict) and result.get("error"):
        return {
            "id": case.get("id"),
            "query": case.get("query"),
            "error": result,
            "negative": not list(case.get("expected") or []),
        }, elapsed
    hits = result if isinstance(result, list) else []
    metrics = _case_metrics(hits, case)
    metrics["id"] = case.get("id")
    metrics["query"] = case.get("query")
    metrics["latency_ms"] = round(elapsed * 1000, 1)
    return metrics, elapsed


def _fmt(value: object) -> str:
    if value is None:
        return "   -  "
    if isinstance(value, float):
        return f"{value:6.3f}"
    return str(value)


def _print_table(report: dict[str, Any]) -> None:
    headers = (
        "R@5h",
        "R@10h",
        "R@10",
        "MRR",
        "nDCG",
        "lect",
        "vols",
        "years",
        "med_ms",
        "p95_ms",
    )
    print(f"{'variant':<28}" + " ".join(f"{h:>8}" for h in headers))
    for block in report["variants"]:
        summary = block["summary"]
        cells = [
            _fmt(summary.get("recall_at_5_haupt")),
            _fmt(summary.get("recall_at_10_haupt")),
            _fmt(summary.get("recall_at_10_all")),
            _fmt(summary.get("mrr_haupt")),
            _fmt(summary.get("ndcg_at_10")),
            _fmt(summary.get("lectures")),
            _fmt(summary.get("volumes")),
            _fmt(summary.get("year_span")),
            _fmt(summary.get("median_ms")),
            _fmt(summary.get("p95_ms")),
        ]
        print(f"{block['label']:<28}" + " ".join(f"{cell:>8}" for cell in cells))


def _print_acceptance(report: dict[str, Any]) -> None:
    by_label = {block["label"]: block for block in report["variants"]}
    baseline = by_label.get("baseline")
    print("\nAcceptance against baseline")
    for label in ("+kandidaten+gruppierung", "+titel", "+rerank"):
        block = by_label.get(label)
        if not block or not baseline:
            continue
        print(f"\n{label}")
        base_cases = {row["id"]: row for row in baseline["cases"]}
        for row in block["cases"]:
            if row.get("error"):
                print(f"  {row.get('id')}: ERROR {row['error']}")
                continue
            if row.get("negative"):
                print(f"  {row.get('id')}: negative top_score={row.get('top_score')}")
                continue
            before = int(base_cases.get(row["id"], {}).get("expected_in_top10") or 0)
            after = int(row.get("expected_in_top10") or 0)
            delta = after - before
            flag = "OK" if delta >= -1 else "LOSS"
            accept = row.get("accept")
            extra = f" accept={accept}" if accept else ""
            print(f"  {row.get('id')}: expected@10 {before} -> {after} ({delta:+d}) {flag}{extra}")


async def evaluate(cases: list[dict[str, Any]], variant_ids: list[str]) -> dict[str, Any]:
    from urllib.parse import urlparse

    from app.config import settings

    selected = [pair for pair in VARIANTS if pair[0] in variant_ids]
    # One uncounted call so the first measured query is not a cold start.
    if cases:
        from app.mcp_server.tools import _search_ga_collection

        try:
            await _search_ga_collection(
                query=str(cases[0]["query"]),
                chunk_types=list(cases[0].get("chunk_types") or ["ga_text"]),
                ga=None,
                lecture_id=None,
                limit=1,
                full_text=False,
                variant="baseline",
            )
        except Exception as exc:
            print(f"warmup failed: {exc}", flush=True)

    blocks: list[dict[str, Any]] = []
    for variant_id, label in selected:
        rows: list[dict[str, Any]] = []
        samples: list[float] = []
        for case in cases:
            metrics, elapsed = await _run_case(case, variant_id)
            rows.append(metrics)
            if not metrics.get("error"):
                samples.append(elapsed)
            print(f"{label} {case.get('id')} {round(elapsed * 1000)} ms", flush=True)
        blocks.append(
            {
                "variant": variant_id,
                "label": label,
                "summary": _summarize(
                    [row for row in rows if not row.get("error") and not row.get("negative")],
                    samples,
                ),
                "cases": rows,
            }
        )
        # Spread and latency include negative cases; recall above does not.
        all_rows = [row for row in rows if not row.get("error")]
        spread = _summarize(all_rows, samples)
        blocks[-1]["summary"]["lectures"] = spread["lectures"]
        blocks[-1]["summary"]["volumes"] = spread["volumes"]
        blocks[-1]["summary"]["year_span"] = spread["year_span"]
        blocks[-1]["summary"]["median_ms"] = spread["median_ms"]
        blocks[-1]["summary"]["p95_ms"] = spread["p95_ms"]

    return {
        "metadata": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "qdrant": urlparse(str(settings.qdrant_url)).netloc,
            "rerank": bool(settings.ga_search_rerank),
            "rerank_url_set": bool(settings.ga_search_rerank_url),
            "cases": len(cases),
        },
        "variants": blocks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GA ranking variants")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=ROOT / "evaluate" / "ga_ranking_testkorpus.yaml",
    )
    parser.add_argument(
        "--variants",
        default="baseline,candidates,title,rerank",
        help="Comma-separated: baseline, candidates, title, rerank",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    document = yaml.safe_load(args.corpus.read_text(encoding="utf-8"))
    cases = list(document.get("cases") or [])
    variant_ids = [part.strip() for part in args.variants.split(",") if part.strip()]
    report = asyncio.run(evaluate(cases, variant_ids))

    results = ROOT / "evaluate" / "results"
    results.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%dT%H%M%S")
    output = args.output or (results / f"{stamp}-ga-ranking.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    _print_table(report)
    _print_acceptance(report)
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
