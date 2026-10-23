"""Check rag_paragraphs numbering and segment_index/segment_slug alignment (MCP plan E2).

Run from ragrun project root:
    python scripts/check_rag_paragraphs_consistency.py
    python scripts/check_rag_paragraphs_consistency.py --json

Env: same Postgres DSN as the app (RAGRUN_POSTGRES_DSN / settings).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text

from app.core.providers import get_sync_engine


def run_checks() -> dict:
    engine = get_sync_engine()
    report: dict = {
        "gap_issues": [],
        "duplicate_number_issues": [],
        "slug_index_conflicts": [],
        "index_slug_mismatch": [],
        "summary": {},
    }

    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT source_id, segment_index, segment_slug, paragraph_number
                FROM rag_paragraphs
                WHERE deprecated_at IS NULL
                ORDER BY source_id, segment_index, paragraph_number
                """
            )
        ).mappings().all()

        by_segment: dict[tuple[str, int], list[int]] = {}
        index_to_slug: dict[tuple[str, int], set[str]] = {}
        slug_to_index: dict[tuple[str, str], set[int]] = {}

        for row in rows:
            sid = str(row["source_id"])
            seg_i = int(row["segment_index"])
            slug = row["segment_slug"]
            num = int(row["paragraph_number"])
            key = (sid, seg_i)
            by_segment.setdefault(key, []).append(num)
            if slug:
                index_to_slug.setdefault(key, set()).add(str(slug))
                skey = (sid, str(slug))
                slug_to_index.setdefault(skey, set()).add(seg_i)

        for (sid, seg_i), nums in by_segment.items():
            nums_sorted = sorted(nums)
            if len(nums_sorted) != len(set(nums_sorted)):
                report["duplicate_number_issues"].append(
                    {"source_id": sid, "segment_index": seg_i, "paragraph_numbers": nums_sorted}
                )
            if not nums_sorted:
                continue
            expected = list(range(1, nums_sorted[-1] + 1))
            if nums_sorted != expected:
                missing = [n for n in expected if n not in nums_sorted]
                report["gap_issues"].append(
                    {
                        "source_id": sid,
                        "segment_index": seg_i,
                        "missing_paragraph_numbers": missing,
                        "max_seen": nums_sorted[-1],
                        "count": len(nums_sorted),
                    }
                )

        for key, slugs in index_to_slug.items():
            if len(slugs) > 1:
                sid, seg_i = key
                report["index_slug_mismatch"].append(
                    {"source_id": sid, "segment_index": seg_i, "segment_slugs": sorted(slugs)}
                )

        for (sid, slug), indices in slug_to_index.items():
            if len(indices) > 1:
                report["slug_index_conflicts"].append(
                    {"source_id": sid, "segment_slug": slug, "segment_indices": sorted(indices)}
                )

    report["summary"] = {
        "segments_checked": len(by_segment),
        "paragraph_rows": len(rows),
        "gap_issue_count": len(report["gap_issues"]),
        "duplicate_number_count": len(report["duplicate_number_issues"]),
        "index_slug_mismatch_count": len(report["index_slug_mismatch"]),
        "slug_index_conflict_count": len(report["slug_index_conflicts"]),
        "ok": not any(
            report[k]
            for k in (
                "gap_issues",
                "duplicate_number_issues",
                "slug_index_conflicts",
                "index_slug_mismatch",
            )
        ),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="rag_paragraphs consistency (E2)")
    parser.add_argument("--json", action="store_true", help="print JSON report")
    args = parser.parse_args()
    report = run_checks()
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        s = report["summary"]
        print(f"segments={s['segments_checked']} rows={s['paragraph_rows']} ok={s['ok']}")
        for label, key in (
            ("gaps", "gap_issues"),
            ("duplicates", "duplicate_number_issues"),
            ("index→multi-slug", "index_slug_mismatch"),
            ("slug→multi-index", "slug_index_conflicts"),
        ):
            items = report[key]
            if items:
                print(f"\n{label}: {len(items)}")
                for item in items[:20]:
                    print(f"  {item}")
                if len(items) > 20:
                    print(f"  … and {len(items) - 20} more")
    return 0 if report["summary"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
