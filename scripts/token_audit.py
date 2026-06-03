#!/usr/bin/env python3
"""Token audit for rag_chunks against a transformer model's max sequence length.

Usage:
    python scripts/token_audit.py --collection philo-von-freisinn-v2
    python scripts/token_audit.py --collection philo-von-freisinn-v2 --output report.csv
    python scripts/token_audit.py --collection philo-von-freisinn-v2 --check-completeness
    python scripts/token_audit.py --collection philo-von-freisinn-v2 --limit 512 --warn 400

What it does:
1. Loads only the tokenizer (no model weights) for the configured embedding model.
2. Queries rag_chunks (partition + __shared__) for all active (non-deprecated) chunks.
3. Reports chunks that exceed --limit tokens, grouped by source.
4. Recommends a safe --max-chars value for rag:chunk per offending source.
5. Optionally: shows a completeness table (chunk_type × source) for the collection.

Run from ragrun project root with the venv active:
    RAGRUN_POSTGRES_DSN=postgresql+psycopg://... python scripts/token_audit.py --collection philo-von-freisinn-v2
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from collections import defaultdict
from typing import Iterator

# Ensure ragrun package is importable when run from project root.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

# Auto-load .env from project root so RAGRUN_POSTGRES_DSN is available without
# having to set it manually in the shell.
def _load_dotenv() -> None:
    env_file = os.path.join(_PROJECT_ROOT, ".env")
    if not os.path.isfile(env_file):
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(env_file, override=False)
    except ImportError:
        # Fallback: parse KEY=VALUE lines manually (no dotenv installed)
        with open(env_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip()
                if key and key not in os.environ:
                    os.environ[key] = val

_load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("token_audit")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "intfloat/multilingual-e5-large"
DEFAULT_TOKEN_LIMIT = 512
DEFAULT_WARN_THRESHOLD = 400
FETCH_BATCH_SIZE = 500

# Known chunk types produced by ragprep (excluding typology, which requires Qdrant)
EXPECTED_CHUNK_TYPES_PRIMARY = {"book", "chapter_summary"}
EXPECTED_CHUNK_TYPES_ASSISTANT = {"secondary_book", "talk", "quote", "begriff"}
COMPLETENESS_ALL = EXPECTED_CHUNK_TYPES_PRIMARY | EXPECTED_CHUNK_TYPES_ASSISTANT

RAG_PARTITION_SHARED = "__shared__"


# ---------------------------------------------------------------------------
# DB helpers (synchronous, sqlalchemy core)
# ---------------------------------------------------------------------------

def _make_engine(dsn: str):
    from sqlalchemy import create_engine
    return create_engine(dsn, pool_pre_ping=True)


def _iter_chunks(engine, partitions: list[str], *, batch: int = FETCH_BATCH_SIZE) -> Iterator[dict]:
    """Stream all active (non-deprecated) chunks for the given partitions.

    Uses psycopg directly with a server-side cursor to avoid the Supabase
    pooler's statement_timeout that breaks offset-based pagination for large
    tables.  Falls back to offset-pagination if psycopg is unavailable.
    """
    from sqlalchemy import select
    from app.db.tables import rag_chunks_table as t

    try:
        import psycopg  # psycopg v3

        # Extract connection kwargs from the SQLAlchemy URL so that URL-encoded
        # characters (e.g. %2A in the password) and dot-in-username Supabase formats
        # are handled correctly.
        url = engine.url
        connect_kwargs: dict = {
            "host": url.host,
            "port": url.port or 5432,
            "dbname": url.database,
            "user": url.username,
            "password": url.password,
        }

        # Fetch all chunks for all partitions in one query to avoid offset-pagination
        # issues with the Supabase session-mode pooler (large OFFSETs time out).
        placeholders = ",".join(["%s"] * len(partitions))
        sql = f"""
            SELECT rag_partition, chunk_id, source_id, chunk_type, text, metadata
            FROM rag_chunks
            WHERE rag_partition IN ({placeholders})
              AND deprecated_at IS NULL
              AND text IS NOT NULL
            ORDER BY rag_partition, chunk_id
        """
        with psycopg.connect(**connect_kwargs) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, partitions)
                rows = cur.fetchall()
        for db_row in rows:
            yield {
                "rag_partition": db_row[0],
                "chunk_id": db_row[1],
                "source_id": db_row[2],
                "chunk_type": db_row[3],
                "text": db_row[4],
                "metadata": db_row[5],
            }
        return
    except Exception as exc:
        logger.warning("psycopg fetchall unavailable (%s); falling back to offset pagination", exc)

    # Fallback: offset-based pagination via SQLAlchemy (may time out for large tables).
    with engine.connect() as conn:
        for partition in partitions:
            offset = 0
            while True:
                stmt = (
                    select(
                        t.c.rag_partition,
                        t.c.chunk_id,
                        t.c.source_id,
                        t.c.chunk_type,
                        t.c.text,
                        t.c.metadata,
                    )
                    .where(t.c.rag_partition == partition)
                    .where(t.c.deprecated_at.is_(None))
                    .where(t.c.text.isnot(None))
                    .order_by(t.c.chunk_id)
                    .limit(batch)
                    .offset(offset)
                )
                rows = conn.execute(stmt).mappings().all()
                if not rows:
                    break
                for row in rows:
                    yield dict(row)
                offset += len(rows)
                if len(rows) < batch:
                    break


def _count_by_type(engine, partitions: list[str]) -> dict[tuple[str, str], int]:
    """Count chunks grouped by (chunk_type, source_id) for the given partitions."""
    from sqlalchemy import select, func
    from app.db.tables import rag_chunks_table as t

    counts: dict[tuple[str, str], int] = {}
    with engine.connect() as conn:
        for partition in partitions:
            stmt = (
                select(
                    t.c.chunk_type,
                    t.c.source_id,
                    t.c.metadata,
                    func.count().label("n"),
                )
                .where(t.c.rag_partition == partition)
                .where(t.c.deprecated_at.is_(None))
                .group_by(t.c.chunk_type, t.c.source_id, t.c.metadata)
            )
            for row in conn.execute(stmt).mappings():
                key = (row["chunk_type"], row["source_id"])
                counts[key] = counts.get(key, 0) + row["n"]
    return counts


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def _load_tokenizer(model_name: str):
    try:
        from transformers import AutoTokenizer
    except ImportError:
        logger.error(
            "transformers not installed. Run: pip install transformers sentencepiece"
        )
        sys.exit(1)
    logger.info("Loading tokenizer for %s (tokenizer files only, no weights)...", model_name)
    tok = AutoTokenizer.from_pretrained(model_name)
    logger.info("Tokenizer loaded.")
    return tok


def _count_tokens(tokenizer, text: str, prefix: str = "") -> int:
    """Count tokens including an optional passage/query prefix (e.g. 'passage: ')."""
    ids = tokenizer.encode(prefix + text, add_special_tokens=True)
    return len(ids)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _source_title(metadata: dict | None) -> str:
    if not isinstance(metadata, dict):
        return ""
    st = metadata.get("source_title") or ""
    if st:
        return str(st)
    inner = metadata.get("metadata")
    if isinstance(inner, dict):
        return str(inner.get("source_title") or "")
    return ""


def _safe_max_chars(violations: list[dict]) -> int:
    """Estimate a safe --max-chars for rag:chunk based on observed chars/token ratio."""
    if not violations:
        return 1400
    ratios = [v["char_count"] / v["token_count"] for v in violations if v["token_count"] > 0]
    if not ratios:
        return 1400
    avg_ratio = sum(ratios) / len(ratios)
    # Target 450 tokens (10% headroom below 512) × observed chars/token
    return int(450 * avg_ratio)


def run_token_audit(
    engine,
    tokenizer,
    partitions: list[str],
    *,
    limit: int,
    warn: int,
    output_csv: str | None,
    prefix_passage: str = "",
) -> list[dict]:
    """Run token audit. Returns list of violation dicts."""

    violations: list[dict] = []
    warn_only: list[dict] = []
    total = 0
    over_limit = 0
    over_warn = 0

    if prefix_passage:
        logger.info("Using passage prefix: %r (%d extra tokens)", prefix_passage,
                    len(tokenizer.encode(prefix_passage, add_special_tokens=False)))
    logger.info("Auditing partitions: %s", partitions)

    for chunk in _iter_chunks(engine, partitions):
        total += 1
        text = chunk.get("text") or ""
        if not text.strip():
            continue
        n_tokens = _count_tokens(tokenizer, text, prefix=prefix_passage)
        char_count = len(text)

        if total % 500 == 0:
            logger.info("  Progress: %d chunks scanned...", total)

        entry = {
            "partition": chunk["rag_partition"],
            "chunk_id": chunk["chunk_id"],
            "source_id": chunk["source_id"],
            "chunk_type": chunk["chunk_type"],
            "source_title": _source_title(chunk.get("metadata")),
            "char_count": char_count,
            "token_count": n_tokens,  # includes prefix tokens if --prefix-passage set
            "over_limit": n_tokens > limit,
        }

        if n_tokens > limit:
            over_limit += 1
            violations.append(entry)
        elif n_tokens > warn:
            over_warn += 1
            warn_only.append(entry)

    # ---- Summary --------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"TOKEN AUDIT — model limit: {limit} tokens / warn: {warn} tokens")
    print("=" * 70)
    print(f"Total chunks scanned  : {total:,}")
    print(f"Over limit (>{limit}) : {over_limit:,}  ({over_limit/max(total,1)*100:.1f}%)")
    print(f"Warning zone ({warn}–{limit}): {over_warn:,}  ({over_warn/max(total,1)*100:.1f}%)")
    print()

    if violations:
        # Group by source
        by_source: dict[str, list[dict]] = defaultdict(list)
        for v in violations:
            key = v["source_title"] or v["source_id"]
            by_source[key].append(v)

        print(f"SOURCES WITH OVER-LIMIT CHUNKS ({len(by_source)} sources):")
        print(f"  safe --max = empf. --max-chars für rag:chunk, basierend auf gemessener")
        print(f"  Inhaltsdichte dieser Quelle (Ziel: 450 Tokens, 12% Puffer unter 512)")
        print()
        print(f"{'Source':<50} {'#chunks':>8} {'max tokens':>12} {'safe --max':>12}")
        print("-" * 84)
        for src, items in sorted(by_source.items(), key=lambda kv: -max(i["token_count"] for i in kv[1])):
            max_tok = max(i["token_count"] for i in items)
            safe = _safe_max_chars(items)
            print(f"{src[:50]:<50} {len(items):>8} {max_tok:>12} {safe:>11}c")

        print()
        print("Recommendation: re-run rag:chunk with --max <safe-max-chars> for affected sources.")
    else:
        print("All chunks within token limit.")

    if warn_only:
        print(f"\n{len(warn_only)} chunks in warning zone ({warn}–{limit} tokens) – fine but close to limit.")

    # ---- CSV output -----------------------------------------------------
    if output_csv:
        all_flagged = violations + warn_only
        fieldnames = ["partition", "chunk_id", "source_id", "chunk_type", "source_title",
                      "char_count", "token_count", "over_limit"]
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_flagged)
        logger.info("Report written to %s (%d rows)", output_csv, len(all_flagged))

    return violations


def run_completeness_check(engine, collection: str, shared_partition: str = RAG_PARTITION_SHARED) -> None:
    """Print a chunk_type × source_id count table for the collection."""
    partitions = [collection, shared_partition]
    counts = _count_by_type(engine, partitions)

    # Resolve source_title per source_id from metadata (we need one exemplar row)
    from sqlalchemy import select
    from app.db.tables import rag_chunks_table as t

    source_titles: dict[str, str] = {}
    with engine.connect() as conn:
        for partition in partitions:
            stmt = (
                select(t.c.source_id, t.c.metadata)
                .where(t.c.rag_partition == partition)
                .where(t.c.deprecated_at.is_(None))
                .distinct(t.c.source_id)
            )
            for row in conn.execute(stmt).mappings():
                sid = row["source_id"]
                if sid not in source_titles:
                    source_titles[sid] = _source_title(row.get("metadata")) or sid

    # Gather all chunk types and source_ids seen
    chunk_types_seen = sorted({ct for ct, _ in counts.keys()})
    source_ids_seen = sorted(set(sid for _, sid in counts.keys()))

    print("\n" + "=" * 70)
    print(f"COMPLETENESS CHECK — collection: {collection}")
    print("=" * 70)

    col_w = 22
    src_w = 42
    header = f"{'Source':<{src_w}}" + "".join(f" {ct[:col_w]:>{col_w}}" for ct in chunk_types_seen)
    print(header)
    print("-" * (src_w + (col_w + 1) * len(chunk_types_seen)))

    total_by_type: dict[str, int] = defaultdict(int)
    for sid in source_ids_seen:
        title = source_titles.get(sid, sid)[:src_w]
        row_str = f"{title:<{src_w}}"
        for ct in chunk_types_seen:
            n = counts.get((ct, sid), 0)
            total_by_type[ct] += n
            cell = str(n) if n > 0 else "—"
            row_str += f" {cell:>{col_w}}"
        print(row_str)

    print("-" * (src_w + (col_w + 1) * len(chunk_types_seen)))
    totals = f"{'TOTAL':<{src_w}}" + "".join(f" {total_by_type[ct]:>{col_w},}" for ct in chunk_types_seen)
    print(totals)

    # Flag missing expected types
    missing: list[str] = []
    for ct in COMPLETENESS_ALL:
        if ct not in chunk_types_seen or total_by_type.get(ct, 0) == 0:
            missing.append(ct)

    if missing:
        print(f"\nWARNING: Expected chunk types missing or empty: {', '.join(missing)}")
        print("Note: 'typology' is intentionally excluded (requires Qdrant to be filled first).")
    else:
        print(f"\nAll expected chunk types present: {', '.join(sorted(COMPLETENESS_ALL))}")
        print("Note: 'typology' is intentionally excluded (requires Qdrant to be filled first).")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit rag_chunks token counts for a given embedding model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--collection",
        required=True,
        help="rag_partition name for the assistant collection (e.g. philo-von-freisinn-v2)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("EMBEDDINGS_MODEL", DEFAULT_MODEL),
        help=f"HuggingFace model name for the tokenizer (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_TOKEN_LIMIT,
        help=f"Token count hard limit (default: {DEFAULT_TOKEN_LIMIT})",
    )
    parser.add_argument(
        "--warn",
        type=int,
        default=DEFAULT_WARN_THRESHOLD,
        help=f"Token count warning threshold (default: {DEFAULT_WARN_THRESHOLD})",
    )
    parser.add_argument(
        "--output",
        metavar="FILE.csv",
        help="Write flagged chunks (over warn or limit) to CSV",
    )
    parser.add_argument(
        "--check-completeness",
        action="store_true",
        help="Also print a completeness table (chunk_type × source count)",
    )
    parser.add_argument(
        "--prefix-passage",
        default=os.environ.get("RAGRUN_EMBEDDING_PREFIX_PASSAGE", ""),
        metavar="PREFIX",
        help=(
            "Passage prefix prepended before each chunk text during embedding "
            "(e.g. 'passage: ' for multilingual-e5-large). "
            "Default: $RAGRUN_EMBEDDING_PREFIX_PASSAGE or empty."
        ),
    )
    parser.add_argument(
        "--no-shared",
        action="store_true",
        help="Skip the __shared__ partition (only audit the assistant collection)",
    )
    parser.add_argument(
        "--dsn",
        default=os.environ.get("RAGRUN_POSTGRES_DSN") or os.environ.get("POSTGRES_DSN"),
        help="Postgres DSN (default: $RAGRUN_POSTGRES_DSN)",
    )
    args = parser.parse_args()

    if not args.dsn:
        logger.error(
            "No Postgres DSN provided. Set RAGRUN_POSTGRES_DSN or pass --dsn."
        )
        sys.exit(1)

    partitions = [args.collection]
    if not args.no_shared:
        partitions.append(RAG_PARTITION_SHARED)

    engine = _make_engine(args.dsn)
    tokenizer = _load_tokenizer(args.model)

    run_token_audit(
        engine,
        tokenizer,
        partitions,
        limit=args.limit,
        warn=args.warn,
        output_csv=args.output,
        prefix_passage=args.prefix_passage,
    )

    if args.check_completeness:
        run_completeness_check(engine, args.collection)


if __name__ == "__main__":
    main()
