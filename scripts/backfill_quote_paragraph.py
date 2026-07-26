"""Backfill metadata.paragraph on quote chunks in rag_chunks.

Sources (in order):
  1. Local quotes-chunks.jsonl / lecture *.quotes.jsonl — match by chunk_id or content_hash
  2. results/augmentation/quotes.md — match by source_id + segment_id + normalized quote text
  3. Propagate paragraph from parent quote → quote_explanation via parent_id

Run from ragrun project root:
    python scripts/backfill_quote_paragraph.py --dry-run
    python scripts/backfill_quote_paragraph.py

Env: RAGRUN_POSTGRES_DSN (or app.config settings), RAGKEEP_PROJECT_ROOT optional.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text

from app.core.providers import get_sync_engine

BOOKDIR_NAMESPACE_UUID = uuid.UUID("a7f2c891-e4b1-4d3a-9c06-8e5f3b2a1d0e")
SEGMENT_ID_COMMENT_RE = re.compile(
    r"<!--\s*segment_id:\s*([^|]+?)\s*\|\s*([^>]+?)\s*-->"
)
QUOTES_MD_HEADING_RE = re.compile(r"^##\s+(Kapitel|Vortrag)\s+(\d+):\s*(.+?)\s*$")
DEFAULT_PARTITION = "philo-von-freisinn"


@dataclass(frozen=True)
class MdQuote:
    segment_id: str
    paragraph: int
    quote: str
    quote_key: str


def _normalize_ws(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _strip_brackets(value: str) -> str:
    return re.sub(r"\[[^\]]*]", " ", value)


def quote_key(value: str) -> str:
    return _normalize_ws(_strip_brackets(value)).lower()


def parse_paragraph_number(raw: str) -> Optional[int]:
    cleaned = (raw or "").replace("|", "").strip()
    if not cleaned:
        return None
    try:
        n = int(cleaned, 10)
    except ValueError:
        return None
    return n if n >= 1 else None


def resolve_ragkeep_root(explicit: str | None) -> Path:
    if explicit and explicit.strip():
        return Path(explicit.strip()).resolve()
    env = (os.environ.get("RAGKEEP_PROJECT_ROOT") or "").strip()
    if env:
        return Path(env).resolve()
    return (Path(__file__).resolve().parent.parent / "ragkeep").resolve()


def find_quote_jsonl_files(ragkeep: Path) -> List[Path]:
    books = sorted(ragkeep.glob("books/**/results/rag-chunks/quotes-chunks.jsonl"))
    lectures = sorted((ragkeep / "lectures" / "chunks" / "quotes").glob("*.quotes.jsonl"))
    return [*books, *lectures]


def iter_jsonl_records(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            yield json.loads(stripped)


def load_jsonl_indexes(
    ragkeep: Path,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    by_chunk_id: Dict[str, int] = {}
    by_content_hash: Dict[str, int] = {}
    for path in find_quote_jsonl_files(ragkeep):
        for record in iter_jsonl_records(path):
            meta = record.get("metadata")
            if not isinstance(meta, dict):
                continue
            chunk_type = str(meta.get("chunk_type") or "")
            if chunk_type not in {"quote", "quote_explanation"}:
                continue
            paragraph = meta.get("paragraph")
            if not isinstance(paragraph, int) or paragraph < 1:
                continue
            chunk_id = str(meta.get("chunk_id") or "").strip()
            content_hash = str(meta.get("content_hash") or "").strip()
            if chunk_id:
                by_chunk_id[chunk_id] = paragraph
            if content_hash and chunk_type == "quote":
                by_content_hash[content_hash] = paragraph
    return by_chunk_id, by_content_hash


def parse_quotes_markdown(raw: str) -> List[MdQuote]:
    lines = raw.splitlines()
    quotes: List[MdQuote] = []
    current_slug = ""
    body_lines: List[str] = []

    def flush_body() -> None:
        nonlocal body_lines
        joined = "\n".join(body_lines).strip()
        if joined:
            for block in re.split(r"\n\s*\n", joined):
                trimmed = block.strip()
                if not trimmed:
                    continue
                first_line, _, rest = trimmed.partition("\n")
                comment = SEGMENT_ID_COMMENT_RE.match(first_line.strip())
                if comment:
                    segment_id = comment.group(1).strip()
                    paragraph = parse_paragraph_number(comment.group(2))
                    quote_text = rest.strip() if rest else ""
                else:
                    segment_id = current_slug
                    paragraph = 1
                    quote_text = trimmed
                if not quote_text or paragraph is None:
                    continue
                normalized = _normalize_ws(quote_text)
                quotes.append(
                    MdQuote(
                        segment_id=segment_id,
                        paragraph=paragraph,
                        quote=normalized,
                        quote_key=quote_key(normalized),
                    )
                )
        body_lines = []

    for line in lines:
        heading = QUOTES_MD_HEADING_RE.match(line)
        if heading:
            flush_body()
            current_slug = _normalize_ws(heading.group(3) or "")
            continue
        if current_slug or line.strip():
            body_lines.append(line)
    flush_body()
    return quotes


def read_book_quotes_source_id(book_dir: Path) -> Optional[str]:
    manifest_path = book_dir / "book-manifest.yaml"
    if not manifest_path.is_file():
        return None
    raw = manifest_path.read_text(encoding="utf-8")
    book_id: Optional[str] = None
    for line in raw.splitlines():
        if line.startswith("book-id:"):
            book_id = line.split(":", 1)[1].strip()
            break
    if book_id:
        return f"{book_id}:quotes"
    book_dir_name = book_dir.name
    derived = uuid.uuid5(BOOKDIR_NAMESPACE_UUID, book_dir_name)
    return f"{derived}:quotes"


def load_quotes_md_index(ragkeep: Path) -> Dict[str, List[MdQuote]]:
    out: Dict[str, List[MdQuote]] = {}
    for quotes_path in sorted(ragkeep.glob("books/**/results/augmentation/quotes.md")):
        book_dir = quotes_path.parent.parent.parent
        source_id = read_book_quotes_source_id(book_dir)
        if not source_id:
            continue
        raw = quotes_path.read_text(encoding="utf-8")
        parsed = parse_quotes_markdown(raw)
        if parsed:
            out[source_id] = parsed
    return out


def fetch_rows_missing_paragraph(engine, partition: str) -> List[Dict[str, Any]]:
    sql = text(
        """
        SELECT chunk_id, chunk_type, source_id, content_hash, text, metadata
        FROM rag_chunks
        WHERE rag_partition = :partition
          AND deprecated_at IS NULL
          AND chunk_type IN ('quote', 'quote_explanation')
          AND (metadata->>'paragraph') IS NULL
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(sql, {"partition": partition}).mappings().all()
    return [dict(row) for row in rows]


def apply_updates(
    engine,
    updates: Dict[str, int],
    *,
    dry_run: bool,
) -> int:
    if not updates:
        return 0
    if dry_run:
        return len(updates)

    now = datetime.now(timezone.utc)
    batch_size = 200
    items = list(updates.items())
    sql = text(
        """
        UPDATE rag_chunks AS rc
        SET metadata = jsonb_set(rc.metadata, '{paragraph}', to_jsonb(v.paragraph), true),
            updated_at = :updated_at
        FROM (
            SELECT unnest(CAST(:chunk_ids AS text[])) AS chunk_id,
                   unnest(CAST(:paragraphs AS integer[])) AS paragraph
        ) AS v
        WHERE rc.chunk_id = v.chunk_id
          AND (rc.metadata->>'paragraph') IS NULL
        """
    )
    applied = 0
    with engine.begin() as conn:
        for offset in range(0, len(items), batch_size):
            batch = items[offset : offset + batch_size]
            chunk_ids = [chunk_id for chunk_id, _ in batch]
            paragraphs = [paragraph for _, paragraph in batch]
            result = conn.execute(
                sql,
                {
                    "chunk_ids": chunk_ids,
                    "paragraphs": paragraphs,
                    "updated_at": now,
                },
            )
            applied += int(result.rowcount or 0)
    return applied


def match_md_quote(
    row: Dict[str, Any],
    md_quotes: List[MdQuote],
) -> Optional[int]:
    meta = row.get("metadata") or {}
    if not isinstance(meta, dict):
        return None
    segment_id = str(meta.get("segment_id") or "").strip().lower()
    if not segment_id:
        return None
    chunk_text = str(row.get("text") or "")
    key = quote_key(chunk_text)
    if not key:
        return None

    candidates = [
        q
        for q in md_quotes
        if q.segment_id.strip().lower() == segment_id and q.quote_key == key
    ]
    if len(candidates) == 1:
        return candidates[0].paragraph
    if len(candidates) > 1:
        return None

    # Fallback: unique segment+paragraph with fuzzy containment
    loose = [
        q
        for q in md_quotes
        if q.segment_id.strip().lower() == segment_id
        and (q.quote_key in key or key in q.quote_key)
    ]
    if len(loose) == 1:
        return loose[0].paragraph
    return None


def build_updates(
    rows: List[Dict[str, Any]],
    by_chunk_id: Dict[str, int],
    by_content_hash: Dict[str, int],
    md_by_source: Dict[str, List[MdQuote]],
) -> Tuple[Dict[str, int], Dict[str, int]]:
    stats = {
        "jsonl_chunk_id": 0,
        "jsonl_content_hash": 0,
        "quotes_md": 0,
        "unmatched": 0,
    }
    updates: Dict[str, int] = {}

    for row in rows:
        chunk_id = str(row["chunk_id"])
        if chunk_id in updates:
            continue

        paragraph: Optional[int] = None
        if chunk_id in by_chunk_id:
            paragraph = by_chunk_id[chunk_id]
            stats["jsonl_chunk_id"] += 1
        else:
            content_hash = str(row.get("content_hash") or "")
            if content_hash and content_hash in by_content_hash:
                paragraph = by_content_hash[content_hash]
                stats["jsonl_content_hash"] += 1

        if paragraph is None and str(row.get("chunk_type")) == "quote":
            source_id = str(row.get("source_id") or "")
            md_quotes = md_by_source.get(source_id)
            if md_quotes:
                paragraph = match_md_quote(row, md_quotes)
                if paragraph is not None:
                    stats["quotes_md"] += 1

        if paragraph is None:
            stats["unmatched"] += 1
            continue
        updates[chunk_id] = paragraph

    return updates, stats


def propagate_explanation_paragraphs(
    engine,
    partition: str,
    *,
    dry_run: bool,
) -> int:
    sql_count = text(
        """
        SELECT COUNT(*) AS n
        FROM rag_chunks child
        JOIN rag_chunks parent
          ON parent.chunk_id = child.metadata->>'parent_id'
         AND parent.rag_partition = child.rag_partition
        WHERE child.rag_partition = :partition
          AND child.deprecated_at IS NULL
          AND parent.deprecated_at IS NULL
          AND child.chunk_type = 'quote_explanation'
          AND (child.metadata->>'paragraph') IS NULL
          AND parent.metadata->>'paragraph' IS NOT NULL
        """
    )
    with engine.connect() as conn:
        pending = int(conn.execute(sql_count, {"partition": partition}).scalar_one())
    if pending == 0 or dry_run:
        return pending

    sql_update = text(
        """
        UPDATE rag_chunks AS child
        SET metadata = jsonb_set(
                child.metadata,
                '{paragraph}',
                to_jsonb((parent.metadata->>'paragraph')::integer),
                true
            ),
            updated_at = :updated_at
        FROM rag_chunks AS parent
        WHERE parent.chunk_id = child.metadata->>'parent_id'
          AND parent.rag_partition = child.rag_partition
          AND child.rag_partition = :partition
          AND child.deprecated_at IS NULL
          AND parent.deprecated_at IS NULL
          AND child.chunk_type = 'quote_explanation'
          AND (child.metadata->>'paragraph') IS NULL
          AND parent.metadata->>'paragraph' IS NOT NULL
        """
    )
    now = datetime.now(timezone.utc)
    with engine.begin() as conn:
        result = conn.execute(sql_update, {"partition": partition, "updated_at": now})
        return int(result.rowcount or 0)


def count_missing(engine, partition: str) -> int:
    sql = text(
        """
        SELECT COUNT(*) AS n
        FROM rag_chunks
        WHERE rag_partition = :partition
          AND deprecated_at IS NULL
          AND chunk_type IN ('quote', 'quote_explanation')
          AND (metadata->>'paragraph') IS NULL
        """
    )
    with engine.connect() as conn:
        return int(conn.execute(sql, {"partition": partition}).scalar_one())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--partition", default=DEFAULT_PARTITION)
    parser.add_argument("--ragkeep-root", default=None)
    args = parser.parse_args()

    ragkeep = resolve_ragkeep_root(args.ragkeep_root)
    if not ragkeep.is_dir():
        raise SystemExit(f"ragkeep root not found: {ragkeep}")

    engine = get_sync_engine()
    before = count_missing(engine, args.partition)
    print(f"Missing paragraph before: {before} (partition={args.partition})", flush=True)

    by_chunk_id, by_content_hash = load_jsonl_indexes(ragkeep)
    md_by_source = load_quotes_md_index(ragkeep)
    print(
        f"Loaded jsonl index: {len(by_chunk_id)} chunk_ids, "
        f"{len(by_content_hash)} content_hashes; "
        f"{len(md_by_source)} books with quotes.md"
    )

    rows = fetch_rows_missing_paragraph(engine, args.partition)
    updates, stats = build_updates(rows, by_chunk_id, by_content_hash, md_by_source)
    applied = apply_updates(engine, updates, dry_run=args.dry_run)
    print(
        f"Quote updates: {applied} "
        f"(jsonl_id={stats['jsonl_chunk_id']}, "
        f"jsonl_hash={stats['jsonl_content_hash']}, "
        f"quotes_md={stats['quotes_md']}, "
        f"unmatched={stats['unmatched']})"
    )

    propagated = propagate_explanation_paragraphs(
        engine, args.partition, dry_run=args.dry_run
    )
    print(f"quote_explanation propagated from parent: {propagated}")

    after = count_missing(engine, args.partition)
    if args.dry_run:
        print(f"Dry-run complete. Would leave missing: {before - applied - propagated}")
    else:
        print(f"Missing paragraph after: {after}")


if __name__ == "__main__":
    main()
