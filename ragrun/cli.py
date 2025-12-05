"""Lightweight CLI helpers for ragrun."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional

import httpx

from ragrun.models import ChunkRecord


def _load_jsonl(path: Path) -> List[dict]:
    """Load JSONL file and return validated chunk dictionaries."""

    raw_chunks: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            payload = json.loads(line)
            chunk = ChunkRecord.from_dict(payload)
            raw_chunks.append(chunk.model_dump(mode="json"))
    return raw_chunks


def _chunk(items: List[dict], size: int) -> Iterable[List[dict]]:
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


def _command_ingest(args: argparse.Namespace) -> int:
    chunks = _load_jsonl(args.file)
    if not chunks:
        print("No chunks found in file.")
        return 1

    api_base = args.api.rstrip("/")

    try:
        with httpx.Client(timeout=args.timeout) as client:
            for idx, batch in enumerate(_chunk(chunks, args.batch_size), start=1):
                payload = {"collection": args.collection, "chunks": batch}
                if args.embedding_model:
                    payload["embedding_model"] = args.embedding_model
                response = client.post(f"{api_base}/rag/upload", json=payload)
                response.raise_for_status()
                body = response.json()
                print(
                    f"Batch {idx}: ingested {body.get('ingested')} of {body.get('requested')} "
                    f"(duplicates {body.get('duplicates', 0)}) "
                    f"[ingestion_id={body.get('ingestion_id')}]"
                )
    except httpx.HTTPError as exc:
        print(f"Ingestion failed: {exc}")
        if hasattr(exc, "response") and exc.response is not None:
            try:
                error_detail = exc.response.json()
                print(f"Error detail: {error_detail}")
            except Exception:
                print(f"Response text: {exc.response.text}")
        return 1

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragrun",
        description="Utilities for working with the ragrun ingestion API.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Upload JSONL chunks via /rag/upload.")
    ingest_parser.add_argument("--file", "-f", type=Path, required=True, help="Path to JSONL chunk file.")
    ingest_parser.add_argument("--collection", "-c", required=True, help="Target collection name.")
    ingest_parser.add_argument(
        "--api",
        default="http://localhost:8000",
        help="ragrun API base URL (default: http://localhost:8000).",
    )
    ingest_parser.add_argument(
        "--embedding-model",
        "-m",
        default=None,
        help="Optional embedding model override.",
    )
    ingest_parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=64,
        help="Chunk count per request (default: 64).",
    )
    ingest_parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout per request in seconds (default: 30).",
    )
    ingest_parser.set_defaults(func=_command_ingest)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

