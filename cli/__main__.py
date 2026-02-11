"""ragrun CLI entry point."""
from __future__ import annotations

import argparse
import sys

from cli.commands.chunks_delete import run_chunks_delete
from cli.commands.chunks_info import run_chunks_info


def main() -> None:
    parser = argparse.ArgumentParser(prog="ragrun", description="ragrun maintenance CLI")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    info_parser = subparsers.add_parser(
        "chunks:info",
        help="Show chunk stats from Qdrant and Postgres",
    )
    info_parser.add_argument(
        "assistant",
        help="Assistant/collection name (e.g. philo-von-freisinn)",
    )

    delete_parser = subparsers.add_parser(
        "chunks:delete",
        help="Delete chunks from Qdrant and Postgres",
    )
    delete_parser.add_argument(
        "assistant",
        help="Assistant/collection name (e.g. philo-von-freisinn)",
    )
    delete_parser.add_argument(
        "--chunk-types",
        nargs="*",
        choices=["book", "essay", "concept"],
        help="Filter by chunk types (book, essay, concept)",
    )
    delete_parser.add_argument(
        "--chunk-type",
        choices=["book", "secondary_book", "essay", "summary", "begriff_list", "quote"],
        help="Delete all chunks of this type (optionally scoped by --book)",
    )
    delete_parser.add_argument(
        "--book",
        nargs="?",
        default=None,
        const="",
        metavar="bookDir",
        help="Book/source_id to delete. Use --book with no value to list books.",
    )
    delete_parser.add_argument(
        "--chunk-id",
        nargs="*",
        metavar="chunk_id",
        dest="chunk_id",
        help="Specific chunk IDs to delete",
    )
    delete_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview deletion without executing",
    )

    args = parser.parse_args()

    if args.command == "chunks:info":
        run_chunks_info(args.assistant)
        return

    if args.command == "chunks:delete":
        run_chunks_delete(
            assistant=args.assistant,
            chunk_types=args.chunk_types,
            chunk_type=args.chunk_type,
            book=args.book,
            chunk_id=args.chunk_id,
            dry_run=args.dry_run,
        )
        return

    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
