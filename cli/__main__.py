"""ragrun CLI entry point."""
from __future__ import annotations

import argparse
import sys

from cli.commands.chunks_delete import run_chunks_delete
from cli.commands.chunks_info import run_chunks_info
from cli.commands.export_reader import run_export_reader
from cli.commands.publish_site import run_publish_site


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
        choices=["book", "concept"],
        help="Filter by chunk types (book, concept)",
    )
    delete_parser.add_argument(
        "--chunk-type",
        choices=["book", "secondary_book", "summary", "begriff", "quote"],
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

    export_parser = subparsers.add_parser(
        "reader:export",
        help="Export rag_paragraphs chapters to HTML (optional R2 upload)",
    )
    export_parser.add_argument(
        "--out-dir",
        default=None,
        help="Write HTML tree locally (assets/, books/...)",
    )
    export_parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload to Cloudflare R2 (RAGRUN_R2_* and RAGRUN_R2_BUCKET)",
    )
    export_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Export at most N chapters (debug)",
    )

    publish_parser = subparsers.add_parser(
        "reader:publish",
        help="Diff the corpus site and upload it to the configured R2 bucket",
    )
    publish_parser.add_argument(
        "--strategy",
        choices=["replace", "sync"],
        required=True,
        help="replace uploads every page and deletes extras; sync uploads changes only",
    )
    publish_parser.add_argument(
        "--apply",
        action="store_true",
        help="Write to R2. Without this flag the command only prints the diff.",
    )
    publish_parser.add_argument(
        "--prune",
        action="store_true",
        help="With --strategy sync, delete bucket objects the corpus no longer renders",
    )

    args = parser.parse_args()

    if args.command == "chunks:info":
        run_chunks_info(args.assistant)
        return

    if args.command == "reader:export":
        run_export_reader(args)
        return

    if args.command == "reader:publish":
        run_publish_site(args)
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
