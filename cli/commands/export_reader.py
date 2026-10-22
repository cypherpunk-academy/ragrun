"""Export corpus chapters to HTML (and optionally Cloudflare R2)."""
from __future__ import annotations

import logging
from argparse import Namespace
from pathlib import Path

from app.db.session import get_engine
from app.reader.export import export_reader

logger = logging.getLogger(__name__)


def run_export_reader(args: Namespace) -> None:
    out = Path(args.out_dir) if args.out_dir else None
    stats = export_reader(
        get_engine(),
        out_dir=out,
        upload=bool(args.upload),
        limit=args.limit,
    )
    logger.info(
        "reader export done segments=%s written=%s failed=%s",
        stats["segments"],
        stats["written"],
        stats["failed"],
    )
    print(
        f"exported {stats['written']}/{stats['segments']} chapters"
        + (f" ({stats['failed']} failed)" if stats["failed"] else "")
    )
