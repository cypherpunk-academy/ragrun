"""chunks:info – show chunk stats from Qdrant and Postgres."""
from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from sqlalchemy import text

from app.config import settings
from app.db.session import get_engine
from app.infra.qdrant_client import QdrantClient


def _resolve_assistant(assistant: str) -> str:
    """Resolve short name (e.g. philo) to full collection name (e.g. philo-von-freisinn)."""
    project_root = Path(__file__).resolve().parent.parent.parent
    assistants_dir = project_root / settings.assistants_root
    if not assistants_dir.is_dir():
        return assistant
    names = [d.name for d in assistants_dir.iterdir() if d.is_dir()]
    if assistant in names:
        return assistant
    matches = [n for n in names if n.startswith(assistant + "-")]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        return matches[0]
    return assistant


def _fmt_num(n: int) -> str:
    """Format number with German thousands separator (e.g. 28000 -> 28.000)."""
    return f"{n:,}".replace(",", ".")


def _fmt_ts(dt: datetime | None) -> str:
    """Format datetime for display."""
    if dt is None:
        return "—"
    base = dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    if dt.tzinfo:
        tz = dt.strftime("%z")
        if len(tz) >= 5:
            base += f" {tz[:3]}:{tz[3:]}"
        else:
            base += f" {tz}"
    return base


def _print_section(
    console: Console,
    header: str,
    version: str,
    total: int,
    oldest: datetime | None,
    newest: datetime | None,
    chunk_types: dict[str, int],
) -> None:
    """Print one section (Qdrant or Postgres) with colored output."""
    # Header: blue
    console.print(header, style="bold blue")
    console.print()

    # Attribute names: yellow, values: white/gray
    attr = "yellow"
    num = "white"
    txt = "dim"

    console.print(f"[{attr}]Version:[/] [{num}]{version}[/]")
    console.print(f"[{attr}]Total[/] [{num}]{_fmt_num(total)}[/] [{txt}]chunks[/]")
    console.print(f"[{attr}]Oldest:[/] [{num}]{_fmt_ts(oldest)}[/]")
    console.print(f"[{attr}]Newest:[/] [{num}]{_fmt_ts(newest)}[/]")
    console.print()
    console.print(f"[{attr}]Chunk-Types:[/]")
    for ct, count in sorted(chunk_types.items(), key=lambda x: -x[1]):
        console.print(f"  [{txt}]{ct}:[/] [{num}]{_fmt_num(count)}[/] [{txt}]chunks[/]")
    console.print()


async def _fetch_qdrant(collection: str) -> dict[str, Any]:
    """Fetch Qdrant stats for collection."""
    client = QdrantClient(
        str(settings.qdrant_url),
        api_key=settings.qdrant_api_key,
        timeout=60.0,
    )
    version = await client.get_version()
    info = await client.get_collection_info(collection)
    if info is None:
        return {
            "version": version,
            "total": 0,
            "oldest": None,
            "newest": None,
            "chunk_types": {},
            "collection_exists": False,
        }

    points_count = info.get("points_count") or 0
    chunk_types: dict[str, int] = defaultdict(int)
    oldest: datetime | None = None
    newest: datetime | None = None

    if points_count > 0:
        points = await client.scroll_all_points(
            collection,
            with_payload=True,
            with_vectors=False,
            limit=500,
        )
        for pt in points:
            payload = pt.get("payload") or {}
            ct = payload.get("chunk_type")
            if ct is not None:
                chunk_types[str(ct)] += 1
            raw_created = payload.get("created_at")
            if raw_created:
                try:
                    if isinstance(raw_created, str):
                        dt = datetime.fromisoformat(raw_created.replace("Z", "+00:00"))
                    else:
                        dt = raw_created
                    if oldest is None or dt < oldest:
                        oldest = dt
                    if newest is None or dt > newest:
                        newest = dt
                except (ValueError, TypeError):
                    pass

    return {
        "version": version,
        "total": points_count,
        "oldest": oldest,
        "newest": newest,
        "chunk_types": dict(chunk_types),
        "collection_exists": True,
    }


def _fetch_postgres(collection: str) -> dict[str, Any]:
    """Fetch Postgres rag_chunks stats for collection."""
    engine = get_engine()
    with engine.connect() as conn:
        version_row = conn.execute(text("SELECT version()")).fetchone()
        pg_version = version_row[0] if version_row else "unknown"

        count_row = conn.execute(
            text(
                "SELECT COUNT(*), MIN(created_at), MAX(created_at) FROM rag_chunks WHERE collection = :coll"
            ),
            {"coll": collection},
        ).fetchone()
        total = count_row[0] or 0
        oldest = count_row[1] if count_row else None
        newest = count_row[2] if count_row else None

        type_rows = conn.execute(
            text(
                "SELECT chunk_type, COUNT(*) FROM rag_chunks WHERE collection = :coll GROUP BY chunk_type"
            ),
            {"coll": collection},
        ).fetchall()
        chunk_types = {row[0]: row[1] for row in type_rows}

    return {
        "version": pg_version,
        "total": total,
        "oldest": oldest,
        "newest": newest,
        "chunk_types": chunk_types,
    }


def run_chunks_info(assistant: str) -> None:
    """Run chunks:info for the given assistant/collection."""
    console = Console()
    collection = _resolve_assistant(assistant.strip())

    try:
        qdrant_data = asyncio.run(_fetch_qdrant(collection))
    except Exception as e:
        console.print(f"[red]Qdrant error:[/] {e}")
        console.print("[dim]Ensure Qdrant is running (e.g. docker-compose up -d qdrant) and RAGRUN_QDRANT_URL is set.[/]")
        raise SystemExit(1) from e

    try:
        pg_data = _fetch_postgres(collection)
    except Exception as e:
        console.print(f"[red]Postgres error:[/] {e}")
        console.print("[dim]Ensure Postgres is running and RAGRUN_POSTGRES_DSN is set.[/]")
        raise SystemExit(1) from e

    if qdrant_data["total"] == 0 and not qdrant_data.get("collection_exists", True):
        console.print(f"[yellow]Collection '{collection}' not found in Qdrant.[/]")
        console.print()

    _print_section(
        console,
        "Qdrant",
        qdrant_data["version"],
        qdrant_data["total"],
        qdrant_data["oldest"],
        qdrant_data["newest"],
        qdrant_data["chunk_types"],
    )
    _print_section(
        console,
        "Postgres (rag_chunks)",
        pg_data["version"],
        pg_data["total"],
        pg_data["oldest"],
        pg_data["newest"],
        pg_data["chunk_types"],
    )
