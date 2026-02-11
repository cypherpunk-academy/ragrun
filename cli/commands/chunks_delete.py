"""chunks:delete – delete chunks from Qdrant and Postgres."""
from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import NAMESPACE_DNS, uuid5

from rich.console import Console
from sqlalchemy import select, text

from app.config import settings
from app.db.session import get_engine
from app.db.tables import chunks_table
from app.ingestion.repositories import ChunkMirrorRepository
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


def _map_chunk_types(user_types: list[str]) -> list[str]:
    """Map user-facing chunk types (book, essay, concept) to DB chunk_type values."""
    mapping = {
        "book": ["book", "chapter_summary", "secondary_book"],
        "essay": ["essay"],
        "concept": ["begriff_list"],
    }
    result: list[str] = []
    for t in user_types:
        t = (t or "").strip().lower()
        if t in mapping:
            result.extend(mapping[t])
    return list(dict.fromkeys(result))  # preserve order, dedupe


def _list_books(engine, collection: str) -> list[tuple[str, str, int, str | None, str | None]]:
    """List books (source_id) with metadata. Only types 'book' and 'secondary_book'."""
    query = text(
        """
        SELECT source_id, chunk_type, COUNT(*) as cnt,
               MAX(COALESCE(metadata->>'source_title', metadata->>'book_title')) as source_title,
               MAX(metadata->>'author') as author
        FROM rag_chunks
        WHERE collection = :coll AND chunk_type IN ('book', 'secondary_book')
        GROUP BY source_id, chunk_type
        ORDER BY source_title, source_id
        """
    )
    rows = engine.connect().execute(query, {"coll": collection}).fetchall()
    return [(r[0], r[1], r[2], r[3], r[4]) for r in rows]


_CHUNK_TYPE_MAP = {
    "summary": "chapter_summary",  # map user-facing "summary" to DB value
}


def _get_chunk_ids(
    engine,
    collection: str,
    book_dir: str | None,
    chunk_types: list[str] | None,
    chunk_type: str | None,  # single type from --chunk-type
    chunk_ids: list[str] | None,
) -> list[str]:
    """Resolve chunk_ids to delete: from explicit list or from Postgres query."""
    if chunk_ids:
        return [cid for cid in chunk_ids if isinstance(cid, str) and cid.strip()]

    db_type = _CHUNK_TYPE_MAP.get(chunk_type, chunk_type) if chunk_type else None
    filters: list = [chunks_table.c.collection == collection]

    if book_dir:
        filters.append(chunks_table.c.source_id == book_dir)
    if chunk_type:
        filters.append(chunks_table.c.chunk_type == db_type)
    elif chunk_types:
        filters.append(chunks_table.c.chunk_type.in_(chunk_types))

    if not book_dir and not chunk_type and not chunk_types:
        return []

    q = select(chunks_table.c.chunk_id).where(*filters)

    with engine.connect() as conn:
        rows = conn.execute(q).fetchall()
    return [r[0] for r in rows]


async def _delete_chunks(collection: str, chunk_ids: list[str]) -> None:
    """Delete chunks from Qdrant and Postgres."""
    if not chunk_ids:
        return

    client = QdrantClient(
        str(settings.qdrant_url),
        api_key=settings.qdrant_api_key,
        timeout=60.0,
    )
    point_uuids = [str(uuid5(NAMESPACE_DNS, cid)) for cid in chunk_ids]
    await client.delete_points(collection, point_uuids)

    mirror = ChunkMirrorRepository(get_engine())
    await mirror.delete_chunks(collection, chunk_ids)


def run_chunks_delete(
    assistant: str,
    chunk_types: list[str] | None = None,
    chunk_type: str | None = None,
    book: str | None = None,
    chunk_id: list[str] | None = None,
    dry_run: bool = False,
) -> None:
    """Run chunks:delete for the given assistant/collection."""
    console = Console()
    collection = _resolve_assistant(assistant.strip())
    mapped_types = _map_chunk_types(chunk_types or []) if chunk_types else None

    # No delete target: error (avoid DB connection)
    if book is None and not chunk_id and not chunk_type:
        console.print("[red]Error:[/] Provide --chunk-id, --chunk-type, or --book <source_id> to delete.")
        console.print("[dim]Use --book with no value to list books.[/]")
        return

    engine = get_engine()

    # If --chunk-id provided: delete those IDs (ignore --book, --chunk-type)
    if chunk_id:
        ids = _get_chunk_ids(engine, collection, None, None, None, chunk_id)
        if not ids:
            console.print("[yellow]No valid chunk_ids provided.[/]")
            return
        if dry_run:
            console.print(f"[bold blue]Dry run: would delete {len(ids)} chunks[/]")
            console.print(f"[dim]Collection: {collection}[/]")
            for cid in ids[:10]:
                console.print(f"  [dim]{cid}[/]")
            if len(ids) > 10:
                console.print(f"  [dim]... and {len(ids) - 10} more[/]")
            return
        console.print(f"[bold orange1]WARNING:[/] [orange1]About to delete {len(ids)} chunks from Qdrant and Postgres.[/]")
        try:
            reply = console.input("[dim]Confirm deletion? [y/N]: [/]").strip().lower()
        except EOFError:
            reply = "n"
        if reply != "y" and reply != "yes":
            console.print("[dim]Aborted.[/]")
            return
        try:
            asyncio.run(_delete_chunks(collection, ids))
            console.print(f"[green]Deleted {len(ids)} chunks from Qdrant and Postgres.[/]")
        except Exception as e:
            console.print(f"[red]Error:[/] {e}")
            raise SystemExit(1) from e
        return

    # If --book with no value: list books and exit (only book/secondary_book types)
    if book is not None and isinstance(book, str) and book.strip() == "":
        rows = _list_books(engine, collection)
        if not rows:
            console.print(f"[yellow]No books found for collection '{collection}'.[/]")
            return

        by_source: dict[str, list[tuple[str, int, str | None, str | None]]] = {}
        for src, ct, cnt, source_title, author in rows:
            by_source.setdefault(src, []).append((ct, cnt, source_title, author))

        console.print(f"[bold blue]Books (book, secondary_book) in collection '{collection}':[/]")
        console.print()
        for src in sorted(by_source.keys(), key=lambda s: (by_source[s][0][2] or s, s)):
            parts = by_source[src]
            total = sum(p[1] for p in parts)
            title = parts[0][2] if parts[0][2] else src
            author_str = f" – {parts[0][3]}" if parts[0][3] else ""
            console.print(f"  [yellow]{title}[/]{author_str}")
            console.print(f"    [dim]source_id:[/] {src}")
            for ct, cnt, _, _ in parts:
                console.print(f"    [dim]{ct}:[/] {cnt} chunks")
            console.print(f"    [dim]Total:[/] {total} chunks")
            console.print()
        console.print("[dim]Use --book <source_id> to delete chunks for a specific book.[/]")
        return

    # Resolve chunk_ids to delete (--chunk-type and/or --book with value)
    book_dir = book.strip() if book and isinstance(book, str) else None
    ids = _get_chunk_ids(
        engine,
        collection,
        book_dir,
        mapped_types if not chunk_type else None,
        chunk_type,
        chunk_id,
    )

    if not ids:
        desc = f"chunk_type={chunk_type}" if chunk_type else f"book '{book}'"
        if chunk_type and book_dir:
            desc = f"{desc} for book '{book_dir}'"
        console.print(f"[yellow]No chunks to delete for {desc}.[/]")
        if mapped_types:
            console.print(f"[dim]Filter: chunk_types={mapped_types}[/]")
        return

    if dry_run:
        console.print(f"[bold blue]Dry run: would delete {len(ids)} chunks[/]")
        console.print(f"[dim]Collection: {collection}[/]")
        if chunk_type:
            console.print(f"[dim]Chunk type: {chunk_type}[/]")
        if book_dir:
            console.print(f"[dim]Book/source_id: {book_dir}[/]")
        for cid in ids[:10]:
            console.print(f"  [dim]{cid}[/]")
        if len(ids) > 10:
            console.print(f"  [dim]... and {len(ids) - 10} more[/]")
        return

    console.print(f"[bold orange1]WARNING:[/] [orange1]About to delete {len(ids)} chunks from Qdrant and Postgres.[/]")
    try:
        reply = console.input("[dim]Confirm deletion? [y/N]: [/]").strip().lower()
    except EOFError:
        reply = "n"
    if reply != "y" and reply != "yes":
        console.print("[dim]Aborted.[/]")
        return
    try:
        asyncio.run(_delete_chunks(collection, ids))
        console.print(f"[green]Deleted {len(ids)} chunks from Qdrant and Postgres.[/]")
    except Exception as e:
        console.print(f"[red]Error:[/] {e}")
        console.print("[dim]Ensure Qdrant and Postgres are running.[/]")
        raise SystemExit(1) from e
