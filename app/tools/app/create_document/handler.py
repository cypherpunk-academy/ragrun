"""create_document — legt einen neuen Arbeitstext-Vorschlag an.

Contract §4.1. execution: client — ragrun validiert nur, ragapp
materialisiert via `materializeDocument.ts` (`NoteRepository.create`).
"""
from __future__ import annotations

from app.tools.limits import MAX_DOCUMENT_CHARS
from app.tools.types import ToolContext, ToolResult

TOOL_ID = "create_document"


async def run(ctx: ToolContext, args: dict) -> ToolResult:
    title = (args.get("title") or "").strip()
    content = args.get("content") or ""

    if not title:
        raise ValueError("create_document: 'title' darf nicht leer sein.")
    if not content.strip():
        raise ValueError("create_document: 'content' darf nicht leer sein.")
    if len(content) > MAX_DOCUMENT_CHARS:
        raise ValueError(
            f"create_document: 'content' überschreitet MAX_DOCUMENT_CHARS ({MAX_DOCUMENT_CHARS})."
        )

    return ToolResult(
        result_key="suggested_document",
        payload={
            "title": title,
            "content": content,
            "summary_for_chat": "Neuer Arbeitstext angelegt.",
        },
    )
