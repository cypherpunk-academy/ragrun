"""update_document — Vorschlag für eine gezielte Änderung am Arbeitsdokument.

Contract §4.3. execution: client — ragrun validiert nur strukturell
(Operation bekannt, Pflichtfelder vorhanden, Größenlimit). Der Patch
selbst wird clientseitig angewendet (`applyDocumentUpdate.ts`), daher
parst dieser Handler den Dokumentbaum bewusst NICHT.
"""
from __future__ import annotations

from app.tools.limits import MAX_DOCUMENT_CHARS
from app.tools.types import ToolContext, ToolResult

_OPERATIONS = {
    "update_paragraph": ("paragraph_id", "content"),
    "update_section": ("heading_path", "content"),
    "update_heading": ("heading_path", "content"),
    "insert_paragraph_after": ("paragraph_id", "content"),
    "delete_paragraph": ("paragraph_id",),
}

TOOL_ID = "update_document"


async def run(ctx: ToolContext, args: dict) -> ToolResult:
    operation = args.get("operation")
    if operation not in _OPERATIONS:
        raise ValueError(f"update_document: unbekannte operation {operation!r}.")

    for field in _OPERATIONS[operation]:
        if not args.get(field):
            raise ValueError(f"update_document: '{field}' fehlt für operation={operation!r}.")

    content = args.get("content")
    if content is not None and len(content) > MAX_DOCUMENT_CHARS:
        raise ValueError(
            f"update_document: 'content' überschreitet MAX_DOCUMENT_CHARS ({MAX_DOCUMENT_CHARS})."
        )

    return ToolResult(
        result_key="suggested_document_update",
        payload={
            "document_id": ctx.linked_document_id,
            "operation": operation,
            "paragraph_id": args.get("paragraph_id"),
            "heading_path": args.get("heading_path"),
            "content": content,
            "summary_for_chat": args.get("summary_for_chat", ""),
        },
    )
