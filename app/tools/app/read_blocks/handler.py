"""read_blocks — liest vollen Text einzelner Absätze/Abschnitte.

Contract §4.2. execution: client, schreibt NICHT in die DB — liefert
Volltext für eine interne Tool-Runde. Löst Adressen gegen
`ctx.document_content` (= `linked_document_content`) auf.
"""
from __future__ import annotations

from app.tools.document_tree import find_by_heading_path, find_paragraph, parse_document_tree
from app.tools.types import ToolContext, ToolResult

TOOL_ID = "read_blocks"


async def run(ctx: ToolContext, args: dict) -> ToolResult:
    addresses = args.get("addresses") or []
    if not ctx.document_content:
        return ToolResult(result_key="document_blocks", payload={"blocks": []})

    tree = parse_document_tree(ctx.document_content)
    blocks: list[dict] = []

    for address in addresses:
        paragraph_id = address.get("paragraph_id")
        heading_path = address.get("heading_path")

        if paragraph_id:
            paragraph = find_paragraph(tree, paragraph_id)
            if paragraph is not None:
                blocks.append({"paragraph_id": paragraph.id, "content": paragraph.text})
            continue

        if heading_path:
            node = find_by_heading_path(tree, heading_path)
            if node is not None:
                content = "\n\n".join(p.text for p in node.paragraphs)
                blocks.append({"heading_path": heading_path, "content": content})

    return ToolResult(result_key="document_blocks", payload={"blocks": blocks})
