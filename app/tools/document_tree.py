"""Read-only Python-Spiegel von ragapp/src/data/lib/documentTree.ts — nur Parsing.

Wird ausschließlich von `read_blocks` benutzt, um Adressen (`paragraph_id` /
`heading_path`) im vollen `linked_document_content` aufzulösen. Patches
(`update_document`) werden clientseitig angewendet (`applyDocumentUpdate.ts`) —
dieses Modul serialisiert/patcht bewusst nichts.

Parsing-Regeln müssen mit documentTree.ts in Sync bleiben (Contract §2.1–2.2):
`#` = Titel (genau eine Zeile), `##` = Kapitel, `###` = Unterabschnitt,
Absatz = Textblock getrennt durch Leerzeile(n). Keine Tabellen im MVP.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$")


@dataclass(slots=True)
class DocumentParagraph:
    id: str
    text: str


@dataclass(slots=True)
class DocumentSubsection:
    heading: str
    heading_path: list[str]
    paragraphs: list[DocumentParagraph] = field(default_factory=list)


@dataclass(slots=True)
class DocumentSection:
    heading: str
    heading_path: list[str]
    paragraphs: list[DocumentParagraph] = field(default_factory=list)
    children: list[DocumentSubsection] = field(default_factory=list)


@dataclass(slots=True)
class DocumentTree:
    title: str
    sections: list[DocumentSection] = field(default_factory=list)


def _split_paragraph_blocks(block: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n{2,}", block) if p.strip()]


def parse_document_tree(content: str) -> DocumentTree:
    lines = content.replace("\r\n", "\n").split("\n")
    title = "Ohne Titel"
    body_start_idx = 0

    for i, raw_line in enumerate(lines):
        trimmed = raw_line.strip()
        m = _HEADING_RE.match(trimmed)
        if m and m.group(1) == "#":
            title = m.group(2).strip()
            body_start_idx = i + 1
            break
        if trimmed:
            break  # Nicht-leere Zeile vor erstem '#' — kein Titel gefunden.

    sections: list[DocumentSection] = []
    chapter_index = 0
    current = DocumentSection(heading="", heading_path=[])
    current_sub: DocumentSubsection | None = None
    buffer: list[str] = []
    para_counter = 0

    def has_content(s: DocumentSection) -> bool:
        return bool(s.heading) or bool(s.paragraphs) or bool(s.children)

    def flush_paragraphs() -> None:
        nonlocal buffer, para_counter
        text = "\n".join(buffer)
        buffer = []
        target = current_sub if current_sub is not None else current
        for block in _split_paragraph_blocks(text):
            para_counter += 1
            target.paragraphs.append(
                DocumentParagraph(id=f"ch{chapter_index}.p{para_counter}", text=block)
            )

    for line in lines[body_start_idx:]:
        trimmed = line.strip()
        m = _HEADING_RE.match(trimmed)

        if m and m.group(1) == "##":
            flush_paragraphs()
            if has_content(current):
                sections.append(current)
            chapter_index += 1
            para_counter = 0
            current_sub = None
            current = DocumentSection(heading=trimmed, heading_path=[trimmed])
            continue
        if m and m.group(1) == "###":
            flush_paragraphs()
            current_sub = DocumentSubsection(
                heading=trimmed, heading_path=[*current.heading_path, trimmed]
            )
            current.children.append(current_sub)
            continue
        if m and m.group(1) == "#":
            continue  # Titel ist genau eine Zeile — weitere '#'-Zeilen werden ignoriert.
        buffer.append(line)

    flush_paragraphs()
    if has_content(current):
        sections.append(current)

    return DocumentTree(title=title, sections=sections)


def find_paragraph(tree: DocumentTree, paragraph_id: str) -> DocumentParagraph | None:
    for section in tree.sections:
        for p in section.paragraphs:
            if p.id == paragraph_id:
                return p
        for sub in section.children:
            for p in sub.paragraphs:
                if p.id == paragraph_id:
                    return p
    return None


def find_by_heading_path(
    tree: DocumentTree, heading_path: list[str]
) -> DocumentSection | DocumentSubsection | None:
    """Doppelte `###`-Titel disambiguieren nur über vollständiges heading_path mit Parent-`##`."""
    if not heading_path:
        return None
    section = next((s for s in tree.sections if s.heading == heading_path[0]), None)
    if section is None:
        return None
    if len(heading_path) == 1:
        return section
    return next((c for c in section.children if c.heading == heading_path[1]), None)
