"""Typen für die App-Tool-Registry.

Bezug: ragrun-app-tools-architecture.md §4–5, filo-arbeitstext-contract.md §3–5.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Execution = Literal["client", "server"]


@dataclass(slots=True)
class ToolContext:
    """Kontext, den die Registry an jeden Handler übergibt.

    `document_content` = `linked_document_content` aus dem Request-Body — geht
    NICHT in den LLM-Prompt, steht aber `read_blocks` zur Verfügung (Contract §3).
    """

    user_id: str
    talk_id: str | None = None
    mode: str | None = None
    personality: str | None = None
    linked_document_id: str | None = None
    document_content: str | None = None
    document_outline: dict[str, Any] | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class ToolResult:
    """Ergebnis eines Handler-Laufs — landet als Eintrag in `tool_results` im SSE-`done`-Event."""

    result_key: str
    payload: dict[str, Any]
    tool_id: str | None = None


@dataclass(slots=True)
class ToolManifest:
    """Geparste `tool-manifest.yaml` + geladenes `schema.json`."""

    id: str
    label: str
    description: str
    category: str
    execution: Execution
    result_key: str
    schema: dict[str, Any]
    # None = keine Anforderung; True = braucht verknüpftes Dokument; False = darf KEIN verknüpftes Dokument haben.
    requires_linked_document: bool | None = None
    modes: list[str] = field(default_factory=list)

    def is_available(self, *, mode: str | None, linked_document_id: str | None) -> bool:
        if self.requires_linked_document is True and not linked_document_id:
            return False
        if self.requires_linked_document is False and linked_document_id:
            return False
        if self.modes and mode is not None and mode not in self.modes:
            return False
        return True
