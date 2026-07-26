"""AppToolRegistry — Discovery, Schema-Bündel fürs Modell, Invoke.

Bezug: ragrun-app-tools-architecture.md §6–7. Discovery: Glob
`app/tools/app/*/tool-manifest.yaml`. Registrierung: app/main.py →
`app.state.app_tool_registry` (siehe index.py).
"""
from __future__ import annotations

import importlib
import json
import logging
from pathlib import Path
from typing import Any, Awaitable, Callable

import yaml

from app.tools.types import ToolContext, ToolManifest, ToolResult

logger = logging.getLogger(__name__)

Handler = Callable[[ToolContext, dict[str, Any]], Awaitable[ToolResult]]

_APP_TOOLS_PACKAGE = "app.tools.app"


def _default_tools_dir() -> Path:
    return Path(__file__).resolve().parent / "app"


def _load_manifest(manifest_path: Path) -> ToolManifest:
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid tool-manifest.yaml: {manifest_path}")

    schema_path = manifest_path.parent / "schema.json"
    schema: dict[str, Any] = {}
    if schema_path.is_file():
        schema = json.loads(schema_path.read_text(encoding="utf-8"))

    availability = data.get("availability") or {}
    return ToolManifest(
        id=data["id"],
        label=data.get("label", data["id"]),
        description=data.get("description", ""),
        category=data.get("category", "app-document"),
        execution=data.get("execution", "client"),
        result_key=data["result_key"],
        schema=schema,
        requires_linked_document=availability.get("requires_linked_document"),
        modes=list(availability.get("modes") or []),
    )


class AppToolRegistry:
    def __init__(self) -> None:
        self._manifests: dict[str, ToolManifest] = {}
        self._handlers: dict[str, Handler] = {}

    def discover(self, base_dir: Path | None = None) -> None:
        """Lädt alle `tool-manifest.yaml` unter `app/tools/app/*/` und bindet `handler.py`."""
        base = base_dir or _default_tools_dir()
        for manifest_path in sorted(base.glob("*/tool-manifest.yaml")):
            try:
                manifest = _load_manifest(manifest_path)
            except Exception:
                logger.exception("Failed to load tool manifest %s", manifest_path)
                continue

            self._manifests[manifest.id] = manifest

            handler_path = manifest_path.parent / "handler.py"
            if handler_path.is_file():
                module_name = f"{_APP_TOOLS_PACKAGE}.{manifest_path.parent.name}.handler"
                module = importlib.import_module(module_name)
                self._handlers[manifest.id] = module.run

            logger.info("Registered app tool: %s (execution=%s)", manifest.id, manifest.execution)

    def list_tools(
        self, *, mode: str | None = None, linked_document_id: str | None = None
    ) -> list[ToolManifest]:
        return [
            m
            for m in self._manifests.values()
            if m.is_available(mode=mode, linked_document_id=linked_document_id)
        ]

    def schemas_for_llm(self, available: list[ToolManifest]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": m.id,
                    "description": m.description,
                    "parameters": m.schema or {"type": "object", "properties": {}},
                },
            }
            for m in available
        ]

    async def invoke(self, tool_id: str, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        if tool_id not in self._manifests:
            raise KeyError(f"Unknown app tool: {tool_id!r}")
        handler = self._handlers.get(tool_id)
        if handler is None:
            raise RuntimeError(f"App tool {tool_id!r} has no handler")
        result = await handler(ctx, args)
        result.tool_id = tool_id
        return result
