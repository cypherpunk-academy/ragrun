"""Startup-Registrierung der App-Tools.

Aufgerufen aus app/main.py (`lifespan`) → `app.state.app_tool_registry`.
"""
from __future__ import annotations

from app.tools.registry import AppToolRegistry


def register_all_tools() -> AppToolRegistry:
    registry = AppToolRegistry()
    registry.discover()
    return registry
