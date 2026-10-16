"""Integration tests for the SQLAlchemy-based mirror repository."""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.asyncio
async def test_repository_upsert_and_delete(tmp_path: Path):
    pytest.skip("vector_chunks mirror integration requires PostgreSQL (ARRAY columns)")
