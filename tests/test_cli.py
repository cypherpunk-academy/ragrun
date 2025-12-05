"""Tests for the ragrun CLI."""
from __future__ import annotations

from pathlib import Path

import pytest

from ragrun import cli as cli_module


class DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class DummyClient:
    def __init__(self, *args, **kwargs):
        self.posts = []
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def post(self, url, json):
        self.posts.append({"url": url, "json": json})
        payload = {
            "ingestion_id": f"ing-{len(self.posts)}",
            "ingested": len(json["chunks"]),
            "requested": len(json["chunks"]),
            "duplicates": 0,
        }
        return DummyResponse(payload)


def test_cli_ingest_batches(monkeypatch, tmp_path: Path):
    sample = tmp_path / "chunks.jsonl"
    sample.write_text(
        '{"id": "c1", "text": "a", "metadata": {"author": "A", "source_id": "s", "chunk_id": "c1", '
        '"chunk_type": "book", "content_hash": "h1", "created_at": "2024-01-01T00:00:00Z", '
        '"updated_at": "2024-01-01T00:00:00Z", "language": "de", "tags": []}}\n'
        '{"id": "c2", "text": "b", "metadata": {"author": "A", "source_id": "s", "chunk_id": "c2", '
        '"chunk_type": "book", "content_hash": "h2", "created_at": "2024-01-01T00:00:00Z", '
        '"updated_at": "2024-01-01T00:00:00Z", "language": "de", "tags": []}}\n',
        encoding="utf-8",
    )

    factory_calls = []

    def fake_client(*args, **kwargs):
        factory_calls.append({"args": args, "kwargs": kwargs})
        return DummyClient(*args, **kwargs)

    monkeypatch.setattr(cli_module.httpx, "Client", fake_client)

    exit_code = cli_module.main(
        [
            "ingest",
            "--file",
            str(sample),
            "--collection",
            "books",
            "--batch-size",
            "1",
            "--api",
            "http://localhost:9999",
        ]
    )

    assert exit_code == 0
    assert factory_calls and factory_calls[0]["kwargs"]["timeout"] == 30.0

