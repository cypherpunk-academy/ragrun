"""Smoke test for check_rag_paragraphs_consistency report shape."""
from __future__ import annotations

from scripts.check_rag_paragraphs_consistency import run_checks


def test_run_checks_summary_keys(monkeypatch):
    class FakeResult:
        def mappings(self):
            return self

        def all(self):
            return [
                {
                    "source_id": "s1",
                    "segment_index": 1,
                    "segment_slug": "a",
                    "paragraph_number": 1,
                },
                {
                    "source_id": "s1",
                    "segment_index": 1,
                    "segment_slug": "a",
                    "paragraph_number": 2,
                },
            ]

    class FakeConn:
        def execute(self, *_args, **_kwargs):
            return FakeResult()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class FakeEngine:
        def connect(self):
            return FakeConn()

    monkeypatch.setattr(
        "scripts.check_rag_paragraphs_consistency.get_sync_engine",
        lambda: FakeEngine(),
    )
    report = run_checks()
    assert report["summary"]["ok"] is True
    assert report["summary"]["segments_checked"] == 1
