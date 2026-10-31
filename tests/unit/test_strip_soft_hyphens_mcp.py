"""MCP responses strip Unicode soft hyphens (U+00AD)."""

from app.mcp_server.tools import (
    _mcp_text,
    _raw_hit_to_mcp_item,
    _sanitize_passage_payload,
    _snippet,
)
from app.utils.text_utils import strip_soft_hyphens


def test_strip_soft_hyphens_removes_u00ad():
    assert strip_soft_hyphens("Entwicklungs\u00adkräfte") == "Entwicklungskräfte"
    assert strip_soft_hyphens("") == ""
    assert strip_soft_hyphens("plain") == "plain"


def test_mcp_snippet_strips_soft_hyphens():
    assert "\u00ad" not in _snippet("Wieder\u00adkennung ist wichtig.")
    assert _mcp_text("a\u00adb") == "ab"


def test_sanitize_passage_payload_strips_paragraph_text():
    payload = {
        "paragraphs": [{"text": "Sinn\u00adeswelt"}, {"text": "ok"}],
        "text": "Doppel\u00adtitel",
    }
    out = _sanitize_passage_payload(payload)
    assert out["paragraphs"][0]["text"] == "Sinneswelt"
    assert out["paragraphs"][1]["text"] == "ok"
    assert out["text"] == "Doppeltitel"


def test_raw_hit_to_mcp_item_returns_full_text_without_snippet():
    """Search hits expose the whole chunk; the old 240-char cut must not apply."""
    prefix = "A" * 240
    tail = " Hegel steht erst hier nach der alten Schnittstelle."
    full = prefix + tail
    item = _raw_hit_to_mcp_item(
        {
            "chunk_id": "c1",
            "chunk_type": "book",
            "score": 0.0323,
            "text": full,
            "snippet": prefix + "…",
            "title": "Die Philosophie der Freiheit",
            "segment_title": "Absatz 16",
        }
    )
    assert item["text"] == full
    assert "Hegel" in item["text"]
    assert "snippet" not in item
    assert item["score"] == 0.0323


def test_raw_hit_to_mcp_item_prefers_text_over_trimmed_snippet():
    item = _raw_hit_to_mcp_item(
        {
            "chunk_id": "c2",
            "chunk_type": "book",
            "score": 0.1,
            "text": "voller Chunk mit Soft\u00adhyphen",
            "snippet": "kurz…",
        }
    )
    assert item["text"] == "voller Chunk mit Softhyphen"
    assert "snippet" not in item


def test_raw_hit_to_mcp_item_falls_back_to_untrimmed_snippet_field():
    body = "B" * 300 + " Ende"
    item = _raw_hit_to_mcp_item(
        {
            "chunk_id": "c3",
            "chunk_type": "secondary_book",
            "score": 0.2,
            "snippet": body,
        }
    )
    assert item["text"] == body
    assert "snippet" not in item
