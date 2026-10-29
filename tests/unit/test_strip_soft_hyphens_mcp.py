"""MCP responses strip Unicode soft hyphens (U+00AD)."""

from app.mcp_server.tools import _mcp_text, _sanitize_passage_payload, _snippet
from app.utils.text_utils import strip_soft_hyphens


def test_strip_soft_hyphens_removes_u00ad():
    assert strip_soft_hyphens("Entwicklungs\u00adkräfte") == "Entwicklungskräfte"
    assert strip_soft_hyphens("") == ""
    assert strip_soft_hyphens("plain") == "plain"


def test_mcp_snippet_strips_soft_hyphens():
    assert "\u00ad" not in _snippet("Wieder\u00aderkennung ist wichtig.")
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
