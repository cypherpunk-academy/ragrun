"""Tests for assistant-manifest-driven lexical redaction (hybrid sparse query)."""

import pytest

from app.retrieval.utils.retrievers import redact_lexical_phrases


@pytest.mark.parametrize(
    "text,phrases,expected",
    [
        (
            "Was sagt Steiner über moralische Technik",
            ["Rudolf Steiner", "Steiner"],
            "Was sagt über moralische Technik",
        ),
        (
            "Rudolf Steiner und Kant",
            ["Rudolf Steiner", "Steiner"],
            "und Kant",
        ),
        (
            "Kein Treffer hier",
            ["Rudolf Steiner", "Steiner"],
            "Kein Treffer hier",
        ),
        (
            "  Viele   Spaces   Steiner  ",
            ["Steiner"],
            "Viele Spaces",
        ),
    ],
)
def test_redact_lexical_phrases(text: str, phrases: list[str], expected: str) -> None:
    assert redact_lexical_phrases(text, phrases) == expected


def test_yaml_order_independent() -> None:
    """Longer phrase must win first even if YAML lists Steiner before Rudolf Steiner."""
    t = "Zitat von Rudolf Steiner und Steiner"
    assert redact_lexical_phrases(t, ["Steiner", "Rudolf Steiner"]) == "Zitat von und"


def test_whitespace_only_reverts_to_original() -> None:
    assert redact_lexical_phrases("Steiner", ["Steiner"]) == "Steiner"
