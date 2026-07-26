from __future__ import annotations

from app.retrieval.utils.embedding_budget import MAX_EMBED_CHUNK_CHARS, trim_to_embedding_budget


def test_trim_to_embedding_budget_keeps_short_text():
    text = "Kurzer Eintrag."
    assert trim_to_embedding_budget(text) == text


def test_trim_to_embedding_budget_truncates_at_sentence():
    long_body = "Wort " * 400
    text = f"Einleitung. {long_body}Schlusssatz."
    trimmed = trim_to_embedding_budget(text, max_chars=200)
    assert len(trimmed) <= 200
    assert trimmed.endswith(".")


def test_trim_to_embedding_budget_respects_max_embed_chunk_chars_default():
    text = "x" * (MAX_EMBED_CHUNK_CHARS + 500)
    trimmed = trim_to_embedding_budget(text)
    assert len(trimmed) <= MAX_EMBED_CHUNK_CHARS + 1  # ellipsis fallback adds one char
