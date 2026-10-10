from __future__ import annotations

from app.retrieval.chains.authentic_concept_explain import MAX_LEXICON_CHARS
from app.retrieval.utils.embedding_budget import MAX_EMBED_CHUNK_CHARS


def test_lexicon_chars_match_shared_embed_budget():
    assert MAX_LEXICON_CHARS == MAX_EMBED_CHUNK_CHARS
