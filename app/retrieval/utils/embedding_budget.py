"""Shared limits for single embed chunks (e5 512 with passage prefix)."""
from __future__ import annotations

import re

# Same budget as quote_explanation / begriff lexicon (420 LLM tokens ≈ 1550 chars).
MAX_EMBED_CHUNK_TOKENS = 420
MAX_EMBED_CHUNK_CHARS = 1550
SENTENCE_END_RE = re.compile(r'[.!?][\"\'\u201c\u201d\u2019\u00BB\)\]]?\s*$')


def trim_to_embedding_budget(text: str, *, max_chars: int = MAX_EMBED_CHUNK_CHARS) -> str:
    """Trim text at the last sentence end within max_chars (1 item → 1 chunk)."""
    stripped = (text or "").strip()
    if len(stripped) <= max_chars:
        return stripped
    cut = stripped[:max_chars]
    best_end = -1
    for sep in (". ", "? ", "! ", ".\n", "?\n", "!\n"):
        pos = cut.rfind(sep)
        if pos != -1:
            best_end = max(best_end, pos + 1)
    if best_end > 0:
        return cut[:best_end].strip()
    return cut.rstrip() + "…"
