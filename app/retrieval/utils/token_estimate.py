"""Grobe Token-Schätzung für die Kontext-Anzeige im Client (Welle 5b).

Kein Tokenizer-Dependency — ausreichend für eine UI-Fortschrittsanzeige,
nicht für Abrechnung/Billing (dafür: `usage_metadata` aus dem LLM-Call).
"""
from __future__ import annotations

from typing import Any, Sequence

_CHARS_PER_TOKEN = 4

# DeepSeek V4 — Filo §11.4.
DEEPSEEK_CONTEXT_LIMIT_TOKENS = 1_000_000

_SYSTEM_PROMPT_TOKENS_ESTIMATE = 800


def estimate_tokens(text: str | None) -> int:
    """~4 Zeichen pro Token (grober Deutsch/Englisch-Mix-Durchschnitt)."""
    if not text:
        return 0
    return max(0, len(text) // _CHARS_PER_TOKEN)


def estimate_context_tokens(
    *,
    history_rows: Sequence[dict[str, Any]],
    citations: Sequence[dict[str, Any]] | None = None,
    linked_document_content: str | None = None,
    context_paragraph_text: str | None = None,
) -> int:
    """Summiert Systemprompt + geladene Turns + Zitat-Chunks + Bezugstext."""
    total = _SYSTEM_PROMPT_TOKENS_ESTIMATE
    for row in history_rows:
        total += estimate_tokens(row.get("user_message"))
        total += estimate_tokens(row.get("assistant_message"))
    for citation in citations or []:
        total += estimate_tokens(citation.get("text"))
    total += estimate_tokens(linked_document_content)
    total += estimate_tokens(context_paragraph_text)
    return total
