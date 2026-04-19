"""Problem-solver graph: automated Socratic dialog to resolve open problems.

Flow:
  1. Retrieve broad context once (primary + secondary + quotes + concepts)
  2. Retrieve primary-only context once (for Socrates rounds)
  3. Loop up to MAX_STEPS, alternating Solver / Socrates turns:
     - Solver (steps 1, 3, 5, 7): uses broad context, proposes solution
     - Socrates (steps 2, 4, 6): uses primary context, checks critically
  4. End early if Socrates agrees ("Ich stimme zu."), or after step 7.

Yields SSE-ready dicts: {type: "start"|"digest"|"done", ...}
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, AsyncGenerator

from app.config import settings
from app.infra.deepseek_client import DeepSeekClient
from app.infra.embedding_client import EmbeddingClient
from app.infra.qdrant_client import QdrantClient
from app.retrieval.services.action_prompt_service import (
    _lemma_lookup_multi_begrif,
    load_assistant_instruction,
)
from app.retrieval.utils.retrievers import build_context, dense_retrieve, hybrid_retrieve

logger = logging.getLogger(__name__)

MAX_STEPS = 7
MAX_TOKENS_PER_STEP = 300
DIGEST_MAX_CHARS = 600

_SENTENCE_END_RE = re.compile(r"[.!?]")
_AGREEMENT_RE = re.compile(
    r"^[\s\u201e\u201c\"\u00bb\u00ab']*(ich stimme zu)",
    re.IGNORECASE,
)


def _actions_root() -> Path:
    return Path(__file__).resolve().parents[1] / "actions"


def _load_prompt_template(filename: str) -> str:
    path = _actions_root() / "problem-solver" / filename
    return path.read_text(encoding="utf-8").strip()


def _fill(template: str, **slots: str) -> str:
    """Fill {placeholder} slots; unknown placeholders become empty string."""

    class _DefaultDict(dict):
        def __missing__(self, key: str) -> str:
            return ""

    return template.format_map(_DefaultDict(slots))


def _extract_digest(text: str) -> str:
    """Return first sentence of *text*, capped at DIGEST_MAX_CHARS."""
    m = _SENTENCE_END_RE.search(text)
    sentence = text[: m.end()].strip() if m else text.strip()
    return sentence[:DIGEST_MAX_CHARS]


def _is_agreement(text: str) -> bool:
    return bool(_AGREEMENT_RE.match(text.strip()))


def _format_history(history: list[dict[str, str]]) -> str:
    if not history:
        return "(kein bisheriger Verlauf)"
    parts = []
    for entry in history:
        label = "Solver" if entry["role"] == "solver" else "Sokrates"
        parts.append(f"[{label}] {entry['text']}")
    return "\n\n".join(parts)


async def _retrieve_broad(
    *,
    query: str,
    collection: str,
    embedding_client: EmbeddingClient,
    qdrant_client: QdrantClient,
) -> dict[str, str]:
    """Retrieve primary + secondary + quotes + concepts. Returns slot → text."""
    k = 6
    primary_hits = await hybrid_retrieve(
        query=query,
        k_dense=k,
        k_sparse=k if settings.use_hybrid_retrieval else 0,
        k_fused=k,
        worldview=None,
        book_types=["book"],
        collection=collection,
        embedding_client=embedding_client,
        qdrant_client=qdrant_client,
    )
    secondary_hits = await dense_retrieve(
        query=query,
        k=4,
        worldview=None,
        book_types=["secondary_book", "talk", "explanation"],
        collection=collection,
        embedding_client=embedding_client,
        qdrant_client=qdrant_client,
    )
    quote_hits = await dense_retrieve(
        query=query,
        k=2,
        worldview=None,
        book_types=["quote"],
        collection=collection,
        embedding_client=embedding_client,
        qdrant_client=qdrant_client,
    )
    concept_hits = await _lemma_lookup_multi_begrif(
        user_prompt=query,
        collection_name=collection,
    )
    primary_ctx, _ = build_context(primary_hits)
    secondary_ctx, _ = build_context(secondary_hits)
    quotes_ctx, _ = build_context(quote_hits)
    concepts_ctx, _ = build_context(concept_hits)
    return {
        "primary": primary_ctx or "(keine primären Quellen)",
        "secondary": secondary_ctx or "(keine sekundären Quellen)",
        "quotes": quotes_ctx or "(keine Zitate)",
        "concepts": concepts_ctx or "(keine Begriffe)",
    }


async def _retrieve_primary_only(
    *,
    query: str,
    collection: str,
    embedding_client: EmbeddingClient,
    qdrant_client: QdrantClient,
) -> str:
    """Retrieve primary sources only (for Socrates rounds)."""
    k = 6
    hits = await hybrid_retrieve(
        query=query,
        k_dense=k,
        k_sparse=k if settings.use_hybrid_retrieval else 0,
        k_fused=k,
        worldview=None,
        book_types=["book"],
        collection=collection,
        embedding_client=embedding_client,
        qdrant_client=qdrant_client,
    )
    ctx, _ = build_context(hits)
    return ctx or "(keine primären Quellen)"


async def run_problem_solver(
    *,
    problem_text: str,
    assistant_slug: str,
    collection: str,
    embedding_client: EmbeddingClient,
    qdrant_client: QdrantClient,
    chat_client: DeepSeekClient,
) -> AsyncGenerator[dict[str, Any], None]:
    """Async generator yielding SSE-ready dicts for the problem-solver dialog."""
    yield {"type": "start"}

    solver_template = _load_prompt_template("solver.prompt")
    socrates_template = _load_prompt_template("socrates.prompt")
    try:
        instruction = load_assistant_instruction(assistant_slug)
    except FileNotFoundError:
        instruction = ""

    broad = await _retrieve_broad(
        query=problem_text,
        collection=collection,
        embedding_client=embedding_client,
        qdrant_client=qdrant_client,
    )
    primary_ctx = await _retrieve_primary_only(
        query=problem_text,
        collection=collection,
        embedding_client=embedding_client,
        qdrant_client=qdrant_client,
    )

    dialog_history: list[dict[str, str]] = []
    step = 0
    agreed = False

    while step < MAX_STEPS and not agreed:
        # --- Solver turn ---
        step += 1
        solver_prompt = _fill(
            solver_template,
            problem_text=problem_text,
            dialog_history=_format_history(dialog_history),
            primary=broad["primary"],
            secondary=broad["secondary"],
            quotes=broad["quotes"],
            concepts=broad["concepts"],
        )
        solver_response = await chat_client.chat(
            [
                {"role": "system", "content": (instruction + "\n\n" + solver_prompt).strip()},
                {"role": "user", "content": "Formuliere deinen Lösungsvorschlag."},
            ],
            temperature=0.4,
            max_tokens=MAX_TOKENS_PER_STEP,
        )
        dialog_history.append({"role": "solver", "text": solver_response})
        yield {
            "type": "digest",
            "step": step,
            "role": "solver",
            "digest": _extract_digest(solver_response),
        }

        if step >= MAX_STEPS:
            break

        # --- Socrates turn ---
        step += 1
        socrates_prompt = _fill(
            socrates_template,
            problem_text=problem_text,
            solver_response=solver_response,
            dialog_history=_format_history(dialog_history[:-1]),  # history before latest solver
            primary=primary_ctx,
            concepts=broad["concepts"],
        )
        socrates_response = await chat_client.chat(
            [
                {"role": "system", "content": (instruction + "\n\n" + socrates_prompt).strip()},
                {"role": "user", "content": "Prüfe den Vorschlag."},
            ],
            temperature=0.3,
            max_tokens=MAX_TOKENS_PER_STEP,
        )
        dialog_history.append({"role": "socrates", "text": socrates_response})
        yield {
            "type": "digest",
            "step": step,
            "role": "socrates",
            "digest": _extract_digest(socrates_response),
        }

        if _is_agreement(socrates_response):
            agreed = True

    final_text = dialog_history[-1]["text"] if dialog_history else ""
    yield {
        "type": "done",
        "agreed": agreed,
        "steps": step,
        "full_dialog": dialog_history,
        "final_answer": final_text,
    }
