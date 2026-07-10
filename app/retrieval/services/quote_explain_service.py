"""Service for quote explanation: retrieve from primary book + lecture, explain via LLM."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import yaml

from app.config import settings
from app.infra.deepseek_client import DeepSeekClient
from app.infra.embedding_client import EmbeddingClient
from app.infra.qdrant_client import QdrantClient
from app.retrieval.models import RetrievedSnippet
from app.retrieval.utils.reference_evaluator import evaluate_chunk_relevance
from app.retrieval.utils.retrievers import build_context, dense_retrieve
from app.retrieval.services.action_prompt_service import load_assistant_embedding_prefixes
from app.debug_agent_log import agent_log

logger = logging.getLogger(__name__)

K_BOOKS = 4
K_LECTURES = 4
K_TOTAL = 8
# Explanation is stored as its own chunk (Chunk B), quote as separate Chunk A.
# 420 LLM output tokens × ~3.7 chars/token ≈ 1550 chars → ~422 e5-tokens incl.
# "passage: " prefix, safely under 512 embedding token limit.
MAX_EXPLANATION_TOKENS = 420


def _resolve_assistant_dir(assistant: str) -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    configured = Path(settings.assistants_root)
    assistants_root = configured if configured.is_absolute() else (repo_root / configured)
    return assistants_root / assistant


def _load_manifest(assistant: str) -> dict[str, Any]:
    """Load assistant-manifest.yaml and return parsed dict."""
    manifest_path = _resolve_assistant_dir(assistant) / "assistant-manifest.yaml"
    if not manifest_path.is_file():
        raise ValueError(f"Assistant manifest not found: {manifest_path}")
    text = manifest_path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    return data if isinstance(data, dict) else {}


def _load_quote_explain_prompt(*, language: str = "de-DE") -> str:
    prompts_dir = Path(__file__).resolve().parents[1] / "prompts"
    if language.startswith("en"):
        prompt_path = prompts_dir / "quote_explain_en.prompt"
    else:
        prompt_path = prompts_dir / "quote_explain.prompt"
    return prompt_path.read_text(encoding="utf-8").strip()


def _build_system_content(*, writing_style: str, language: str = "de-DE") -> str:
    """Sprache, Länge und optional writing-style aus dem Assistant-Manifest."""
    base = (
        "Du erklärst Zitate aus philosophischen und geisteswissenschaftlichen Texten."
    )
    if language.startswith("en"):
        lang_rule = (
            "Das Zitat bleibt im Original (Englisch); die Erklärung schreibst du auf Deutsch "
            "(ca. 200–300 Wörter)."
        )
    else:
        lang_rule = (
            "Antworte auf Deutsch. Halte die Erklärung prägnant (ca. 200–300 Wörter)."
        )
    parts: list[str] = []
    style = writing_style.strip()
    if style:
        parts.append(style)
    parts.extend([base, lang_rule])
    return "\n\n".join(parts)


def _build_quote_explain_prompt(
    *,
    quote: str,
    context: str,
    language: str = "de-DE",
    writing_style: str = "",
) -> list[Mapping[str, str]]:
    template = _load_quote_explain_prompt(language=language)
    user_content = template.format(quote=quote.strip(), context=context)
    system_content = _build_system_content(
        writing_style=writing_style, language=language
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


async def explain_quote(
    *,
    quote: str,
    assistant: str = "philo-von-freisinn",
    language: str = "de-DE",
    embedding_client: EmbeddingClient,
    qdrant_client: QdrantClient,
    chat_client: DeepSeekClient,
) -> dict[str, Any]:
    """
    Retrieve chunks from primary book + lecture, generate explanation, evaluate references.

    Returns a chunk-shaped dict: {text: str, metadata: dict} with text = quote + explanation.
    """
    quote = (quote or "").strip()
    if not quote:
        raise ValueError("quote is required")

    # region agent log
    agent_log(
        location="quote_explain_service.py:explain_quote:entry",
        message="explain_quote started",
        data={"assistant": assistant, "quote_len": len(quote)},
        hypothesis_id="A",
    )
    # endregion

    manifest = _load_manifest(assistant)
    collection = manifest.get("rag-collection") or assistant
    if not isinstance(collection, str):
        collection = assistant
    writing_style = manifest.get("writing-style")
    if not isinstance(writing_style, str):
        writing_style = ""

    _prefix_passage, query_prefix = load_assistant_embedding_prefixes(assistant)

    # Retrieve: 4 from primary books, 4 from lectures (talk/talk_summary)
    hits_books = await dense_retrieve(
        query=quote,
        k=K_BOOKS,
        worldview=None,
        book_types=["primary"],
        collection=collection,
        embedding_client=embedding_client,
        qdrant_client=qdrant_client,
        query_prefix=query_prefix,
    )
    hits_lectures = await dense_retrieve(
        query=quote,
        k=K_LECTURES,
        worldview=None,
        book_types=["talk", "talk_summary"],
        collection=collection,
        embedding_client=embedding_client,
        qdrant_client=qdrant_client,
        query_prefix=query_prefix,
    )

    all_hits: list[RetrievedSnippet] = list(hits_books) + list(hits_lectures)
    # region agent log
    agent_log(
        location="quote_explain_service.py:explain_quote:after_retrieve",
        message="retrieval complete",
        data={
            "books_hits": len(hits_books),
            "lectures_hits": len(hits_lectures),
            "collection": collection,
        },
        hypothesis_id="C-D",
    )
    # endregion
    if not all_hits:
        logger.warning("No chunks retrieved for quote; using empty context")

    context_str, _ = build_context(all_hits, max_chars=8000)
    if not context_str.strip():
        context_str = "(Kein Kontext verfügbar.)"

    messages = _build_quote_explain_prompt(
        quote=quote,
        context=context_str,
        language=language,
        writing_style=writing_style,
    )
    # region agent log
    agent_log(
        location="quote_explain_service.py:explain_quote:before_main_chat",
        message="calling main DeepSeek chat",
        data={"max_tokens": MAX_EXPLANATION_TOKENS},
        hypothesis_id="A",
    )
    # endregion
    explanation = await chat_client.chat(
        messages,
        temperature=0.3,
        max_tokens=MAX_EXPLANATION_TOKENS,
        _debug_caller="quote_explain_main",
    )
    explanation = explanation.content.strip()
    # region agent log
    agent_log(
        location="quote_explain_service.py:explain_quote:after_main_chat",
        message="main DeepSeek chat ok",
        data={"explanation_len": len(explanation)},
        hypothesis_id="A",
    )
    # endregion

    # Evaluate chunk relevance
    references: list[dict[str, Any]] = []
    if all_hits:
        # region agent log
        agent_log(
            location="quote_explain_service.py:explain_quote:before_ref_eval",
            message="calling reference evaluation",
            data={"hit_count": len(all_hits)},
            hypothesis_id="B",
        )
        # endregion
        references = await evaluate_chunk_relevance(
            generated_text=explanation,
            retrieved_chunks=all_hits,
            llm=chat_client,
            max_chunks=K_TOTAL,
        )
        # region agent log
        agent_log(
            location="quote_explain_service.py:explain_quote:after_ref_eval",
            message="reference evaluation done",
            data={"reference_count": len(references)},
            hypothesis_id="B",
        )
        # endregion

    # Extract chunk_ids for event recording
    chunk_ids: list[str] = []
    for hit in all_hits:
        payload = hit.payload
        if isinstance(payload, Mapping):
            inner = payload.get("payload")
            p = inner if isinstance(inner, Mapping) else payload
            cid = p.get("chunk_id")
            if isinstance(cid, str) and cid.strip():
                chunk_ids.append(cid.strip())

    # Build metadata from first hit for chunk_id/source_id/segment_id pattern
    metadata: dict[str, Any] = {
        "references": references,
    }
    if all_hits:
        payload = all_hits[0].payload
        if isinstance(payload, Mapping):
            inner = payload.get("payload")
            p = inner if isinstance(inner, Mapping) else payload
            for key in ("chunk_id", "source_id", "segment_id"):
                val = p.get(key)
                if isinstance(val, str):
                    metadata[key] = val

    return {
        "text": quote,
        "explanation": explanation,
        "metadata": metadata,
        "chunk_ids": chunk_ids,
        "collection": collection,
    }
