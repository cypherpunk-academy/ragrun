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

logger = logging.getLogger(__name__)

K_BOOKS = 4
K_LECTURES = 4
K_TOTAL = 8
MAX_EXPLANATION_TOKENS = 600


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


def _build_quote_explain_prompt(
    *, quote: str, context: str, language: str = "de-DE"
) -> list[Mapping[str, str]]:
    template = _load_quote_explain_prompt(language=language)
    user_content = template.format(quote=quote.strip(), context=context)
    # Erklärung immer auf Deutsch; Zitat bleibt in Originalsprache
    system_content = "Antworte auf Deutsch. Halte die Erklärung prägnant (ca. 200–300 Wörter)."
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

    manifest = _load_manifest(assistant)
    collection = manifest.get("rag-collection") or assistant
    if not isinstance(collection, str):
        collection = assistant

    # Retrieve: 4 from primary books, 4 from lectures (talk/talk_summary)
    hits_books = await dense_retrieve(
        query=quote,
        k=K_BOOKS,
        worldview=None,
        book_types=["primary"],
        collection=collection,
        embedding_client=embedding_client,
        qdrant_client=qdrant_client,
    )
    hits_lectures = await dense_retrieve(
        query=quote,
        k=K_LECTURES,
        worldview=None,
        book_types=["talk", "talk_summary"],
        collection=collection,
        embedding_client=embedding_client,
        qdrant_client=qdrant_client,
    )

    all_hits: list[RetrievedSnippet] = list(hits_books) + list(hits_lectures)
    if not all_hits:
        logger.warning("No chunks retrieved for quote; using empty context")

    context_str, _ = build_context(all_hits, max_chars=8000)
    if not context_str.strip():
        context_str = "(Kein Kontext verfügbar.)"

    messages = _build_quote_explain_prompt(quote=quote, context=context_str, language=language)
    explanation = await chat_client.chat(
        messages,
        temperature=0.3,
        max_tokens=MAX_EXPLANATION_TOKENS,
    )
    explanation = explanation.strip()

    # Combined output: quote (original) + Erklärung (immer Deutsch)
    combined_text = f"{quote}\n\nErklärung:\n\n{explanation}"

    # Evaluate chunk relevance
    references: list[dict[str, Any]] = []
    if all_hits:
        references = await evaluate_chunk_relevance(
            generated_text=combined_text,
            retrieved_chunks=all_hits,
            llm=chat_client,
            max_chunks=K_TOTAL,
        )

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
        "text": combined_text,
        "metadata": metadata,
        "chunk_ids": chunk_ids,
        "collection": collection,
    }
