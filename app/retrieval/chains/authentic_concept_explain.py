"""Chain for authentic concept explanation: Steiner-first -> retrieve -> verify -> lexicon."""
from __future__ import annotations

import logging
import re
import uuid
from functools import lru_cache
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence, Any

from app.config import settings
from app.infra.deepseek_client import DeepSeekClient
from app.infra.embedding_client import EmbeddingClient
from app.infra.qdrant_client import QdrantClient
from app.retrieval.models import AuthenticConceptExplainResult, RetrievedSnippet
from app.retrieval.prompts.authentic_concept_explain import (
    build_steiner_lexicon_prompt,
    build_steiner_prior_prompt,
    build_steiner_verify_query_prompt,
    build_steiner_verify_prompt,
)
from app.retrieval.services.graph_event_recorder import GraphEventRecorder
from app.retrieval.utils.reference_evaluator import evaluate_chunk_relevance
from app.retrieval.utils.retrievers import build_context, dense_retrieve, rerank_by_embedding
from app.retrieval.utils.retry import retry_async

logger = logging.getLogger(__name__)
SENTENCE_END_RE = re.compile(r'[.!?][\"\'\u201c\u201d\u2019\u00BB\)\]]?\s*$')


class RetryableCompletionError(RuntimeError):
    """Retryable error when the model response is too short or incomplete."""


def _extract_section_b(verification_report: str) -> str:
    """
    Extract section b) ("Ungestützte/zu starke Behauptungen") from the verifier text.
    Expected format:
      a) ...
      b) ...
      c) ...
      Bewertung: N/10
    """
    text = (verification_report or "").strip()
    if not text:
        return ""

    # Capture from "b)" until the next section header ("c)" or "Bewertung:")
    m = re.search(r"(?is)\bb\)\s*(.*?)(?=\n\s*c\)\s*|\n\s*bewertung\s*:)", text)
    if m:
        return m.group(1).strip()

    # Fallback: capture from "b)" to end
    m2 = re.search(r"(?is)\bb\)\s*(.*)$", text)
    return m2.group(1).strip() if m2 else ""
def _resolve_philo_assistant_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    configured = Path(settings.assistants_root)
    assistants_root = configured if configured.is_absolute() else (repo_root / configured)
    return assistants_root / "philo-von-freisinn"


def _parse_primary_books_ids(manifest_text: str) -> list[str]:
    """
    Minimal YAML-ish parser for our `assistant-manifest.yaml` shape.
    We avoid adding a YAML dependency (PyYAML) to ragrun.
    """
    ids: list[str] = []
    in_primary = False
    for raw in manifest_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("primary-books:"):
            in_primary = True
            continue
        if in_primary and line.endswith(":") and not line.startswith("-"):
            # next top-level section begins (e.g. secondary-books:)
            break
        if in_primary and line.startswith("- "):
            value = line[2:].strip().strip("'\"")
            if value:
                ids.append(value)
    return ids


def _pretty_primary_book(book_id: str) -> str:
    """
    Convert ids like `Rudolf_Steiner#Die_Philosophie_der_Freiheit#4`
    into a human-readable title.
    """
    parts = book_id.split("#")
    if len(parts) >= 3:
        author = parts[0].replace("_", " ").strip()
        title = parts[1].replace("_", " ").strip()
        ga = parts[2].strip()
        if author and title and ga:
            return f"{author}: {title} (GA {ga})"
        if author and title:
            return f"{author}: {title}"
    return book_id.replace("_", " ").strip()


@lru_cache(maxsize=1)
def _load_primary_books_list_text() -> str:
    """
    Load and format the 'primary-books' from `assistant-manifest.yaml` as a bullet list.
    Returns an empty string if the manifest can't be read or contains no primary books.
    """
    manifest_path = _resolve_philo_assistant_dir() / "assistant-manifest.yaml"
    try:
        text = manifest_path.read_text(encoding="utf-8")
    except OSError:
        return ""

    ids = _parse_primary_books_ids(text)
    if not ids:
        return ""
    pretty = [_pretty_primary_book(x) for x in ids]
    return "\n".join(f"- {x}" for x in pretty if x)


def _is_retryable_llm_error(exc: Exception) -> bool:
    if isinstance(exc, RetryableCompletionError):
        return True
    message = str(exc).lower()
    if "deepseek returned empty content" in message:
        return True
    if "deepseek returned no choices" in message:
        return True
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    return bool(status_code and status_code >= 500)


def _extract_context_source(snippets: Sequence[RetrievedSnippet]) -> list[dict[str, Any]]:
    """
    Build a provenance-only context source list from retrieved snippets.

    We intentionally do NOT include chunk text here; only titles from rag_chunks.metadata:
    - source_title
    - segment_title
    """

    out: list[dict[str, Any]] = []
    for snip in snippets:
        outer: Mapping[str, Any] = snip.payload or {}
        inner = outer.get("payload")
        payload: Mapping[str, Any] = inner if isinstance(inner, Mapping) else outer

        chunk_id = payload.get("chunk_id") if isinstance(payload.get("chunk_id"), str) else None
        md = payload.get("metadata")
        metadata: Mapping[str, Any] = md if isinstance(md, Mapping) else {}

        source_title = metadata.get("source_title") if isinstance(metadata.get("source_title"), str) else None
        segment_title = metadata.get("segment_title") if isinstance(metadata.get("segment_title"), str) else None

        out.append(
            {
                "chunk_id": chunk_id,
                "source_title": source_title,
                "segment_title": segment_title,
            }
        )
    return out


async def _chat_with_retry(
    client: DeepSeekClient,
    messages: Sequence[Mapping[str, str]],
    *,
    temperature: float,
    max_tokens: int,
    operation: str,
    retries: int = 3,
    min_chars: int | None = None,
    require_sentence_end: bool = True,
    completion_instruction: str = "Schließe den Text sauber ab. Setze einen klaren Schlusssatz.",
    verbose: bool = False,
) -> tuple[str, list[Mapping[str, str]]]:
    """Call DeepSeek with best-effort completion retry to avoid truncated text."""
    nonce = datetime.now(timezone.utc).isoformat()
    nonce_message = {"role": "system", "content": f"nonce: {nonce} (ignore)"}
    outbound_messages = [nonce_message, *messages]

    def _log_prompt(prompt_messages: Sequence[Mapping[str, str]]) -> None:
        if not verbose:
            return
        formatted = "\n\n".join(
            f"[{m.get('role', 'unknown')}]\n{m.get('content', '')}" for m in prompt_messages
        )
        logger.warning("Prompt for %s:\n%s", operation, formatted)

    def _is_incomplete(text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return True
        if require_sentence_end and not SENTENCE_END_RE.search(stripped):
            return True
        if min_chars is not None and len(stripped) < min_chars:
            return True
        return False

    async def _run_once() -> tuple[str, list[Mapping[str, str]]]:
        _log_prompt(outbound_messages)
        result = await client.chat(outbound_messages, temperature=temperature, max_tokens=max_tokens)

        if require_sentence_end and not SENTENCE_END_RE.search(result.strip()):
            completion_nonce = datetime.now(timezone.utc).isoformat()
            completion_messages = [
                {"role": "system", "content": f"nonce: {completion_nonce} (ignore)"},
                *messages,
                {"role": "assistant", "content": result},
                {"role": "user", "content": completion_instruction},
            ]
            _log_prompt(completion_messages)
            completion = await client.chat(
                completion_messages, temperature=temperature, max_tokens=max_tokens
            )
            combined = f"{result.rstrip()} {completion.lstrip()}".strip()
            if _is_incomplete(combined):
                raise RetryableCompletionError("Completion produced an incomplete response")
            return combined, completion_messages

        if _is_incomplete(result):
            raise RetryableCompletionError("Response too short or incomplete")

        return result, outbound_messages

    return await retry_async(
        _run_once,
        retries=max(0, int(retries)),
        base_delay=1.0,
        max_delay=8.0,
        jitter=0.2,
        retry_on=_is_retryable_llm_error,
        logger=logger,
        operation=operation,
    )


def _should_widen(snippets: Sequence[RetrievedSnippet], k_final: int, min_chars: int = 400) -> bool:
    if len(snippets) < k_final:
        return True
    total_chars = sum(len(s.text) for s in snippets)
    return total_chars < min_chars


@dataclass(slots=True)
class VerifyRetrievalConfig:
    k_base: int = 18
    k_final: int = 8
    widen_to: int = 48


async def run_authentic_concept_explain_chain(
    *,
    concept: str,
    collection: str,
    embedding_client: EmbeddingClient,
    qdrant_client: QdrantClient,
    chat_client: DeepSeekClient,
    cfg: VerifyRetrievalConfig | None = None,
    event_recorder: GraphEventRecorder | None = None,
    verbose: bool = False,
    llm_retries: int = 3,
) -> AuthenticConceptExplainResult:
    if not concept or not concept.strip():
        raise ValueError("concept is required")

    cfg = cfg or VerifyRetrievalConfig()
    graph_event_id = uuid.uuid4()
    graph_name = "authentic_concept_explain"
    recorder = event_recorder or GraphEventRecorder()

    async def _record_event(step: str, **kwargs: object) -> None:
        if verbose:
            # Log the exact payload we send to persistence for easy debugging.
            # Note: this may include long texts (context/prompts/responses).
            try:
                logger.warning(
                    "[graph_event] graph=%s graph_event_id=%s step=%s payload=%s",
                    graph_name,
                    str(graph_event_id),
                    step,
                    {
                        "graph_event_id": str(graph_event_id),
                        "graph_name": graph_name,
                        "step": step,
                        "concept": concept,
                        **kwargs,
                    },
                )
            except Exception:
                logger.warning(
                    "[graph_event] graph=%s graph_event_id=%s step=%s payload_repr=%r",
                    graph_name,
                    str(graph_event_id),
                    step,
                    kwargs,
                )
        await recorder.record_event(
            graph_event_id=graph_event_id,
            graph_name=graph_name,
            step=step,
            concept=concept,
            **kwargs,
        )

    # Step 1.1: Steiner prior (no retrieval)
    primary_books_list = _load_primary_books_list_text()
    prior_prompt = build_steiner_prior_prompt(concept=concept, primary_books_list=primary_books_list)
    steiner_prior_text, prior_prompt_messages = await _chat_with_retry(
        chat_client,
        prior_prompt,
        temperature=0.4,
        max_tokens=520,
        operation="steiner_prior",
        retries=llm_retries,
        min_chars=900,
        verbose=verbose,
    )
    await _record_event(
        "steiner_prior",
        prompt_messages=prior_prompt_messages,
        response_text=steiner_prior_text,
    )

    # Step 1.2: Retrieve Steiner-ish chunks (chunk_type=book) + verify against context
    raw_hits = await dense_retrieve(
        query=steiner_prior_text,
        k=cfg.k_base,
        worldview=None,
        book_types=["primary"],  # maps to chunk_type=book
        collection=collection,
        embedding_client=embedding_client,
        qdrant_client=qdrant_client,
    )
    reranked = await rerank_by_embedding(
        query=steiner_prior_text, snippets=raw_hits, embedding_client=embedding_client, k_final=cfg.k_final
    )
    if _should_widen(reranked, cfg.k_final):
        widened = await dense_retrieve(
            query=steiner_prior_text,
            k=cfg.widen_to,
            worldview=None,
            book_types=["primary"],
            collection=collection,
            embedding_client=embedding_client,
            qdrant_client=qdrant_client,
        )
        reranked = await rerank_by_embedding(
            query=steiner_prior_text, snippets=widened, embedding_client=embedding_client, k_final=cfg.k_final
        )

    verify_context, verify_refs = build_context(reranked)
    verify_sources = _extract_context_source(reranked)
    await _record_event(
        "steiner_verify_retrieval",
        query_text=steiner_prior_text,
        context_refs=verify_refs,
        context_source=verify_sources,
        retrieval_mode="dense",
        metadata={
            "k_base": cfg.k_base,
            "k_final": cfg.k_final,
            "widen_to": cfg.widen_to,
            "chunk_type": "book",
        },
        errors=["no_hits"] if not reranked else None,
    )
    if not reranked:
        raise ValueError("Verification retrieval returned no hits; aborting")

    verify_prompt = build_steiner_verify_prompt(
        steiner_prior_text=steiner_prior_text, context=verify_context
    )
    verification_report, verify_prompt_messages = await _chat_with_retry(
        chat_client,
        verify_prompt,
        temperature=0.1,
        max_tokens=700,
        operation="steiner_verify_reasoning",
        retries=llm_retries,
        min_chars=500,
        verbose=verbose,
    )
    await _record_event(
        "steiner_verify_reasoning",
        prompt_messages=verify_prompt_messages,
        response_text=verification_report,
        context_refs=verify_refs,
        context_source=verify_sources,
    )

    # Step 1.2b: Generate a short missing-support query from section (b) and retrieve extra context
    b_section = _extract_section_b(verification_report)
    missing_query_text = ""
    extra_context = ""
    extra_refs: list[str] = []
    extra_sources: list[dict[str, Any]] = []
    missing_reranked: list[RetrievedSnippet] = []
    if b_section:
        verify_query_prompt = build_steiner_verify_query_prompt(
            b_section=b_section, primary_books_list=primary_books_list
        )
        missing_query_text, verify_query_prompt_messages = await _chat_with_retry(
            chat_client,
            verify_query_prompt,
            temperature=0.1,
            max_tokens=180,
            operation="steiner_verify_query",
            retries=llm_retries,
            min_chars=40,
            verbose=verbose,
        )

        # Retrieve additional Steiner context with the generated query
        missing_hits = await dense_retrieve(
            query=missing_query_text,
            k=cfg.k_base,
            worldview=None,
            book_types=["primary"],
            collection=collection,
            embedding_client=embedding_client,
            qdrant_client=qdrant_client,
        )
        missing_reranked = await rerank_by_embedding(
            query=missing_query_text,
            snippets=missing_hits,
            embedding_client=embedding_client,
            k_final=cfg.k_final,
        )

        extra_context, extra_refs = build_context(missing_reranked)
        extra_sources = _extract_context_source(missing_reranked)
        await _record_event(
            "steiner_verify_missing_retrieval",
            query_text=missing_query_text,
            context_refs=extra_refs,
            context_source=extra_sources,
            retrieval_mode="dense",
            metadata={
                "k_base": cfg.k_base,
                "k_final": cfg.k_final,
                "chunk_type": "book",
                "source": "verify_b_section",
            },
            errors=["no_hits"] if not missing_reranked else None,
        )

    # Step 1.3: Final lexicon entry grounded in retrieved context + verification (+ optional extra context)
    combined_context = verify_context
    if extra_context:
        # Keep input size bounded; the model doesn't need unlimited context here.
        extra_trimmed = extra_context if len(extra_context) <= 8000 else extra_context[:8000] + "…"
        combined_context = (
            f"{verify_context}\n\n---\nZusätzlicher Kontext (aus Hinweisen in b)):\n{extra_trimmed}"
        )

    lexicon_prompt = build_steiner_lexicon_prompt(
        concept=concept, context=combined_context, verification_report=verification_report
    )
    lexicon_entry, lexicon_prompt_messages = await _chat_with_retry(
        chat_client,
        lexicon_prompt,
        temperature=0.3,
        max_tokens=520,
        operation="steiner_lexicon",
        retries=llm_retries,
        min_chars=900,
        verbose=verbose,
    )
    await _record_event(
        "steiner_lexicon",
        prompt_messages=lexicon_prompt_messages,
        response_text=lexicon_entry,
        context_refs=verify_refs,
        context_source=verify_sources,
        metadata={
            "chunk_words_min": settings.ace_chunk_min_words,
            "chunk_words_target": settings.ace_chunk_target_words,
            "chunk_words_max": settings.ace_chunk_max_words,
        },
    )

    # Evaluate references after final lexicon_entry is complete.
    all_retrieved_chunks: list[RetrievedSnippet] = list(reranked)
    if missing_reranked:
        all_retrieved_chunks.extend(missing_reranked)
    references = await evaluate_chunk_relevance(
        generated_text=lexicon_entry,
        retrieved_chunks=all_retrieved_chunks,
        llm=chat_client,
        max_chunks=20,
    )

    return AuthenticConceptExplainResult(
        concept=concept,
        steiner_prior_text=steiner_prior_text,
        verify_refs=verify_refs,
        verification_report=verification_report,
        lexicon_entry=lexicon_entry,
        references=references,
        graph_event_id=str(graph_event_id),
    )

