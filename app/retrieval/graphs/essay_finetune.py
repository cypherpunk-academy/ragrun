"""Essay tune part graph: modify an existing essay part with custom instructions."""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from app.config import settings
from app.infra.deepseek_client import DeepSeekClient
from app.infra.embedding_client import EmbeddingClient
from app.infra.qdrant_client import QdrantClient
from app.retrieval.models import EssayCompletionResult
from app.retrieval.prompts.essay_completion import (
    build_header_prompt,
    build_tune_part_prompt,
)
from app.retrieval.utils.retrievers import build_context, dense_retrieve
from app.retrieval.utils.retry import retry_async

logger = logging.getLogger(__name__)
SENTENCE_END_RE = re.compile(r'[.!?][\"\'\u201c\u201d\u2019\u00BB\)\]]?\s*$')


MOOD_NAMES = {
    1: "okkult",
    2: "transzendental",
    3: "mystisch",
    4: "empirisch",
    5: "voluntaristisch",
    6: "logistisch",
    7: "gnostisch",
}


class RetryableCompletionError(RuntimeError):
    """Retryable error when the model response is too short or incomplete."""


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


def _resolve_assistant_dir(assistant: str) -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    configured = Path(settings.assistants_root)
    assistants_root = configured if configured.is_absolute() else (repo_root / configured)
    return assistants_root / assistant


def _load_text_or_throw(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Missing required file: {path}") from exc
    trimmed = text.strip()
    if not trimmed:
        raise ValueError(f"Required file is empty: {path}")
    return text


def _load_soul_mood_style(mood_index: int, mood_name: str) -> str:
    """Load style.md for a soul mood."""
    repo_root = Path(__file__).resolve().parents[3]
    configured = Path(settings.assistants_root)
    assistants_root = configured if configured.is_absolute() else (repo_root / configured)
    
    mood_dir = assistants_root / "sigrid-von-gleich" / "soul-moods" / f"{mood_index}_{mood_name}"
    style_path = mood_dir / "style.md"
    
    try:
        style_text = style_path.read_text(encoding="utf-8")
        trimmed = style_text.strip()
        if not trimmed:
            logger.warning(f"style.md is empty for {mood_index}_{mood_name}")
            return ""
        return trimmed
    except OSError as exc:
        logger.warning(f"Could not load style.md for {mood_index}_{mood_name}: {exc}")
        return ""


def _parse_manifest_value(manifest_text: str, key: str) -> str | None:
    needle = f"{key}:"
    for raw in manifest_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(needle):
            val = line[len(needle) :].strip().strip("'\"")
            return val or None
    return None


def _resolve_collection(assistant_dir: Path) -> str:
    manifest_path = assistant_dir / "assistant-manifest.yaml"
    try:
        manifest_text = manifest_path.read_text(encoding="utf-8")
    except OSError:
        return assistant_dir.name
    coll = _parse_manifest_value(manifest_text, "rag-collection")
    return coll or assistant_dir.name


def _load_previous_parts(
    assistant_dir: Path,
    essay_slug: str,
    current_mood_index: int,
    provided_parts: str | None = None,
) -> str:
    """Return formatted previous parts for parts 2-7.
    
    Args:
        assistant_dir: Path to assistant directory
        essay_slug: Essay slug/filename
        current_mood_index: Current mood index (1-7)
        provided_parts: Pre-formatted previous parts string from API caller
    
    Returns:
        Provided previous parts text or empty string for part 1.
    """
    if current_mood_index <= 1:
        return ""
    
    if provided_parts is None:
        return ""
    
    return provided_parts.strip()


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
    nonce = datetime.now(timezone.utc).isoformat()
    nonce_message = {"role": "system", "content": f"nonce: {nonce} (ignore)"}
    outbound_messages = [nonce_message, *messages]
    first_attempt = True

    def _log_prompt(prompt_messages: Sequence[Mapping[str, str]], is_retry: bool = False) -> None:
        if not verbose:
            return
        if is_retry:
            logger.warning("RETRYING PROMPT %s", operation)
        else:
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
        nonlocal first_attempt
        is_first = first_attempt
        _log_prompt(outbound_messages, is_retry=not is_first)
        first_attempt = False
        result = await client.chat(outbound_messages, temperature=temperature, max_tokens=max_tokens)

        if require_sentence_end and not SENTENCE_END_RE.search(result.strip()):
            completion_nonce = datetime.now(timezone.utc).isoformat()
            completion_messages = [
                {"role": "system", "content": f"nonce: {completion_nonce} (ignore)"},
                *messages,
                {"role": "assistant", "content": result},
                {"role": "user", "content": completion_instruction},
            ]
            _log_prompt(completion_messages, is_retry=not is_first)
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


async def run_essay_tune_part_graph(
    *,
    assistant: str,
    essay_slug: str,
    essay_title: str,
    mood_index: int,
    mood_name: str | None,
    current_text: str | None,
    current_header: str | None,
    previous_parts: str | None,
    modifications: str,
    k: int,
    embedding_client: EmbeddingClient,
    qdrant_client: QdrantClient,
    chat_client: DeepSeekClient,
    verbose: bool = False,
    llm_retries: int = 3,
) -> EssayCompletionResult:
    assistant = (assistant or "").strip()
    if not assistant:
        raise ValueError("assistant is required")
    if not essay_slug or not str(essay_slug).strip():
        raise ValueError("essay_slug is required")
    if not essay_title or not str(essay_title).strip():
        raise ValueError("essay_title is required")
    if mood_index < 1 or mood_index > 7:
        raise ValueError("mood_index must be between 1 and 7")
    if not current_text or not str(current_text).strip():
        raise ValueError("current_text is required")
    if not modifications or not str(modifications).strip():
        raise ValueError("modifications is required")

    assistant_dir = _resolve_assistant_dir(assistant)
    if not assistant_dir.exists():
        raise ValueError(f"Assistant directory not found: {assistant_dir}")

    collection = _resolve_collection(assistant_dir)
    
    # Resolve mood name if not provided
    final_mood_name = mood_name or MOOD_NAMES.get(mood_index, "")
    if not final_mood_name:
        raise ValueError(f"Could not resolve mood name for mood_index={mood_index}")
    
    # Load soul mood style
    soul_mood_style = _load_soul_mood_style(mood_index, final_mood_name)

    # Stage 1: Dense retrieve from primary books using current_text as query
    hits_primary = await dense_retrieve(
        query=current_text,
        k=int(k),
        worldview=None,
        book_types=["primary"],
        collection=collection,
        embedding_client=embedding_client,
        qdrant_client=qdrant_client,
    )
    primary_context, primary_refs = build_context(hits_primary, max_chars=12000)
    
    # Stage 2: Dense retrieve from secondary books using current_text as query
    hits_secondary = await dense_retrieve(
        query=current_text,
        k=int(k),
        worldview=None,
        book_types=["secondary"],
        collection=collection,
        embedding_client=embedding_client,
        qdrant_client=qdrant_client,
    )
    secondary_context, secondary_refs = build_context(hits_secondary, max_chars=12000)

    # Stage 3: Build tune prompt with modification instructions
    essay_parts_context = _load_previous_parts(
        assistant_dir, str(essay_slug), mood_index, provided_parts=previous_parts
    )
    
    tune_prompt = build_tune_part_prompt(
        current_text=current_text,
        modifications=modifications,
        primary_context=primary_context,
        secondary_context=secondary_context,
        part=str(mood_index),
        style=soul_mood_style,
        essay_parts=essay_parts_context,
        mood_index=mood_index,
        mood_name=final_mood_name,
    )
    
    # Stage 4: Generate revised text with LLM
    revised_text, _ = await _chat_with_retry(
        chat_client,
        tune_prompt,
        temperature=0.3,
        max_tokens=650,
        operation="essay_tune_part",
        retries=llm_retries,
        min_chars=200,
        verbose=verbose,
        require_sentence_end=True,
    )
    
    # Combine refs for all_books_refs
    all_books_refs = primary_refs + secondary_refs

    # Stage 5: Generate header for revised text
    final_header_prompt = build_header_prompt(
        assistant=assistant,
        essay_title=str(essay_title),
        mood_index=int(mood_index),
        mood_name=final_mood_name,
        text=revised_text,
    )
    revised_header, _ = await _chat_with_retry(
        chat_client,
        final_header_prompt,
        temperature=0.3,
        max_tokens=50,
        operation="essay_tune_part_header",
        retries=llm_retries,
        min_chars=10,
        verbose=verbose,
        require_sentence_end=False,
    )
    revised_header = revised_header.strip()[:100]

    return EssayCompletionResult(
        assistant=assistant,
        essay_slug=str(essay_slug),
        essay_title=str(essay_title),
        mood_index=int(mood_index),
        mood_name=final_mood_name,
        header=current_header or revised_header,
        draft_header=revised_header,
        draft_text=current_text.strip(),
        verification_report="",
        revised_header=revised_header,
        revised_text=revised_text.strip(),
        verify_refs=primary_refs,
        all_books_refs=all_books_refs,
        graph_event_id=None,
    )
