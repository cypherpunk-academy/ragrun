"""Essay completion graph: generate + verify + rewrite a single mood section."""
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
    build_authenticity_check_prompt,
    build_completion_prompt,
    build_header_prompt,
    build_length_adjustment_prompt,
    build_rewrite_prompt,
    build_verification_prompt,
)
from app.retrieval.utils.retrievers import build_context, dense_retrieve
from app.retrieval.utils.retry import retry_async

logger = logging.getLogger(__name__)
SENTENCE_END_RE = re.compile(r"[.!?][\"'”»\)\]]?\s*$")


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


def _resolve_soul_moods_path() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    configured = Path(settings.assistants_root)
    assistants_root = configured if configured.is_absolute() else (repo_root / configured)
    return assistants_root / "sigrid-von-gleich" / "soul-moods" / "seelenstimmungen_steiner.md"


def _load_text_or_throw(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Missing required file: {path}") from exc
    trimmed = text.strip()
    if not trimmed:
        raise ValueError(f"Required file is empty: {path}")
    return text


def _load_soul_mood_files(mood_index: int, mood_name: str) -> tuple[str, str]:
    """Load instruction.md and description.md for a soul mood."""
    repo_root = Path(__file__).resolve().parents[3]
    configured = Path(settings.assistants_root)
    assistants_root = configured if configured.is_absolute() else (repo_root / configured)
    
    mood_dir = assistants_root / "sigrid-von-gleich" / "soul-moods" / f"{mood_index}_{mood_name}"
    instruction_path = mood_dir / "instruction.md"
    description_path = mood_dir / "description.md"
    
    instruction = _load_text_or_throw(instruction_path)
    description = _load_text_or_throw(description_path)
    
    return instruction, description


def _count_tokens(text: str) -> int:
    """Estimate token count using word-based heuristic for German text."""
    if not text or not text.strip():
        return 0
    words = text.strip().split()
    return int(len(words) * 1.3)


def _needs_length_adjustment(text: str, force: bool) -> tuple[bool, str]:
    """Check if text needs length adjustment based on token count."""
    tokens = _count_tokens(text)
    if force:
        return True, "force=true"
    if tokens < 200:
        return True, f"too_short ({tokens} tokens)"
    if tokens > 300:
        return True, f"too_long ({tokens} tokens)"
    return False, "ok"


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


def _parse_primary_books_ids(manifest_text: str) -> list[str]:
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
            break
        if in_primary and line.startswith("- "):
            value = line[2:].strip().strip("'\"")
            if value:
                ids.append(value)
    return ids


def _pretty_primary_book(book_id: str) -> str:
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


def _load_primary_books_list_text(assistant_dir: Path) -> str:
    manifest_path = assistant_dir / "assistant-manifest.yaml"
    try:
        text = manifest_path.read_text(encoding="utf-8")
    except OSError:
        return ""
    ids = _parse_primary_books_ids(text)
    if not ids:
        return ""
    pretty = [_pretty_primary_book(x) for x in ids]
    return "\n".join(f"- {x}" for x in pretty if x)


def _parse_soul_moods(text: str) -> list[dict[str, str]]:
    moods: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    desc_lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        m = re.match(r"^##\s+(\d+)\.\s+(.+?)\s*\((.+?)\)\s*$", line)
        if m:
            if current:
                current["description"] = "\n".join(desc_lines).strip()
                moods.append(current)
            desc_lines = []
            current = {
                "index": m.group(1),
                "name": m.group(2).strip(),
                "planet": m.group(3).strip(),
                "description": "",
            }
            continue
        if current and line:
            desc_lines.append(line)
    if current:
        current["description"] = "\n".join(desc_lines).strip()
        moods.append(current)
    return moods


def _find_mood(moods: list[dict[str, str]], mood_index: int) -> dict[str, str]:
    for mood in moods:
        try:
            idx = int(mood.get("index", "0"))
        except ValueError:
            continue
        if idx == mood_index:
            return mood
    raise ValueError(f"Unknown mood index: {mood_index}")


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


async def run_essay_completion_graph(
    *,
    assistant: str,
    essay_slug: str,
    essay_title: str,
    mood_index: int,
    mood_name: str | None,
    current_text: str | None,
    current_header: str | None,
    k: int,
    embedding_client: EmbeddingClient,
    qdrant_client: QdrantClient,
    chat_client: DeepSeekClient,
    verbose: bool = False,
    llm_retries: int = 3,
    force: bool = False,
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

    assistant_dir = _resolve_assistant_dir(assistant)
    if not assistant_dir.exists():
        raise ValueError(f"Assistant directory not found: {assistant_dir}")

    collection = _resolve_collection(assistant_dir)
    
    # Resolve mood name if not provided
    final_mood_name = mood_name or MOOD_NAMES.get(mood_index, "")
    if not final_mood_name:
        raise ValueError(f"Could not resolve mood name for mood_index={mood_index}")
    
    # Load soul mood files
    soul_mood_instruction, soul_mood_description = _load_soul_mood_files(
        mood_index, final_mood_name
    )
    
    primary_books_list = _load_primary_books_list_text(assistant_dir)

    # Generate draft text using soul mood instruction as system prompt
    draft_prompt = [
        {"role": "system", "content": soul_mood_instruction},
        {
            "role": "user",
            "content": f"""Schreibe einen Essay-Abschnitt zum Thema: "{essay_title}"

Vorhandener Text (falls vorhanden):
{(current_text or '').strip() or '(leer)'}

Anforderungen:
- 200-300 Tokens (keine Stichpunkte)
- Keine Meta-Erklärungen
- Keine Überschriften im Text
- Klare, lesbare Prosa
""".strip(),
        },
    ]

    draft_text, _draft_messages = await _chat_with_retry(
        chat_client,
        draft_prompt,
        temperature=0.4,
        max_tokens=650,
        operation="essay_completion_draft",
        retries=llm_retries,
        min_chars=200,
        verbose=verbose,
        require_sentence_end=True,
    )

    # Token check and length adjustment
    needs_adjustment, reason = _needs_length_adjustment(draft_text, force)
    adjusted_text = draft_text
    
    if needs_adjustment:
        target_tokens = 250
        adjustment_prompt = build_length_adjustment_prompt(
            soul_mood_instruction=soul_mood_instruction,
            text=draft_text,
            target_tokens=target_tokens,
            reason=reason,
        )
        adjusted_text, _ = await _chat_with_retry(
            chat_client,
            adjustment_prompt,
            temperature=0.3,
            max_tokens=650,
            operation="essay_completion_adjust_length",
            retries=llm_retries,
            min_chars=200,
            verbose=verbose,
            require_sentence_end=True,
        )

    # Generate header
    header_prompt = build_header_prompt(
        assistant=assistant,
        essay_title=str(essay_title),
        mood_name=final_mood_name,
        text=adjusted_text,
    )
    draft_header, _ = await _chat_with_retry(
        chat_client,
        header_prompt,
        temperature=0.3,
        max_tokens=50,
        operation="essay_completion_header",
        retries=llm_retries,
        min_chars=10,
        verbose=verbose,
        require_sentence_end=False,
    )
    draft_header = draft_header.strip()[:100]

    # First Qdrant search: Primary books only
    hits_primary = await dense_retrieve(
        query=adjusted_text,
        k=int(k),
        worldview=None,
        book_types=["primary"],
        collection=collection,
        embedding_client=embedding_client,
        qdrant_client=qdrant_client,
    )
    primary_context, verify_refs = build_context(hits_primary, max_chars=12000)

    # Authenticity check with soul mood description
    authenticity_prompt = build_authenticity_check_prompt(
        draft_text=adjusted_text,
        primary_books_context=primary_context,
        primary_books_list=primary_books_list,
        soul_mood_description=soul_mood_description,
        mood_name=final_mood_name,
    )
    verification_report, _ = await _chat_with_retry(
        chat_client,
        authenticity_prompt,
        temperature=0.1,
        max_tokens=600,
        operation="essay_completion_authenticity",
        retries=llm_retries,
        min_chars=300,
        verbose=verbose,
        require_sentence_end=True,
    )

    # Second Qdrant search: All books (primary + secondary)
    hits_all = await dense_retrieve(
        query=adjusted_text,
        k=int(k),
        worldview=None,
        book_types=["primary", "secondary"],
        collection=collection,
        embedding_client=embedding_client,
        qdrant_client=qdrant_client,
    )
    all_books_context, all_books_refs = build_context(hits_all, max_chars=12000)

    # Final rewrite with all books context
    rewrite_prompt = build_rewrite_prompt(
        assistant=assistant,
        draft_text=adjusted_text,
        verification_report=verification_report,
        context=all_books_context,
    )
    revised_text, _ = await _chat_with_retry(
        chat_client,
        rewrite_prompt,
        temperature=0.3,
        max_tokens=650,
        operation="essay_completion_rewrite",
        retries=llm_retries,
        min_chars=200,
        verbose=verbose,
        require_sentence_end=True,
    )

    # Generate final header for revised text
    final_header_prompt = build_header_prompt(
        assistant=assistant,
        essay_title=str(essay_title),
        mood_name=final_mood_name,
        text=revised_text,
    )
    revised_header, _ = await _chat_with_retry(
        chat_client,
        final_header_prompt,
        temperature=0.3,
        max_tokens=50,
        operation="essay_completion_final_header",
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
        header=current_header or draft_header,
        draft_header=draft_header,
        draft_text=draft_text.strip(),
        adjusted_text=adjusted_text.strip(),
        verification_report=verification_report.strip(),
        revised_header=revised_header,
        revised_text=revised_text.strip(),
        verify_refs=verify_refs,
        all_books_refs=all_books_refs,
        graph_event_id=None,
    )
