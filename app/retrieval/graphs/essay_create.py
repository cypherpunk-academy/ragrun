"""Essay generation graph: 7-step pitch-driven essay with dense retrieval context."""
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
from app.retrieval.models import EssayCreateResult, EssayCreateStepInfo
from app.retrieval.utils.retrievers import build_context, dense_retrieve
from app.retrieval.utils.retry import retry_async

logger = logging.getLogger(__name__)
SENTENCE_END_RE = re.compile(r"[.!?][\"'”»\)\]]?\s*$")


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


def _parse_manifest_value(manifest_text: str, key: str) -> str | None:
    """Minimal YAML-ish parser for top-level 'key: value'."""
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


def _resolve_step_prompt_file(assistant_dir: Path, step: int) -> str:
    prompts_dir = assistant_dir / "prompts" / "essays"
    if not prompts_dir.exists():
        raise ValueError(f"Missing prompts directory: {prompts_dir}")
    matches = sorted([p.name for p in prompts_dir.glob(f"step{step}_*_draft.prompt")])
    if not matches:
        raise ValueError(f"Missing prompt for step {step}. Expected: {prompts_dir}/step{step}_*_draft.prompt")
    if len(matches) > 1:
        raise ValueError(f"Ambiguous prompt for step {step}: {matches} (keep exactly one)")
    return matches[0]


def _render_template(template: str, *, vars: Mapping[str, str]) -> str:
    out = template
    for k, v in vars.items():
        out = out.replace(f"{{{k}}}", v)
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


async def run_essay_create_graph(
    *,
    assistant: str,
    essay_slug: str,
    essay_title: str,
    pitch_steps: Sequence[str],
    k: int,
    embedding_client: EmbeddingClient,
    qdrant_client: QdrantClient,
    chat_client: DeepSeekClient,
    verbose: bool = False,
) -> EssayCreateResult:
    assistant = (assistant or "").strip()
    if not assistant:
        raise ValueError("assistant is required")
    if not essay_slug or not str(essay_slug).strip():
        raise ValueError("essay_slug is required")
    if not essay_title or not str(essay_title).strip():
        raise ValueError("essay_title is required")
    if not pitch_steps or len(pitch_steps) != 7:
        raise ValueError("pitch_steps must contain exactly 7 strings")

    assistant_dir = _resolve_assistant_dir(assistant)
    if not assistant_dir.exists():
        raise ValueError(f"Assistant directory not found: {assistant_dir}")

    collection = _resolve_collection(assistant_dir)

    system_prompt_path = assistant_dir / "prompts" / "essays" / "instruction.prompt"
    system_prompt = _load_text_or_throw(system_prompt_path).strip()

    essay_draft = ""
    steps_info: list[EssayCreateStepInfo] = []

    for step in range(1, 8):
        pitch_text = str(pitch_steps[step - 1] or "").strip()
        if not pitch_text:
            raise ValueError(f"pitch_steps[{step}] is empty")

        # Dense retrieval against primary books (chunk_type=book)
        hits = await dense_retrieve(
            query=pitch_text,
            k=int(k),
            worldview=None,
            book_types=["primary"],
            collection=collection,
            embedding_client=embedding_client,
            qdrant_client=qdrant_client,
        )
        context_text, _refs = build_context(hits, max_chars=12000)

        prompt_file = _resolve_step_prompt_file(assistant_dir, step)
        template_path = assistant_dir / "prompts" / "essays" / prompt_file
        template = _load_text_or_throw(template_path)

        user_prompt = _render_template(
            template,
            vars={
                "essay_title": str(essay_title),
                "text": pitch_text,
                "context": context_text,
                "essay_draft": essay_draft,
            },
        ).strip()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        step_text, _prompt_messages = await _chat_with_retry(
            chat_client,
            messages,
            temperature=0.4,
            max_tokens=650,
            operation=f"essay_create_step_{step}",
            retries=3,
            min_chars=400,
            verbose=verbose,
        )

        if essay_draft.strip():
            essay_draft = f"{essay_draft.rstrip()}\n\n{step_text.strip()}"
        else:
            essay_draft = step_text.strip()

        steps_info.append(EssayCreateStepInfo(step=step, prompt_file=prompt_file))

    return EssayCreateResult(
        assistant=assistant,
        essay_slug=str(essay_slug),
        essay_title=str(essay_title),
        final_text=essay_draft.strip(),
        steps=steps_info,
    )

