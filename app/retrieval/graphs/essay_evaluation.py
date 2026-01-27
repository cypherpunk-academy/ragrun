"""Essay evaluation graph: score text against pitch criteria."""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Mapping, Sequence
from app.infra.deepseek_client import DeepSeekClient
from app.retrieval.models import EssayEvaluationResult
from app.retrieval.prompts.essay_evaluation import build_evaluation_prompt
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


async def _chat_with_retry(
    client: DeepSeekClient,
    messages: Sequence[Mapping[str, str]],
    *,
    temperature: float,
    max_tokens: int,
    operation: str,
    retries: int = 3,
    min_chars: int | None = None,
    require_sentence_end: bool = False,
    completion_instruction: str = "Gib nur das JSON aus.",
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


async def run_essay_evaluation_graph(
    *,
    assistant: str,
    essay_slug: str,
    essay_title: str,
    text: str,
    mood_index: int | None,
    mood_name: str | None,
    chat_client: DeepSeekClient,
    verbose: bool = False,
    llm_retries: int = 3,
) -> EssayEvaluationResult:
    assistant = (assistant or "").strip()
    if not assistant:
        raise ValueError("assistant is required")
    if not essay_slug or not str(essay_slug).strip():
        raise ValueError("essay_slug is required")
    if not essay_title or not str(essay_title).strip():
        raise ValueError("essay_title is required")
    if not text or not str(text).strip():
        raise ValueError("text is required")

    messages = build_evaluation_prompt(
        assistant=assistant,
        essay_title=str(essay_title),
        mood_index=mood_index,
        mood_name=mood_name,
        text=str(text).strip(),
    )

    response_text, _prompt_messages = await _chat_with_retry(
        chat_client,
        messages,
        temperature=0.1,
        max_tokens=500,
        operation="essay_evaluation",
        retries=llm_retries,
        min_chars=120,
        verbose=verbose,
    )

    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Evaluation JSON parse failed: {exc}") from exc

    overall_score = int(parsed.get("overall_score", 0))
    criteria_scores_raw = parsed.get("criteria_scores", {})
    criteria_scores: dict[str, int] = {}
    if isinstance(criteria_scores_raw, dict):
        for k, v in criteria_scores_raw.items():
            if isinstance(k, str):
                try:
                    criteria_scores[k] = int(v)
                except (TypeError, ValueError):
                    continue
    issues_raw = parsed.get("issues", [])
    issues = [str(x) for x in issues_raw] if isinstance(issues_raw, list) else []
    instruction = str(parsed.get("instruction", "")).strip()

    return EssayEvaluationResult(
        assistant=assistant,
        essay_slug=str(essay_slug),
        essay_title=str(essay_title),
        mood_index=mood_index,
        mood_name=mood_name,
        overall_score=overall_score,
        criteria_scores=criteria_scores,
        issues=issues,
        instruction=instruction,
        graph_event_id=None,
    )
