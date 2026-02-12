"""Reusable chunk reference evaluator for essay retrieval graphs."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from app.retrieval.models import RetrievedSnippet

logger = logging.getLogger(__name__)

DEFAULT_TEMPLATE = """
Du bewertest, welche abgerufenen Chunks für den generierten Text tatsächlich relevant waren. Antworte auf Deutsch.

Generated Text:
{generated_text}

Retrieved Chunks:
{chunks_list}

Gib für jeden relevanten Chunk an:
1. chunk_id
2. description: Formuliere den Inhalt des Chunks als direkte Aussage (auf Deutsch). Keine Meta-Phrasen wie „Der Chunk beschreibt“, „thematisiert“ oder „Dies entspricht der Kernthese“. Sage direkt, was der Fall ist.
3. Relevanz-Score (0.0–1.0)

Gib NUR ein JSON-Array mit Objekten zurück:
[
  {{
    "chunk_id": "string",
    "description": "string",
    "relevance": 0.0
  }}
]
""".strip()

_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")
_FALLBACK_DESCRIPTION = "Hohe semantische Aehnlichkeit zum finalen Text."
_GENERATED_TEXT_TOKEN = "__GENERATED_TEXT_TOKEN__"
_CHUNKS_LIST_TOKEN = "__CHUNKS_LIST_TOKEN__"


class ChatModel(Protocol):
    async def chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 300,
    ) -> str: ...


def _resolve_template_path() -> Path:
    retrieval_dir = Path(__file__).resolve().parents[1]
    return retrieval_dir / "prompts" / "templates" / "evaluate_chunk_relevance.prompt"


def _load_template() -> str:
    prompt_path = _resolve_template_path()
    try:
        text = prompt_path.read_text(encoding="utf-8").strip()
        if text:
            return text
    except OSError:
        logger.warning("Could not read reference evaluation template: %s", prompt_path)
    return DEFAULT_TEMPLATE


def _extract_chunk_id(payload: Mapping[str, Any]) -> str | None:
    inner = payload.get("payload")
    candidate = inner if isinstance(inner, Mapping) else payload
    chunk_id = candidate.get("chunk_id")
    return chunk_id if isinstance(chunk_id, str) and chunk_id.strip() else None


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return f"{value[: max(0, limit - 3)].rstrip()}..."


def _normalize_text(value: str) -> str:
    return _WHITESPACE_RE.sub(" ", value).strip()


def _format_reference_prompt(
    template: str,
    *,
    generated_text: str,
    chunks_list: str,
) -> str:
    """Safely format template placeholders without breaking on JSON braces."""
    protected = (
        template.replace("{generated_text}", _GENERATED_TEXT_TOKEN).replace(
            "{chunks_list}", _CHUNKS_LIST_TOKEN
        )
    )
    escaped = protected.replace("{", "{{").replace("}", "}}")
    return (
        escaped.replace(_GENERATED_TEXT_TOKEN, "{generated_text}")
        .replace(_CHUNKS_LIST_TOKEN, "{chunks_list}")
        .format(generated_text=generated_text, chunks_list=chunks_list)
    )


def _serialize_chunks(
    snippets: Sequence[RetrievedSnippet],
    *,
    max_chunks: int,
    max_text_chars: int = 500,
) -> tuple[str, list[str], list[tuple[str, float]]]:
    lines: list[str] = []
    allowed_chunk_ids: list[str] = []
    ranked_for_fallback: list[tuple[str, float]] = []

    for idx, snippet in enumerate(snippets[: max(1, max_chunks)], start=1):
        chunk_id = _extract_chunk_id(snippet.payload) or f"unknown-{idx}"
        normalized_text = _normalize_text(snippet.text)
        excerpt = _truncate(normalized_text, max_text_chars)
        lines.append(
            (
                f"{idx}. chunk_id={chunk_id}\n"
                f"score={snippet.score:.6f}\n"
                f"text={excerpt}"
            )
        )
        allowed_chunk_ids.append(chunk_id)
        ranked_for_fallback.append((chunk_id, float(snippet.score)))

    return "\n\n".join(lines), allowed_chunk_ids, ranked_for_fallback


def _parse_json_content(raw_content: str) -> list[Mapping[str, Any]]:
    stripped = raw_content.strip()
    if not stripped:
        return []

    candidates = [stripped]
    fence_match = _CODE_FENCE_RE.search(stripped)
    if fence_match:
        candidates.insert(0, fence_match.group(1).strip())

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, Mapping)]
        if isinstance(parsed, Mapping):
            refs = parsed.get("references")
            if isinstance(refs, list):
                return [item for item in refs if isinstance(item, Mapping)]
    return []


def _normalize_relevance(value: Any) -> float:
    try:
        rel = float(value)
    except (TypeError, ValueError):
        rel = 0.0
    rel = max(0.0, min(1.0, rel))
    return round(rel, 4)


def _normalize_description(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    normalized = _normalize_text(value)
    return _truncate(normalized, 220)


def _normalize_references(
    raw_references: Sequence[Mapping[str, Any]],
    *,
    allowed_chunk_ids: Sequence[str],
) -> list[dict[str, Any]]:
    allowed = set(allowed_chunk_ids)
    dedup: dict[str, dict[str, Any]] = {}

    for raw in raw_references:
        chunk_id = raw.get("chunk_id")
        if not isinstance(chunk_id, str):
            continue
        chunk_id = chunk_id.strip()
        if not chunk_id or chunk_id not in allowed:
            continue
        description = _normalize_description(raw.get("description"))
        if not description:
            continue
        relevance = _normalize_relevance(raw.get("relevance"))
        candidate = {
            "chunk_id": chunk_id,
            "description": description,
            "relevance": relevance,
        }
        previous = dedup.get(chunk_id)
        if previous is None or relevance > float(previous.get("relevance", 0.0)):
            dedup[chunk_id] = candidate

    ordered = sorted(dedup.values(), key=lambda item: item["relevance"], reverse=True)
    return ordered


def _fallback_references(ranked_chunks: Sequence[tuple[str, float]]) -> list[dict[str, Any]]:
    if not ranked_chunks:
        return []
    max_score = max(score for _, score in ranked_chunks) or 1.0
    out: list[dict[str, Any]] = []
    for chunk_id, score in sorted(ranked_chunks, key=lambda item: item[1], reverse=True)[:3]:
        normalized = max(0.05, min(1.0, score / max_score))
        out.append(
            {
                "chunk_id": chunk_id,
                "description": _FALLBACK_DESCRIPTION,
                "relevance": round(normalized, 4),
            }
        )
    return out


async def evaluate_chunk_relevance(
    *,
    generated_text: str,
    retrieved_chunks: Sequence[RetrievedSnippet],
    llm: ChatModel,
    max_chunks: int = 20,
) -> list[dict[str, Any]]:
    """Evaluate which retrieved chunks influenced the final generated text."""
    text = generated_text.strip()
    if not text or not retrieved_chunks:
        return []

    chunks_list, allowed_chunk_ids, ranked_for_fallback = _serialize_chunks(
        retrieved_chunks, max_chunks=max_chunks
    )
    if not chunks_list:
        return []

    template = _load_template()
    user_prompt = _format_reference_prompt(
        template,
        generated_text=_truncate(text, 5000),
        chunks_list=chunks_list,
    )
    messages = [
        {
            "role": "system",
            "content": "Return strictly valid JSON and no extra commentary.",
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    try:
        raw_response = await llm.chat(messages, temperature=0.0, max_tokens=1200)
    except Exception:
        logger.exception("Reference evaluation LLM call failed; using fallback references")
        return _fallback_references(ranked_for_fallback)

    parsed = _parse_json_content(raw_response)
    normalized = _normalize_references(parsed, allowed_chunk_ids=allowed_chunk_ids)
    if normalized:
        return normalized

    logger.warning("Reference evaluation returned no valid references; using fallback references")
    return _fallback_references(ranked_for_fallback)
