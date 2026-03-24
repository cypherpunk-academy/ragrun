"""Service for generate-prompt / execute-prompt (ASSISTANTS_CHAT_PLAN_V2)."""
from __future__ import annotations

import asyncio
import logging
import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml
from sqlalchemy import text

from app.config import settings
from app.db.async_session import get_async_sessionmaker
from app.infra.embedding_client import EmbeddingClient
from app.infra.qdrant_client import QdrantClient
from app.retrieval.models import RetrievedSnippet
from app.retrieval.utils.retrievers import (
    build_context,
    build_context_numbered,
    dense_retrieve,
    hybrid_retrieve,
    hybrid_retrieve_quote_parallel,
    sparse_retrieve,
)

logger = logging.getLogger(__name__)

# Max chars per chunk text in chunk_index_map; keep in sync with action_prompt.REFERENCE_TEXT_MAX_CHARS
CHUNK_TEXT_MAX_CHARS = 2000

_ARTICLE_PREFIX = re.compile(
    r"^(der|die|das|ein|eine|des|dem|den|einem|einer)\s+",
    re.IGNORECASE,
)


def _normalize_lemma(raw: str) -> str:
    """Normalize user prompt to lemma format (lowercase, no article, last word)."""
    cleaned = _ARTICLE_PREFIX.sub("", raw.strip()).lower()
    parts = cleaned.split()
    return parts[-1] if parts else cleaned


def _lemma_lookup_variants(lemma: str) -> list[str]:
    """Lemma variants for DB lookup (exact first, then singular -en)."""
    variants = [lemma]
    if len(lemma) > 3 and lemma.endswith("en"):
        variants.append(lemma[:-2])
    return variants


# Prompt state cache: prompt_id -> {state, expires_at}
# State includes: filled_prompt, instruction_prompt, query_results, citations_metadata,
# assistant_slug, action_id, collection_name, context_refs, retrieved_snippets
_PROMPT_STATE_CACHE: dict[str, dict[str, Any]] = {}
_PROMPT_CACHE_TTL_SECONDS = 1800  # 30 minutes


def _resolve_actions_root() -> Path:
    """Resolve actions directory (app/retrieval/actions)."""
    return Path(__file__).resolve().parents[1] / "actions"


def _resolve_assistants_root() -> Path:
    """Resolve assistants root (ragkeep/assistants)."""
    repo_root = Path(__file__).resolve().parents[3]
    configured = Path(settings.assistants_root)
    return configured if configured.is_absolute() else (repo_root / configured)


def load_action_manifest(action_id: str) -> dict[str, Any]:
    """Load action-manifest.yaml for the given action_id."""
    actions_root = _resolve_actions_root()
    manifest_path = actions_root / action_id / "action-manifest.yaml"
    if not manifest_path.is_file():
        raise ValueError(f"Action manifest not found: {manifest_path}")
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def load_action_prompt(action_id: str) -> str:
    """Load prompt.prompt template for the given action_id."""
    actions_root = _resolve_actions_root()
    prompt_path = actions_root / action_id / "prompt.prompt"
    if not prompt_path.is_file():
        raise ValueError(f"Action prompt not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8").strip()


def load_assistant_instruction(assistant_slug: str) -> str:
    """Load instruction.prompt for the assistant from ragkeep."""
    prompt_path = (
        _resolve_assistants_root()
        / assistant_slug
        / "prompts"
        / "instruction.prompt"
    )
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Assistant instruction not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8").strip()


def list_actions() -> list[dict[str, Any]]:
    """List all available actions (from action-manifest files)."""
    actions_root = _resolve_actions_root()
    out: list[dict[str, Any]] = []
    for manifest_path in actions_root.glob("*/action-manifest.yaml"):
        try:
            data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data.get("id"):
                out.append({
                    "id": data["id"],
                    "label": data.get("label", data["id"]),
                    "description": data.get("description", ""),
                    "requires_prompt": data.get("requires_prompt", True),
                })
        except Exception as exc:
            logger.warning("Failed to load manifest %s: %s", manifest_path, exc)
    return sorted(out, key=lambda a: a["id"])


async def _lemma_lookup_begriff_list(
    *,
    user_prompt: str,
    collection_name: str,
) -> list[RetrievedSnippet]:
    """Exact lemma lookup in Postgres rag_chunks for begriff_list. Returns matching chunks."""
    lemma = _normalize_lemma(user_prompt)
    if not lemma:
        return []

    async_session = get_async_sessionmaker()
    for variant in _lemma_lookup_variants(lemma):
        async with async_session() as session:
            row = await session.execute(
                text(
                    "SELECT chunk_id, text, metadata FROM rag_chunks "
                    "WHERE collection = :col "
                    "  AND chunk_type = 'begriff_list' "
                    "  AND LOWER(metadata->>'segment_title') = :lemma "
                    "  AND (worldviews IS NULL OR array_length(worldviews, 1) IS NULL) "
                    "LIMIT 5"
                ),
                {"col": collection_name, "lemma": variant},
            )
            rows = row.fetchall()
        if rows:
            out: list[RetrievedSnippet] = []
            for rec in rows:
                meta = dict(rec.metadata) if rec.metadata else {}
                meta["chunk_id"] = rec.chunk_id
                meta.setdefault("segment_title", variant)
                out.append(
                    RetrievedSnippet(
                        text=rec.text or "",
                        score=1.0,
                        payload={"payload": meta, "chunk_id": rec.chunk_id},
                    )
                )
            return out
    return []


async def _lemma_lookup_multi_begriff_list(
    *,
    user_prompt: str,
    collection_name: str,
) -> list[RetrievedSnippet]:
    """Multi-lemma lookup: find all terms from user_prompt that exist in begriff_list, return their chunks."""
    MIN_WORD_LEN = 3
    MAX_CHUNKS = 10

    async_session = get_async_sessionmaker()

    async with async_session() as session:
        row = await session.execute(
            text(
                "SELECT DISTINCT LOWER(metadata->>'segment_title') FROM rag_chunks "
                "WHERE collection = :col "
                "  AND chunk_type = 'begriff_list' "
                "  AND metadata->>'segment_title' IS NOT NULL "
                "  AND metadata->>'segment_title' != '' "
                "  AND (worldviews IS NULL OR array_length(worldviews, 1) IS NULL)"
            ),
            {"col": collection_name},
        )
        valid_lemmas = {r[0] for r in row.fetchall() if r[0]}

    if not valid_lemmas:
        return []

    cleaned = _ARTICLE_PREFIX.sub("", user_prompt.strip()).lower()
    words = re.findall(r"[a-zäöüß]+", cleaned)
    seen: set[str] = set()
    matched_lemmas: list[str] = []

    for w in words:
        if len(w) < MIN_WORD_LEN or w in seen:
            continue
        for variant in _lemma_lookup_variants(w):
            if variant in valid_lemmas:
                seen.add(variant)
                matched_lemmas.append(variant)
                break

    if not matched_lemmas:
        return []

    out: list[RetrievedSnippet] = []
    for lemma in matched_lemmas[:MAX_CHUNKS]:
        async with async_session() as session:
            row = await session.execute(
                text(
                    "SELECT chunk_id, text, metadata FROM rag_chunks "
                    "WHERE collection = :col "
                    "  AND chunk_type = 'begriff_list' "
                    "  AND LOWER(metadata->>'segment_title') = :lemma "
                    "  AND (worldviews IS NULL OR array_length(worldviews, 1) IS NULL) "
                    "LIMIT 1"
                ),
                {"col": collection_name, "lemma": lemma},
            )
            rows = row.fetchall()
        if rows:
            rec = rows[0]
            meta = dict(rec.metadata) if rec.metadata else {}
            meta["chunk_id"] = rec.chunk_id
            meta.setdefault("segment_title", lemma)
            out.append(
                RetrievedSnippet(
                    text=rec.text or "",
                    score=1.0,
                    payload={"payload": meta, "chunk_id": rec.chunk_id},
                )
            )
    return out


async def _run_query(
    *,
    query: str,
    query_spec: dict[str, Any],
    collection: str,
    embedding_client: EmbeddingClient,
    qdrant_client: QdrantClient,
) -> list[RetrievedSnippet]:
    """Execute a single query against Qdrant based on manifest spec."""
    chunk_types: list[str] = query_spec.get("chunk_types") or []
    k = int(query_spec.get("k") or 8)
    method = (query_spec.get("method") or "dense").lower()
    author = query_spec.get("author")

    worldview: str | None = query_spec.get("worldview")
    book_types = chunk_types

    if method == "hybrid":
        k_dense = k
        k_sparse = k if settings.use_hybrid_retrieval else 0
        k_fused = k
        hits = await hybrid_retrieve(
            query=query,
            k_dense=k_dense,
            k_sparse=k_sparse,
            k_fused=k_fused,
            worldview=worldview,
            book_types=book_types,
            collection=collection,
            embedding_client=embedding_client,
            qdrant_client=qdrant_client,
            author=author,
        )
    elif method == "sparse":
        hits = await sparse_retrieve(
            query=query,
            k=k,
            worldview=worldview,
            book_types=book_types,
            collection=collection,
            qdrant_client=qdrant_client,
            author=author,
        )
    else:
        hits = await dense_retrieve(
            query=query,
            k=k,
            worldview=worldview,
            book_types=book_types,
            collection=collection,
            embedding_client=embedding_client,
            qdrant_client=qdrant_client,
            author=author,
        )
    return hits


async def run_queries_and_fill_prompt(
    *,
    action_id: str,
    assistant_slug: str,
    user_prompt: str,
    collection_name: str,
    conversation_context: str,
    embedding_client: EmbeddingClient,
    qdrant_client: QdrantClient,
) -> tuple[str, str, dict[str, Any], list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    """
    Run queries per action-manifest, fill prompt template. Returns:
    (full_system_prompt, action_prompt_filled, query_results, citations_metadata,
     context_refs, retrieved_snippets_serialized).
    """
    manifest = load_action_manifest(action_id)
    if not manifest.get("requires_retrieval", True):
        # thanks-feedback: no retrieval
        prompt_template = load_action_prompt(action_id)
        all_placeholder_names = {
            "primary-books", "secondary-books", "primary", "secondary",
            "concepts", "lemma-lookup", "fallback-books", "quotes", "steiner-books", "works",
        }
        slot_values = {
            "conversation_context": conversation_context or "(keine)",
            "user_prompt": user_prompt,
        }
        for key in all_placeholder_names:
            slot_values[key] = "(kein Retrieval)"
        filled = prompt_template.format(**slot_values)
        instruction = load_assistant_instruction(assistant_slug)
        full_system = f"{instruction}\n\n{filled}"
        return full_system, filled, {}, [], [], [], None, []

    prompt_template = load_action_prompt(action_id)
    queries: list[dict[str, Any]] = manifest.get("queries") or []

    # Group parallel queries (parallel_with): run in parallel, fill each slot separately
    parallel_names: set[str] = set()
    for q in queries:
        if q.get("parallel_with"):
            parallel_names.add(q.get("name") or "")
            parallel_names.add(q.get("parallel_with"))
    standalone = [q for q in queries if (q.get("name") or "") not in parallel_names]

    query_results: dict[str, Any] = {}
    all_refs: list[str] = []
    all_snippets: list[RetrievedSnippet] = []

    chunk_index_map: list[dict[str, Any]] = []
    _next_index = [1]  # mutable to update in closure

    def _strip_quote_explanation(text: str) -> str:
        """Remove 'Erklärung:' block from quote chunks. LLM gets only the quote."""
        if not text or "\n\nErklärung:" not in text and "\n\nErklaerung:" not in text:
            return text
        for sep in ("\n\nErklärung:\n\n", "\n\nErklaerung:\n\n"):
            if sep in text:
                return text.split(sep)[0].strip()
        return text

    def _build_and_store(name: str, hits: list) -> None:
        # For quotes: strip explanation, pass only the quote text to the prompt
        if name == "quotes":
            hits = [
                RetrievedSnippet(
                    text=_strip_quote_explanation(s.text),
                    score=s.score,
                    payload=s.payload,
                )
                for s in hits
            ]
        ctx, refs, imap = build_context_numbered(
            hits, start_index=_next_index[0], include_score=True
        )
        _next_index[0] += len(imap)
        for num, cid, text, meta in imap:
            chunk_index_map.append({
                "index": num,
                "chunk_id": cid,
                "text": (text or "")[:CHUNK_TEXT_MAX_CHARS],
                "source_title": meta.get("source_title", ""),
                "segment_title": meta.get("segment_title", ""),
                "slot": name,
            })
        preview = ctx[:500] + "..." if len(ctx) > 500 else ctx
        query_results[name] = {
            "chunk_count": len(hits),
            "chunk_ids": refs,
            "preview": preview,
            "full_context": ctx,
        }

    for q in standalone:
        name = q.get("name") or "unnamed"
        chunk_types = q.get("chunk_types") or []
        method = (q.get("method") or "").lower()
        if (name == "lemma-lookup" or method == "lemma-lookup") and "begriff_list" in chunk_types:
            hits = await _lemma_lookup_multi_begriff_list(
                user_prompt=user_prompt,
                collection_name=collection_name,
            )
        else:
            hits = await _run_query(
                query=user_prompt,
                query_spec=q,
                collection=collection_name,
                embedding_client=embedding_client,
                qdrant_client=qdrant_client,
            )
        _build_and_store(name, hits)
        all_refs.extend(query_results[name].get("chunk_ids", []))
        all_snippets.extend(hits)

    # Parallel group: run each query, fill each slot (optionally fuse for citations)
    if parallel_names:
        parallel_specs = [(q.get("name"), q) for q in queries if (q.get("name") or "") in parallel_names]
        if len(parallel_specs) == 2 and any("quote" in (s[1].get("chunk_types") or []) for s in parallel_specs):
            # find-quote: use hybrid_retrieve_quote_parallel, then split or use fused for both
            fused = await hybrid_retrieve_quote_parallel(
                query=user_prompt,
                k_per_branch=5,
                k_fused=10,
                collection=collection_name,
                embedding_client=embedding_client,
                qdrant_client=qdrant_client,
            )
            ctx, refs, imap = build_context_numbered(
                fused, start_index=_next_index[0], include_score=True
            )
            for num, cid, text, meta in imap:
                ct = meta.get("chunk_type") if isinstance(meta, dict) else None
                slot = "quotes" if ct == "quote" else "primary-books"
                chunk_index_map.append({
                    "index": num,
                    "chunk_id": cid,
                    "text": (text or "")[:CHUNK_TEXT_MAX_CHARS],
                    "source_title": meta.get("source_title", ""),
                    "segment_title": meta.get("segment_title", ""),
                    "slot": slot,
                })
            _next_index[0] += len(imap)
            preview = ctx[:500] + "..." if len(ctx) > 500 else ctx
            for name, _ in parallel_specs:
                query_results[name] = {
                    "chunk_count": len(fused),
                    "chunk_ids": refs,
                    "preview": preview,
                    "full_context": ctx,
                }
            all_refs.extend(refs)
            all_snippets.extend(fused)
        else:
            tasks = [
                _run_query(
                    query=user_prompt,
                    query_spec=spec,
                    collection=collection_name,
                    embedding_client=embedding_client,
                    qdrant_client=qdrant_client,
                )
                for _, spec in parallel_specs
            ]
            results = await asyncio.gather(*tasks)
            for (name, _), hits in zip(parallel_specs, results):
                _build_and_store(name, hits)
                all_refs.extend(query_results[name].get("chunk_ids", []))
                all_snippets.extend(hits)

    # Build slot values for template (provide all known placeholders for safety)
    all_placeholder_names = {
        "primary-books", "secondary-books", "primary", "secondary",
        "concepts", "lemma-lookup", "fallback-books", "quotes", "steiner-books", "works",
    }
    slot_values: dict[str, str] = {
        "conversation_context": conversation_context or "(keine)",
        "user_prompt": user_prompt,
    }
    for name in all_placeholder_names:
        slot_values[name] = "(nicht abgerufen)"
    for name, res in query_results.items():
        slot_values[name] = res.get("full_context", res.get("preview", ""))

    filled = prompt_template.format(**slot_values)
    instruction = load_assistant_instruction(assistant_slug)
    full_system = f"{instruction}\n\n{filled}"

    # Build citations metadata (from chunk payloads)
    citations_metadata: list[dict[str, Any]] = []
    for s in all_snippets:
        payload = s.payload if isinstance(s.payload, dict) else {}
        meta = payload.get("payload", payload) if isinstance(payload.get("payload"), dict) else payload
        if not isinstance(meta, dict):
            meta = {}
        citations_metadata.append({
            "chunk_id": meta.get("chunk_id"),
            "source_title": meta.get("source_title", ""),
            "segment_title": meta.get("segment_title", ""),
            "segment_index": meta.get("segment_index"),
            "lecture_date": meta.get("lecture_date", ""),
        })

    # Serialize snippets for cache (execute-prompt needs them for citation evaluation)
    retrieved_serialized: list[dict[str, Any]] = [
        {"text": s.text, "score": s.score, "payload": dict(s.payload) if isinstance(s.payload, dict) else {}}
        for s in all_snippets
    ]

    # direct_response: when requires_prompt=False (e.g. clarify-concept), return retrieval result directly
    direct_response: str | None = None
    if not manifest.get("requires_prompt", True):
        lemma_res = query_results.get("lemma-lookup", {})
        if lemma_res.get("chunk_count", 0) > 0:
            direct_response = lemma_res.get("full_context", lemma_res.get("preview", ""))
        else:
            direct_response = "Den Begriff habe ich nicht gefunden."

    return (
        full_system,
        filled,
        query_results,
        citations_metadata,
        all_refs,
        retrieved_serialized,
        direct_response,
        chunk_index_map,
    )


def store_prompt_state(
    *,
    prompt_id: str,
    filled_prompt: str,
    instruction_prompt: str,
    action_prompt_filled: str,
    query_results: dict[str, Any],
    citations_metadata: list[dict[str, Any]],
    context_refs: list[str],
    assistant_slug: str,
    action_id: str,
    collection_name: str,
    user_prompt: str,
    retrieved_snippets: list[dict[str, Any]],
    chunk_index_map: list[dict[str, Any]] | None = None,
) -> None:
    """Store prompt state for execute-prompt (cached with TTL)."""
    expires = datetime.now(timezone.utc) + timedelta(seconds=_PROMPT_CACHE_TTL_SECONDS)
    _PROMPT_STATE_CACHE[prompt_id] = {
        "filled_prompt": filled_prompt,
        "instruction_prompt": instruction_prompt,
        "action_prompt_filled": action_prompt_filled,
        "query_results": query_results,
        "citations_metadata": citations_metadata,
        "context_refs": context_refs,
        "chunk_index_map": chunk_index_map or [],
        "assistant_slug": assistant_slug,
        "action_id": action_id,
        "collection_name": collection_name,
        "user_prompt": user_prompt,
        "retrieved_snippets": retrieved_snippets,
        "expires_at": expires.isoformat(),
    }


def get_prompt_state(prompt_id: str) -> dict[str, Any] | None:
    """Retrieve stored prompt state. Returns None if expired or not found."""
    entry = _PROMPT_STATE_CACHE.get(prompt_id)
    if not entry:
        return None
    expires_at = entry.get("expires_at")
    if expires_at:
        try:
            exp = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            if datetime.now(timezone.utc) >= exp:
                del _PROMPT_STATE_CACHE[prompt_id]
                return None
        except Exception:
            pass
    return entry


def generate_prompt_id() -> str:
    return str(uuid.uuid4())
