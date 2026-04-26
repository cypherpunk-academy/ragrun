"""Service for generate-prompt / execute-prompt (ASSISTANTS_CHAT_PLAN_V2)."""
from __future__ import annotations

import asyncio
import logging
import re
import unicodedata
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
    _extract_chunk_id,
    build_context,
    build_context_numbered,
    dense_retrieve,
    hybrid_retrieve,
    hybrid_retrieve_quote_parallel,
    redact_lexical_phrases,
    snippet_author,
    sparse_retrieve,
)

logger = logging.getLogger(__name__)

# Max chars per chunk text in chunk_index_map; keep in sync with action_prompt.REFERENCE_TEXT_MAX_CHARS
CHUNK_TEXT_MAX_CHARS = 2000

def _nfc(s: str) -> str:
    """NFC-normalize a string so umlaut comparisons work across macOS/Linux/Postgres."""
    return unicodedata.normalize("NFC", s)


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


def _resolve_personalities_root() -> Path | None:
    """Resolve external chat-personalities directory from config. Returns None if not configured."""
    if settings.personalities_root:
        return Path(settings.personalities_root)
    return None


def _resolve_default_personality_root() -> Path:
    """Resolve built-in default-personality directory (always available as fallback)."""
    return Path(__file__).resolve().parents[1] / "default-personality"


def _resolve_helpers_root() -> Path:
    """Resolve helper-actions directory (app/retrieval/helper-actions)."""
    return Path(__file__).resolve().parents[1] / "helper-actions"


def _resolve_assistants_root() -> Path:
    """Resolve assistants root (ragkeep/assistants)."""
    repo_root = Path(__file__).resolve().parents[3]
    configured = Path(settings.assistants_root)
    return configured if configured.is_absolute() else (repo_root / configured)


def _action_root(action_id: str) -> Path:
    """Return the directory for the given action_id.

    Search order: external personalities → built-in default-personality → helpers.
    """
    roots: list[Path] = []
    personalities_root = _resolve_personalities_root()
    if personalities_root is not None:
        roots.append(personalities_root)
    roots.append(_resolve_default_personality_root())
    roots.append(_resolve_helpers_root())
    for root in roots:
        # Fast path: folder name matches action_id directly.
        candidate = root / action_id
        if candidate.is_dir():
            return candidate
        # Compatibility path: folder slug differs, but action-manifest.yaml contains the id.
        for manifest_path in root.glob("*/action-manifest.yaml"):
            try:
                data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and data.get("id") == action_id:
                    return manifest_path.parent
            except Exception as exc:
                logger.warning("Failed to load manifest %s: %s", manifest_path, exc)
    raise ValueError(f"Action not found in any root: {action_id!r}")


def load_action_manifest(action_id: str) -> dict[str, Any]:
    """Load action-manifest.yaml for the given action_id."""
    manifest_path = _action_root(action_id) / "action-manifest.yaml"
    if not manifest_path.is_file():
        raise ValueError(f"Action manifest not found: {manifest_path}")
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def load_action_prompt(action_id: str) -> str:
    """Load user.prompt template for the given action_id."""
    prompt_path = _action_root(action_id) / "user.prompt"
    if not prompt_path.is_file():
        raise ValueError(f"Action prompt not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8").strip()


def _load_personality_system(action_id: str) -> str | None:
    """Load system.prompt for a personality action, if present.

    Searches external personalities root first, then built-in default-personality.
    """
    try:
        path = _action_root(action_id) / "system.prompt"
    except ValueError:
        return None
    if path.is_file():
        text = path.read_text(encoding="utf-8").strip()
        return text or None
    return None


def load_assistant_name(assistant_slug: str) -> str:
    """Load display name from assistant-manifest.yaml (falls back to slug)."""
    manifest_path = _resolve_assistants_root() / assistant_slug / "assistant-manifest.yaml"
    if not manifest_path.is_file():
        return assistant_slug
    try:
        data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("name"), str):
            return data["name"]
    except Exception:
        pass
    return assistant_slug


def load_assistant_rag_collection(assistant_slug: str) -> str:
    """Qdrant collection name from assistant-manifest ``rag-collection``, else ``assistant_slug``."""
    manifest_path = _resolve_assistants_root() / assistant_slug / "assistant-manifest.yaml"
    if not manifest_path.is_file():
        return assistant_slug
    try:
        data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            raw = data.get("rag-collection")
            if isinstance(raw, str):
                c = raw.strip()
                if c:
                    return c
    except Exception:
        logger.warning("Could not read rag-collection from %s", manifest_path, exc_info=True)
    return assistant_slug


def load_assistant_lexical_strip_hybrid(assistant_slug: str) -> list[str]:
    """Phrases to remove from the user prompt for the sparse branch of hybrid retrieval only."""
    manifest_path = _resolve_assistants_root() / assistant_slug / "assistant-manifest.yaml"
    if not manifest_path.is_file():
        return []
    try:
        data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return []
        raw = data.get("lexical-strip-hybrid")
        if raw is None:
            return []
        if not isinstance(raw, list):
            return []
        out: list[str] = []
        for x in raw:
            if isinstance(x, str) and x.strip():
                out.append(x.strip())
        return out
    except Exception:
        logger.warning(
            "Could not read lexical-strip-hybrid from %s", manifest_path, exc_info=True
        )
        return []


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


# Primary chat personas shown in the menu (order = display order). Not listed = hidden.
_PRIMARY_CHAT_MENU_ORDER: tuple[str, ...] = (
    "mit-kontext",
    "assistant-host",
    "socrates",
    "der-cypherpunk",
    "der-open-source-handwerker",
    "der-free-software-aktivist",
    "die-menschheitsaktivistin",
)

# Mark these primary labels with a (beta) suffix in the API list (menu).
_PRIMARY_CHAT_BETA_IDS: frozenset[str] = frozenset(
    {
        "der-cypherpunk",
        "der-open-source-handwerker",
        "der-free-software-aktivist",
        "die-menschheitsaktivistin",
    },
)


def list_actions(assistant_name: str = "") -> list[dict[str, Any]]:
    """List all available actions (from action-manifest files in personalities + helpers)."""
    out: list[dict[str, Any]] = []
    roots: list[Path] = []
    personalities_root = _resolve_personalities_root()
    if personalities_root is not None:
        roots.append(personalities_root)
    roots.append(_resolve_default_personality_root())
    roots.append(_resolve_helpers_root())
    for root in roots:
        for manifest_path in root.glob("*/action-manifest.yaml"):
            try:
                data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and data.get("id"):
                    label = data.get("label", data["id"])
                    if assistant_name:
                        label = label.replace("{assistant_name}", assistant_name)
                    out.append({
                        "id": data["id"],
                        "label": label,
                        "description": data.get("description", ""),
                        "category": data.get("category", "primary"),
                        "requires_prompt": data.get("requires_prompt", True),
                        "allows_empty_prompt": data.get("allows-empty-prompt", False),
                        "position_in_chat": data.get("position-in-chat", ["start", "continue"]),
                        "follow_ups": data.get("follow_ups", []),
                    })
            except Exception as exc:
                logger.warning("Failed to load manifest %s: %s", manifest_path, exc)

    order_index = {aid: i for i, aid in enumerate(_PRIMARY_CHAT_MENU_ORDER)}
    primaries = [a for a in out if a.get("category") == "primary"]
    others = [a for a in out if a.get("category") != "primary"]
    filtered_primaries = [a for a in primaries if a["id"] in order_index]
    for a in filtered_primaries:
        if a["id"] in _PRIMARY_CHAT_BETA_IDS:
            lab = (a.get("label") or "").strip()
            if lab and "(beta)" not in lab.lower():
                a["label"] = f"{lab} (beta)"
    filtered_primaries.sort(key=lambda a: order_index[a["id"]])
    return filtered_primaries + sorted(others, key=lambda x: x["id"])


async def _lemma_lookup_begrif(
    *,
    user_prompt: str,
    collection_name: str,
) -> list[RetrievedSnippet]:
    """Exact lemma lookup in Postgres rag_chunks for begriff. Returns matching chunks."""
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
                    "  AND chunk_type = 'begriff' "
                    "  AND LOWER(metadata->>'segment_title') = :lemma "
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


async def _lemma_lookup_multi_begrif(
    *,
    user_prompt: str,
    collection_name: str,
) -> list[RetrievedSnippet]:
    """Multi-lemma lookup: find all terms from user_prompt that exist in begriff, return their chunks."""
    MIN_WORD_LEN = 3
    MAX_CHUNKS = 10

    async_session = get_async_sessionmaker()

    async with async_session() as session:
        row = await session.execute(
            text(
                "SELECT DISTINCT LOWER(metadata->>'segment_title') FROM rag_chunks "
                "WHERE collection = :col "
                "  AND chunk_type = 'begriff' "
                "  AND metadata->>'segment_title' IS NOT NULL "
                "  AND metadata->>'segment_title' != '' "
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
                    "  AND chunk_type = 'begriff' "
                    "  AND LOWER(metadata->>'segment_title') = :lemma "
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


async def fetch_snippets_for_chunk_ids(
    chunk_ids: list[str],
    collection_name: str,
) -> list[RetrievedSnippet]:
    """Fetch primary-source snippets from rag_chunks for a list of chunk_ids."""
    if not chunk_ids:
        return []
    async_session = get_async_sessionmaker()
    async with async_session() as session:
        row = await session.execute(
            text(
                "SELECT chunk_id, text, metadata FROM rag_chunks "
                "WHERE collection = :col AND chunk_id = ANY(:ids)"
            ),
            {"col": collection_name, "ids": chunk_ids},
        )
        rows = row.fetchall()
    out: list[RetrievedSnippet] = []
    for rec in rows:
        meta = dict(rec.metadata) if rec.metadata else {}
        meta["chunk_id"] = rec.chunk_id
        out.append(
            RetrievedSnippet(
                text=rec.text or "",
                score=1.0,
                payload={"payload": meta, "chunk_id": rec.chunk_id},
            )
        )
    return out


async def _lemma_lookup_multi_typology(
    *,
    user_prompt: str,
    collection_name: str,
) -> list[RetrievedSnippet]:
    """Multi-lemma lookup for typology chunks: matches segment_title and aliases against user prompt."""
    MAX_CHUNKS = 5

    async_session = get_async_sessionmaker()

    # Build term -> segment_title map: include segment_titles and all aliases
    async with async_session() as session:
        seg_row = await session.execute(
            text(
                "SELECT DISTINCT LOWER(metadata->>'segment_title') FROM rag_chunks "
                "WHERE collection = :col "
                "  AND chunk_type = 'typology' "
                "  AND metadata->>'segment_title' IS NOT NULL "
                "  AND metadata->>'segment_title' != '' "
            ),
            {"col": collection_name},
        )
        segment_titles = {_nfc(r[0]) for r in seg_row.fetchall() if r[0]}

        alias_row = await session.execute(
            text(
                "SELECT LOWER(metadata->>'segment_title'), LOWER(alias_val) "
                "FROM rag_chunks, "
                "     jsonb_array_elements_text(metadata->'aliases') AS alias_val "
                "WHERE collection = :col "
                "  AND chunk_type = 'typology' "
                "  AND jsonb_typeof(metadata->'aliases') = 'array' "
            ),
            {"col": collection_name},
        )
        alias_to_segment: dict[str, str] = {}
        for r in alias_row.fetchall():
            seg, alias = r[0], r[1]
            if alias and seg:
                alias_to_segment[_nfc(alias)] = _nfc(seg)

    if not segment_titles and not alias_to_segment:
        return []

    # term -> segment_title (segment_title maps to itself), all NFC-normalized
    term_to_segment: dict[str, str] = {_nfc(s): _nfc(s) for s in segment_titles}
    term_to_segment.update(alias_to_segment)

    cleaned = _nfc(_ARTICLE_PREFIX.sub("", user_prompt.strip()).lower())

    # Match longest terms first (to prefer "12 weltanschauungen" over "weltanschauungen")
    matched_segments: list[str] = []
    seen_segments: set[str] = set()
    for term in sorted(term_to_segment, key=lambda t: -len(t)):
        seg = term_to_segment[term]
        if seg not in seen_segments and term in cleaned:
            seen_segments.add(seg)
            matched_segments.append(seg)

    if not matched_segments:
        return []

    out: list[RetrievedSnippet] = []
    for segment in matched_segments[:MAX_CHUNKS]:
        async with async_session() as session:
            row = await session.execute(
                text(
                    "SELECT chunk_id, text, metadata FROM rag_chunks "
                    "WHERE collection = :col "
                    "  AND chunk_type = 'typology' "
                    "  AND LOWER(metadata->>'segment_title') = :seg "
                    "LIMIT 1"
                ),
                {"col": collection_name, "seg": segment},
            )
            rows = row.fetchall()
        if rows:
            rec = rows[0]
            meta = dict(rec.metadata) if rec.metadata else {}
            meta["chunk_id"] = rec.chunk_id
            meta.setdefault("segment_title", segment)
            out.append(
                RetrievedSnippet(
                    text=rec.text or "",
                    score=1.0,
                    payload={"payload": meta, "chunk_id": rec.chunk_id},
                )
            )
    return out


def _resolve_refs_for_hits(hits: list[RetrievedSnippet]) -> list[str]:
    """Return chunk_ids to use as potential references for the given hits.

    For talk chunks: use the chunk_ids listed in their `references` metadata field
    (primary sources) rather than the talk chunk itself. For all other chunk types: use
    the direct chunk_id as usual.
    """
    ref_ids: list[str] = []
    seen: set[str] = set()

    for snip in hits:
        payload = snip.payload if isinstance(snip.payload, dict) else {}
        inner = payload.get("payload", payload) if isinstance(payload.get("payload"), dict) else payload
        if not isinstance(inner, dict):
            continue

        chunk_type = inner.get("chunk_type", "")
        if chunk_type == "talk":
            refs = inner.get("references") or []
            # Fallback: references may be nested under a "metadata" key
            if not refs and isinstance(inner.get("metadata"), dict):
                refs = inner["metadata"].get("references") or []
            for r in refs:
                if isinstance(r, dict):
                    cid = r.get("chunk_id")
                    if isinstance(cid, str) and cid and cid not in seen:
                        seen.add(cid)
                        ref_ids.append(cid)
        else:
            cid = _extract_chunk_id(payload) or ""
            if cid and cid not in seen:
                seen.add(cid)
                ref_ids.append(cid)

    return ref_ids


async def _run_query(
    *,
    query: str,
    query_spec: dict[str, Any],
    collection: str,
    embedding_client: EmbeddingClient,
    qdrant_client: QdrantClient,
    sparse_query: str | None = None,
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
            sparse_query=sparse_query,
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
            sparse_query=sparse_query,
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
    context_chunk_ids: list[str] | None = None,
) -> tuple[str, str, dict[str, Any], list[dict[str, Any]], list[str], list[dict[str, Any]], str | None, list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Run queries per action-manifest, fill prompt template. Returns:
    (system_prompt, action_prompt_filled, query_results, citations_metadata,
     context_refs, retrieved_snippets_serialized, direct_response, chunk_index_map,
     context_ref_snippets_serialized).
    system_prompt = identity (instruction.prompt) + personality mode (system.prompt).
    action_prompt_filled = filled user.prompt with sources — becomes the HumanMessage.
    """
    manifest = load_action_manifest(action_id)

    # allows-empty-prompt: fall back to last user turn from conversation_context
    if not user_prompt.strip() and manifest.get("allows-empty-prompt"):
        lines = [l.strip() for l in (conversation_context or "").splitlines() if l.strip()]
        user_prompt = lines[-1] if lines else user_prompt

    if not manifest.get("requires_retrieval", True):
        # give-feedback / summarize: no retrieval
        prompt_template = load_action_prompt(action_id)
        all_placeholder_names = {
            "primary-books", "secondary-books", "primary", "secondary",
            "concepts", "lemma-lookup", "fallback-books", "quotes", "steiner-books", "works",
        }
        slot_values = {
            "conversation_context": conversation_context or "(keine)",
            "user_prompt": user_prompt,
            "assistant_name": load_assistant_name(assistant_slug),
        }
        for key in all_placeholder_names:
            slot_values[key] = "(kein Retrieval)"
        filled = prompt_template.format(**slot_values)
        instruction = load_assistant_instruction(assistant_slug)
        personality_system = _load_personality_system(action_id)
        if personality_system:
            system_prompt = f"{instruction}\n\n---\n{personality_system}\n---"
        else:
            system_prompt = instruction
        # save-quote: return text directly — no execute-prompt / LLM (CLI handles locally too)
        direct_no_retrieval: str | None = None
        if action_id == "save-quote" and user_prompt.strip():
            direct_no_retrieval = user_prompt.strip()
        return system_prompt, filled, {}, [], [], [], direct_no_retrieval, [], []

    prompt_template = load_action_prompt(action_id)
    queries: list[dict[str, Any]] = manifest.get("queries") or []

    lexical_strip = load_assistant_lexical_strip_hybrid(assistant_slug)
    hybrid_sparse_query: str | None = None
    if lexical_strip:
        redacted = redact_lexical_phrases(user_prompt, lexical_strip)
        if redacted != user_prompt:
            hybrid_sparse_query = redacted

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
        for num, cid, text, meta, score in imap:
            chunk_index_map.append({
                "index": num,
                "chunk_id": cid,
                "text": (text or "")[:CHUNK_TEXT_MAX_CHARS],
                "source_title": meta.get("source_title", ""),
                "segment_title": meta.get("segment_title", ""),
                "slot": name,
                "score": score,
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
        if method == "lemma-lookup" or name == "lemma-lookup":
            hits = []
            if "begriff" in chunk_types:
                hits += await _lemma_lookup_multi_begrif(
                    user_prompt=user_prompt,
                    collection_name=collection_name,
                )
            if "typology" in chunk_types:
                hits += await _lemma_lookup_multi_typology(
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
                sparse_query=hybrid_sparse_query,
            )
        _build_and_store(name, hits)
        all_refs.extend(_resolve_refs_for_hits(hits))
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
                sparse_query=hybrid_sparse_query,
            )
            ctx, refs, imap = build_context_numbered(
                fused, start_index=_next_index[0], include_score=True
            )
            for num, cid, text, meta, score in imap:
                ct = meta.get("chunk_type") if isinstance(meta, dict) else None
                slot = "quotes" if ct == "quote" else "primary-books"
                chunk_index_map.append({
                    "index": num,
                    "chunk_id": cid,
                    "text": (text or "")[:CHUNK_TEXT_MAX_CHARS],
                    "source_title": meta.get("source_title", ""),
                    "segment_title": meta.get("segment_title", ""),
                    "slot": slot,
                    "score": score,
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
                    sparse_query=hybrid_sparse_query,
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
        "kontext",
    }
    slot_values: dict[str, str] = {
        "conversation_context": conversation_context or "(keine)",
        "user_prompt": user_prompt,
        "assistant_name": load_assistant_name(assistant_slug),
    }
    for name in all_placeholder_names:
        slot_values[name] = "(nicht abgerufen)"
    for name, res in query_results.items():
        slot_values[name] = res.get("full_context", res.get("preview", ""))

    # Inject pre-selected context chunks ({kontext} slot) when provided
    if context_chunk_ids:
        context_snippets = await fetch_snippets_for_chunk_ids(context_chunk_ids, collection_name)
        if context_snippets:
            ctx, _, _ = build_context_numbered(context_snippets, start_index=1, include_score=False)
            slot_values["kontext"] = ctx
        else:
            slot_values["kontext"] = "(Abschnitt nicht gefunden)"
    else:
        slot_values["kontext"] = "(kein Abschnitt gewählt)"

    filled = prompt_template.format(**slot_values)
    instruction = load_assistant_instruction(assistant_slug)
    personality_system = _load_personality_system(action_id)
    if personality_system:
        system_prompt = f"{instruction}\n\n---\n{personality_system}\n---"
    else:
        system_prompt = instruction

    # Build citations metadata (from chunk payloads)
    citations_metadata: list[dict[str, Any]] = []
    for s in all_snippets:
        payload = s.payload if isinstance(s.payload, dict) else {}
        meta = payload.get("payload", payload) if isinstance(payload.get("payload"), dict) else payload
        if not isinstance(meta, dict):
            meta = {}
        auth = snippet_author(payload) or ""
        citations_metadata.append({
            "chunk_id": meta.get("chunk_id"),
            "chunk_type": meta.get("chunk_type", ""),
            "source_title": meta.get("source_title", ""),
            "segment_title": meta.get("segment_title", ""),
            "segment_index": meta.get("segment_index"),
            "lecture_date": meta.get("lecture_date", ""),
            "author": auth,
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

    # Deduplicate all_refs while preserving order (talk reference expansion can produce
    # the same primary-source chunk_id from multiple talk hits)
    seen_refs: set[str] = set()
    deduped_refs: list[str] = []
    for cid in all_refs:
        if cid not in seen_refs:
            seen_refs.add(cid)
            deduped_refs.append(cid)

    # Fetch primary-source snippets for relevance evaluation in execute-prompt
    context_ref_snippets = await fetch_snippets_for_chunk_ids(deduped_refs, collection_name)
    context_ref_snippets_serialized: list[dict[str, Any]] = [
        {"text": s.text, "score": s.score, "payload": dict(s.payload) if isinstance(s.payload, dict) else {}}
        for s in context_ref_snippets
    ]

    return (
        system_prompt,
        filled,
        query_results,
        citations_metadata,
        deduped_refs,
        retrieved_serialized,
        direct_response,
        chunk_index_map,
        context_ref_snippets_serialized,
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
    context_ref_snippets: list[dict[str, Any]] | None = None,
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
        "context_ref_snippets": context_ref_snippets or [],
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
