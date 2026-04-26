"""LangGraph Chat-Graph fuer assistenten-uebergreifenden Chat.

Iteration 1: Intent-Klassifikation, Lemma-Lookup, Qdrant-Retrieval (hybrid/dense),
             Antwort-Generierung mit Philo-Persona, Citation-Aufloesung, SSE-Streaming.

Graph-Fluss (via graph.get_graph().draw_mermaid()):

    __start__ --> classify_intent
    classify_intent -->|skip| finalize
    classify_intent -->|begriffe| lemma_lookup
    classify_intent -->|retrieval| route_retrieval_plan
    lemma_lookup -->|found| return_begriff_chunk --> finalize
    lemma_lookup -->|not found| run_authentic_concept_explain --> finalize | retrieve_chunks
    lemma_lookup -->|empty lemma| retrieve_chunks
    route_retrieval_plan --> retrieve_chunks
    retrieve_chunks -->|insufficient+retry| retrieve_chunks
    retrieve_chunks -->|ok| compose_answer
    compose_answer --> attach_citations
    attach_citations --> finalize
    finalize --> __end__
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from sqlalchemy import text
from typing_extensions import TypedDict

from app.config import settings
from app.core.providers import get_embedding_client, get_qdrant_client
from app.db.async_session import get_async_sessionmaker
from app.retrieval.services.providers import get_deepseek_chat
from app.retrieval.graphs.intents import INTENT_CHUNK_TYPE_MAP, SKIP_RETRIEVAL_INTENTS
from app.retrieval.prompts.philo_von_freisinn import load_system_prompt
from app.retrieval.models import RetrievedSnippet
from app.retrieval.chains.authentic_concept_explain import run_authentic_concept_explain_chain
from app.retrieval.services.graph_event_recorder import GraphEventRecorder
from app.retrieval.services.usage_recorder import UsageRecorder, enqueue_record_usage
from app.services.pricing_service import calculate_cost
from app.retrieval.utils.reference_evaluator import evaluate_chunk_relevance
from app.retrieval.utils.retrievers import build_context, hybrid_retrieve, hybrid_retrieve_quote_parallel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt loader
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_intent_classify_prompt() -> str:
    path = Path(__file__).parent.parent / "prompts" / "intent_classify.prompt"
    if not path.is_file():
        raise FileNotFoundError(f"Intent classify prompt not found: {path}")
    return path.read_text(encoding="utf-8").strip()


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def _make_llm(streaming: bool = False) -> ChatOpenAI:
    if not settings.deepseek_api_key:
        raise RuntimeError("RAGRUN_DEEPSEEK_API_KEY is required for chat")
    return ChatOpenAI(
        model=settings.deepseek_chat_model or "deepseek-chat",
        openai_api_key=settings.deepseek_api_key,
        openai_api_base=f"{str(settings.deepseek_base_url).rstrip('/')}/",
        temperature=0.3,
        max_tokens=1200,
        streaming=streaming,
    )


# ---------------------------------------------------------------------------
# Lemma normalisation
# ---------------------------------------------------------------------------

_ARTICLE_PREFIX = re.compile(
    r"^(der|die|das|ein|eine|des|dem|den|einem|einer)\s+",
    re.IGNORECASE,
)


def _normalize_lemma(raw: str) -> str:
    """Normalize a raw LLM-supplied lemma to the DB format (lowercase, no article).

    Examples:
        "das Ich"           -> "ich"
        "Ich-Begriff"       -> "ich-begriff"
        "menschliche Seele" -> "seele"   (last word = core noun)
        "Geist"             -> "geist"
    """
    cleaned = _ARTICLE_PREFIX.sub("", raw.strip()).lower()
    parts = cleaned.split()
    return parts[-1] if parts else cleaned


def _lemma_lookup_variants(lemma: str) -> list[str]:
    """Return lemma variants to try for DB lookup (exact first, then singular).

    German nouns in begriff are stored in singular. When the user asks
    "Was bedeutet Geister bei Steiner?", the LLM may extract "geister" (plural).
    We try exact match first, then plural -en → singular (remove -en).
    """
    variants = [lemma]
    if len(lemma) > 3 and lemma.endswith("en"):
        variants.append(lemma[:-2])
    return variants


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class _ChatStateRequired(TypedDict):
    assistant_slug: str
    collection_name: str
    user_message: str
    messages: Annotated[list[BaseMessage], add_messages]


class ChatState(_ChatStateRequired, total=False):
    # Intent
    intent: str
    intent_confidence: float
    extracted_lemma: str
    lemma_found: bool

    # Retrieval
    retrieval_plan: list[str]
    context_text: str
    context_refs: list[str]
    retrieved_snippets: list[dict]  # serializable dicts for evaluate_chunk_relevance
    retrieval_mode: str
    sufficiency: str
    retry_count: int

    # Begriff direct path
    begriff_chunk: dict | None

    # Output
    citations: list[dict]
    final_response: str
    confidence_score: float

    # Billing
    account_id: str


# ---------------------------------------------------------------------------
# Pydantic schema for structured LLM output
# ---------------------------------------------------------------------------

class IntentResult(BaseModel):
    intent: str
    confidence: float
    reasoning: str
    lemma: str = ""


# ---------------------------------------------------------------------------
# Sufficiency helper
# ---------------------------------------------------------------------------

def _compute_sufficiency(chunks: list) -> str:
    if not chunks:
        return "insufficient"
    scores = [getattr(c, "score", 0.0) for c in chunks]
    high = sum(1 for s in scores if s > 0.7)
    if high >= 3:
        return "high"
    if len(chunks) >= 2 or any(s > 0.5 for s in scores):
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Node 1: classify_intent
# ---------------------------------------------------------------------------

async def classify_intent(state: ChatState, config: RunnableConfig) -> dict:
    prompt_template = _load_intent_classify_prompt()

    await adispatch_custom_event(
        "ace_progress",
        {"step": "classify_intent", "label": "Ich klassifiziere den Prompt"},
        config=config,
    )

    recent_human = [
        m.content for m in state.get("messages", []) if isinstance(m, HumanMessage)
    ][-3:]
    conversation_context = (
        "\n".join(f"- {t}" for t in recent_human)
        if recent_human
        else "(kein vorheriger Kontext)"
    )

    prompt = prompt_template.format(conversation_context=conversation_context)

    llm = _make_llm(streaming=False).with_structured_output(
        IntentResult, method="json_mode", include_raw=True
    )
    raw_result = await llm.ainvoke(
        [
            SystemMessage(content=prompt),
            HumanMessage(content=state["user_message"]),
        ],
        config,
    )
    result: IntentResult = raw_result["parsed"]
    _usage_meta = (raw_result.get("raw") or object())
    _usage_meta = getattr(_usage_meta, "usage_metadata", None) or {}
    _ci_model = settings.deepseek_chat_model or "deepseek-chat"
    _ci_cost = calculate_cost(_ci_model, _usage_meta.get("input_tokens"), _usage_meta.get("output_tokens"))
    enqueue_record_usage(
        UsageRecorder(),
        account_id=state.get("account_id") or "anonymous",
        thread_id=(config.get("configurable") or {}).get("thread_id"),
        endpoint="graphs/assistant_chat/classify_intent",
        model=_ci_model,
        provider="deepseek",
        prompt_tokens=_usage_meta.get("input_tokens"),
        completion_tokens=_usage_meta.get("output_tokens"),
        total_tokens=_usage_meta.get("total_tokens"),
        extra=_ci_cost if (_ci_cost["cost_usd"] > 0) else None,
    )
    logger.info(
        "Intent: %s (confidence=%.2f, lemma=%r)",
        result.intent, result.confidence, result.lemma,
    )

    if result.intent == "begriff_definieren":
        normalized_lemma = _normalize_lemma(result.lemma)
        await adispatch_custom_event(
            "ace_progress",
            {
                "step": "classify_intent",
                "label": f'Erkannt als "begriff_definieren", Begriff ist: "{normalized_lemma}"',
            },
            config=config,
        )

    return {
        "intent": result.intent,
        "intent_confidence": result.confidence,
        "extracted_lemma": result.lemma,
        # Record the human turn in conversation history
        "messages": [HumanMessage(content=state["user_message"])],
    }


# ---------------------------------------------------------------------------
# Node 1b: lemma_lookup
# ---------------------------------------------------------------------------

async def _record_lemma_lookup_event(
    *,
    config: RunnableConfig,
    user_message: str,
    collection: str,
    intent: str,
    intent_confidence: float,
    extracted_lemma: str,
    lemma_found: bool,
    begriff_decision: str,
) -> None:
    """Best-effort: write classification + begriff decision to event_metadata/event_content."""
    try:
        thread_id = (config.get("configurable") or {}).get("thread_id", "") or str(uuid.uuid4())
        recorder = GraphEventRecorder()
        await recorder.record_event(
            graph_event_id=thread_id,
            graph_name="assistant_chat",
            step="lemma_lookup",
            concept=extracted_lemma or user_message[:200],
            collection=collection,
            query_text=user_message,
            prompt_messages={
                "classification": {
                    "intent": intent,
                    "confidence": intent_confidence,
                    "extracted_lemma": extracted_lemma,
                },
                "begriff_decision": begriff_decision,
            },
            metadata={
                "intent": intent,
                "intent_confidence": intent_confidence,
                "extracted_lemma": extracted_lemma,
                "lemma_found": lemma_found,
                "begriff_decision": begriff_decision,
            },
        )
    except Exception:
        logger.warning("lemma_lookup event recording failed (best-effort)", exc_info=True)


async def lemma_lookup(state: ChatState, config: RunnableConfig) -> dict:
    raw_lemma = state.get("extracted_lemma", "")
    lemma = _normalize_lemma(raw_lemma)
    intent = state.get("intent", "begriff_definieren")
    intent_confidence = state.get("intent_confidence") or 0.0

    if not lemma:
        logger.info("Lemma lookup: empty lemma, falling back to broad retrieval")
        await adispatch_custom_event(
            "ace_progress",
            {"step": "lemma_lookup", "label": "Begriff nicht gefunden, ich bestimme ihn neu (das kann eine Weile dauern)."},
            config=config,
        )
        await _record_lemma_lookup_event(
            config=config,
            user_message=state["user_message"],
            collection=state.get("collection_name", ""),
            intent=intent,
            intent_confidence=intent_confidence,
            extracted_lemma=raw_lemma,
            lemma_found=False,
            begriff_decision="retrieve_chunks",
        )
        return {
            "lemma_found": False,
            "retrieval_plan": ["book", "talk", "chapter_summary"],
        }

    # Try lemma variants (exact, then singular) since DB stores concepts in singular
    async_session = get_async_sessionmaker()
    rec = None
    matched_lemma = lemma
    for variant in _lemma_lookup_variants(lemma):
        async with async_session() as session:
            row = await session.execute(
                text(
                    "SELECT chunk_id, text, metadata FROM rag_chunks "
                    "WHERE collection = :col "
                    "  AND chunk_type = 'begriff' "
                    "  AND metadata->>'segment_title' = :lemma "
                    "LIMIT 1"
                ),
                {"col": state["collection_name"], "lemma": variant},
            )
            rec = row.fetchone()
        if rec and rec.text:
            matched_lemma = variant
            break

    if rec and rec.text:
        if matched_lemma != lemma:
            logger.info("Lemma lookup: '%s' found in begriff (via singular '%s')", lemma, matched_lemma)
        else:
            logger.info("Lemma lookup: '%s' found in begriff", lemma)
        meta = rec.metadata or {}
        await adispatch_custom_event(
            "ace_progress",
            {"step": "lemma_lookup", "label": "Begriff gefunden, gebe ihn zurück"},
            config=config,
        )
        await _record_lemma_lookup_event(
            config=config,
            user_message=state["user_message"],
            collection=state.get("collection_name", ""),
            intent=intent,
            intent_confidence=intent_confidence,
            extracted_lemma=lemma,
            lemma_found=True,
            begriff_decision="use_begriff_chunk",
        )
        return {
            "lemma_found": True,
            "begriff_chunk": {
                "chunk_id": rec.chunk_id,
                "text": rec.text,
                "metadata": dict(meta) if hasattr(meta, "items") else meta,
            },
            "retrieval_plan": ["begriff", "explanation"],
        }

    logger.info("Lemma lookup: '%s' not found, using authentic_concept_explain", lemma)
    await adispatch_custom_event(
        "ace_progress",
        {"step": "lemma_lookup", "label": "Begriff nicht gefunden, ich bestimme ihn neu (das kann eine Weile dauern)."},
        config=config,
    )
    await _record_lemma_lookup_event(
        config=config,
        user_message=state["user_message"],
        collection=state.get("collection_name", ""),
        intent=intent,
        intent_confidence=intent_confidence,
        extracted_lemma=lemma,
        lemma_found=False,
        begriff_decision="run_authentic_concept_explain",
    )
    return {
        "lemma_found": False,
        "begriff_chunk": None,
        "retrieval_plan": ["book", "talk", "chapter_summary"],
    }


# ---------------------------------------------------------------------------
# Node 1c: return_begriff_chunk
# ---------------------------------------------------------------------------

async def return_begriff_chunk(state: ChatState, config: RunnableConfig) -> dict:
    """Return begriff chunk text directly with citation (no evaluation)."""
    chunk = state.get("begriff_chunk")
    if not chunk or not chunk.get("text"):
        logger.warning("return_begriff_chunk: no begriff_chunk text, fallback to retrieval")
        return {"retrieval_plan": ["book", "talk", "chapter_summary"], "begriff_fallback": True}

    meta = chunk.get("metadata") or {}
    citation = {
        "chunk_id":      chunk.get("chunk_id", ""),
        "source_title":  meta.get("source_title", ""),
        "segment_title": meta.get("segment_title", ""),
        "segment_index": meta.get("segment_index"),
        "lecture_date":  meta.get("lecture_date", ""),
    }
    return {
        "messages": [AIMessage(content=chunk["text"])],
        "citations": [citation],
        "begriff_fallback": False,
    }


# ---------------------------------------------------------------------------
# Node 1d: run_authentic_concept_explain
# ---------------------------------------------------------------------------

async def run_authentic_concept_explain(state: ChatState, config: RunnableConfig) -> dict:
    """Run authentic_concept_explain chain when begriff has no chunk for lemma."""
    lemma = _normalize_lemma(state.get("extracted_lemma", ""))
    if not lemma:
        logger.warning("run_authentic_concept_explain: empty lemma, fallback to retrieval")
        return {"retrieval_plan": ["book", "talk", "chapter_summary"], "ace_failed": True}

    async def _emit_progress(step: str, label: str) -> None:
        await adispatch_custom_event(
            "ace_progress",
            {"step": step, "label": label},
            config=config,
        )

    try:
        embedding_client = get_embedding_client()
        qdrant_client = get_qdrant_client()
        chat_client = get_deepseek_chat()

        result = await run_authentic_concept_explain_chain(
            concept=lemma,
            collection=state["collection_name"],
            embedding_client=embedding_client,
            qdrant_client=qdrant_client,
            chat_client=chat_client,
            progress_callback=_emit_progress,
        )
    except Exception:
        logger.warning("authentic_concept_explain failed; fallback to retrieval", exc_info=True)
        return {"retrieval_plan": ["book", "talk", "chapter_summary"], "ace_failed": True}

    # Convert references (chunk_id, description, relevance) to citations (chunk_id, source_title, ...)
    refs = result.references or []
    chunk_ids = [r["chunk_id"] for r in refs]
    citations: list[dict] = []

    if chunk_ids:
        async_session = get_async_sessionmaker()
        async with async_session() as session:
            rows = await session.execute(
                text(
                    "SELECT chunk_id, metadata FROM rag_chunks "
                    "WHERE chunk_id = ANY(:ids)"
                ),
                {"ids": chunk_ids},
            )
            for row in rows:
                meta = row.metadata or {}
                citations.append({
                    "chunk_id":      row.chunk_id,
                    "source_title":  meta.get("source_title", ""),
                    "segment_title": meta.get("segment_title", ""),
                    "segment_index": meta.get("segment_index"),
                    "lecture_date":  meta.get("lecture_date", ""),
                })

    return {
        "messages": [AIMessage(content=result.lexicon_entry)],
        "citations": citations,
        "ace_failed": False,
    }


# ---------------------------------------------------------------------------
# Node 2: route_retrieval_plan
# ---------------------------------------------------------------------------

async def route_retrieval_plan(state: ChatState, config: RunnableConfig) -> dict:
    plan = INTENT_CHUNK_TYPE_MAP.get(state.get("intent", "sonstiges"), [])
    logger.info("Retrieval plan for intent '%s': %s", state.get("intent"), plan)
    return {"retrieval_plan": plan}


# ---------------------------------------------------------------------------
# Node 3: retrieve_chunks
# ---------------------------------------------------------------------------

async def retrieve_chunks(state: ChatState, config: RunnableConfig) -> dict:
    retry = state.get("retry_count", 0)
    embedding_client = get_embedding_client()
    qdrant_client = get_qdrant_client()
    collection = state["collection_name"]

    # On retry: widen to all chunk types (no filter)
    book_types: list[str] = [] if retry >= 1 else list(state.get("retrieval_plan") or [])

    if state.get("intent") == "quelle_suchen" and retry == 0:
        chunks = await hybrid_retrieve_quote_parallel(
            query=state["user_message"],
            k_per_branch=8,
            k_fused=12,
            collection=collection,
            embedding_client=embedding_client,
            qdrant_client=qdrant_client,
        )
    else:
        chunks = await hybrid_retrieve(
            query=state["user_message"],
            k_dense=15 + retry * 10,
            k_sparse=15 + retry * 10,
            k_fused=10 + retry * 5,
            worldview=None,
            book_types=book_types,
            collection=collection,
            embedding_client=embedding_client,
            qdrant_client=qdrant_client,
        )

    sufficiency = _compute_sufficiency(chunks)
    context_text, context_refs = build_context(chunks)
    retrieval_mode = "hybrid" if settings.use_hybrid_retrieval else "dense"

    # Serialize snippets for state (evaluate_chunk_relevance needs RetrievedSnippet later)
    retrieved_snippets = [
        {"text": s.text, "score": s.score, "payload": dict(s.payload) if isinstance(s.payload, dict) else s.payload}
        for s in chunks
    ]

    logger.info(
        "Retrieve: collection=%s retry=%d chunks=%d sufficiency=%s mode=%s",
        collection, retry, len(chunks), sufficiency, retrieval_mode,
    )
    return {
        "context_text": context_text,
        "context_refs": context_refs,
        "retrieved_snippets": retrieved_snippets,
        "retrieval_mode": retrieval_mode,
        "sufficiency": sufficiency,
        "retry_count": retry + 1,
    }


# ---------------------------------------------------------------------------
# Node 4: compose_answer
# ---------------------------------------------------------------------------

async def compose_answer(state: ChatState, config: RunnableConfig) -> dict:
    # TODO (Iter. 2+): resolve persona by assistant_slug
    persona = load_system_prompt()
    llm = _make_llm(streaming=True)

    context_text = state.get("context_text", "")
    recent_messages = list(state.get("messages") or [])[-6:]  # last 3 turns

    messages_in: list[BaseMessage] = [SystemMessage(content=persona)]
    if context_text:
        messages_in.append(SystemMessage(content=f"Quellen-Kontext:\n{context_text}"))
    messages_in.extend(recent_messages)
    messages_in.append(HumanMessage(content=state["user_message"]))

    response = ""
    usage_meta: dict = {}
    async for chunk in llm.astream(messages_in, config):
        response += chunk.content
        if chunk.usage_metadata:
            usage_meta = chunk.usage_metadata

    thread_id = (config.get("configurable") or {}).get("thread_id")
    usage_comment = ""
    if usage_meta:
        _model = settings.deepseek_chat_model or "deepseek-chat"
        _pt = usage_meta.get("input_tokens")
        _ct = usage_meta.get("output_tokens")
        _tt = usage_meta.get("total_tokens")
        _cost = calculate_cost(_model, _pt, _ct)
        usage_comment = (
            f'\n<!-- usage {json.dumps({"prompt_tokens": _pt, "completion_tokens": _ct, "total_tokens": _tt, "model": _model, "cost_usd": _cost["cost_usd"], "cost_eur": _cost["cost_eur"]})} -->'
        )
        enqueue_record_usage(
            UsageRecorder(),
            account_id=state.get("account_id") or "anonymous",
            thread_id=thread_id,
            endpoint="graphs/assistant_chat/compose_answer",
            model=_model,
            provider="deepseek",
            prompt_tokens=_pt,
            completion_tokens=_ct,
            total_tokens=_tt,
            extra=_cost if (_cost["cost_usd"] > 0) else None,
        )

    return {"messages": [AIMessage(content=response + usage_comment)]}


# ---------------------------------------------------------------------------
# Node 5: attach_citations
# ---------------------------------------------------------------------------

def _snippets_from_state(raw: list[dict]) -> list[RetrievedSnippet]:
    """Reconstruct RetrievedSnippet list from serialized state."""
    out: list[RetrievedSnippet] = []
    for d in raw or []:
        if not isinstance(d, dict):
            continue
        text = d.get("text") or ""
        score = float(d.get("score", 0.0))
        payload = d.get("payload")
        if payload is None:
            payload = {}
        if not isinstance(payload, dict):
            payload = dict(payload) if hasattr(payload, "items") else {}
        out.append(RetrievedSnippet(text=text, score=score, payload=payload))
    return out


async def attach_citations(state: ChatState, config: RunnableConfig) -> dict:
    context_refs = list(state.get("context_refs") or [])
    if not context_refs:
        return {"citations": []}

    # Get generated answer from last AI message
    last_ai = next(
        (m for m in reversed(list(state.get("messages") or [])) if isinstance(m, AIMessage)),
        None,
    )
    generated_text = (last_ai.content or "").strip() if last_ai else ""

    # Evaluate which chunks actually support the answer
    raw_snippets = state.get("retrieved_snippets") or []
    chunk_ids_to_cite = context_refs  # fallback: all

    if generated_text and raw_snippets:
        snippets = _snippets_from_state(raw_snippets)
        if snippets:
            try:
                llm = get_deepseek_chat()
                refs = await evaluate_chunk_relevance(
                    generated_text=generated_text,
                    retrieved_chunks=snippets,
                    llm=llm,
                    max_chunks=20,
                )
                threshold = settings.citation_relevance_threshold
                chunk_ids_to_cite = [
                    r["chunk_id"] for r in refs if float(r.get("relevance", 0.0)) >= threshold
                ]
                if not chunk_ids_to_cite:
                    # Keep top 3 by relevance if all below threshold
                    chunk_ids_to_cite = [r["chunk_id"] for r in refs[:3]]
            except Exception:
                logger.warning("Citation evaluation failed; using all context_refs", exc_info=True)

    async_session = get_async_sessionmaker()
    async with async_session() as session:
        rows = await session.execute(
            text(
                "SELECT chunk_id, metadata FROM rag_chunks "
                "WHERE chunk_id = ANY(:ids)"
            ),
            {"ids": chunk_ids_to_cite},
        )
        citations = []
        for row in rows:
            meta = row.metadata or {}
            citations.append({
                "chunk_id":      row.chunk_id,
                "source_title":  meta.get("source_title", ""),
                "segment_title": meta.get("segment_title", ""),
                "segment_index": meta.get("segment_index"),
                "lecture_date":  meta.get("lecture_date", ""),
            })
    return {"citations": citations}


# ---------------------------------------------------------------------------
# Node 6: finalize
# ---------------------------------------------------------------------------

async def finalize(state: ChatState, config: RunnableConfig) -> dict:
    last_ai = next(
        (m for m in reversed(list(state.get("messages") or [])) if isinstance(m, AIMessage)),
        None,
    )

    if last_ai is None:
        # Skip path: no compose_answer was called → generate direct response
        llm = _make_llm(streaming=False)
        persona = load_system_prompt()
        msg = await llm.ainvoke(
            [
                SystemMessage(content=persona),
                HumanMessage(content=state["user_message"]),
            ],
            config,
        )
        response = msg.content
        _skip_usage = msg.usage_metadata or {}
        if _skip_usage:
            _thread_id = (config.get("configurable") or {}).get("thread_id")
            _sk_model = settings.deepseek_chat_model or "deepseek-chat"
            _sk_cost = calculate_cost(_sk_model, _skip_usage.get("input_tokens"), _skip_usage.get("output_tokens"))
            response += (
                f'\n<!-- usage {json.dumps({"prompt_tokens": _skip_usage.get("input_tokens"), "completion_tokens": _skip_usage.get("output_tokens"), "total_tokens": _skip_usage.get("total_tokens"), "model": _sk_model, "cost_usd": _sk_cost["cost_usd"], "cost_eur": _sk_cost["cost_eur"]})} -->'
            )
            enqueue_record_usage(
                UsageRecorder(),
                account_id=state.get("account_id") or "anonymous",
                thread_id=_thread_id,
                endpoint="graphs/assistant_chat/finalize_skip",
                model=_sk_model,
                provider="deepseek",
                prompt_tokens=_skip_usage.get("input_tokens"),
                completion_tokens=_skip_usage.get("output_tokens"),
                total_tokens=_skip_usage.get("total_tokens"),
                extra=_sk_cost if (_sk_cost["cost_usd"] > 0) else None,
            )
    else:
        response = last_ai.content
        if state.get("sufficiency") == "insufficient":
            response = (
                "Zu diesem Thema finde ich in meinen Quellen keinen ausreichenden Beleg.\n\n"
                + response
            )

    # Event logging (best-effort, Schritt 7)
    try:
        from app.retrieval.services.graph_event_recorder import GraphEventRecorder
        recorder = GraphEventRecorder()
        thread_id = (config.get("configurable") or {}).get("thread_id", "")
        await recorder.record_event(
            graph_event_id=thread_id or str(uuid.uuid4()),
            graph_name="assistant_chat",
            step="finalize",
            concept=state["user_message"][:200],
            collection=state.get("collection_name"),
            query_text=state["user_message"],
            response_text=response,
            retrieval_mode=state.get("retrieval_mode"),
            sufficiency=state.get("sufficiency"),
            context_refs=list(state.get("context_refs") or []),
            metadata={
                "assistant_slug":    state.get("assistant_slug"),
                "intent":            state.get("intent"),
                "intent_confidence": state.get("intent_confidence"),
            },
        )
    except Exception:
        logger.warning("Chat event logging failed (best-effort)", exc_info=True)

    return {
        "final_response": response,
        "confidence_score": state.get("intent_confidence") or 0.0,
        # Pass through for SSE done-event
        "citations":  list(state.get("citations") or []),
        "intent":     state.get("intent", ""),
        "sufficiency": state.get("sufficiency", ""),
    }


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def route_after_intent(state: ChatState) -> str:
    intent = state.get("intent", "sonstiges")
    if intent in SKIP_RETRIEVAL_INTENTS:
        return "finalize"
    if intent == "begriff_definieren":
        return "lemma_lookup"
    return "route_retrieval_plan"


def route_after_lemma_lookup(state: ChatState) -> str:
    lemma = _normalize_lemma(state.get("extracted_lemma", ""))
    if not lemma:
        return "retrieve_chunks"
    if state.get("lemma_found"):
        return "return_begriff_chunk"
    return "run_authentic_concept_explain"


def route_after_ace(state: ChatState) -> str:
    if state.get("ace_failed"):
        return "retrieve_chunks"
    return "finalize"


def route_after_return_begriff(state: ChatState) -> str:
    if state.get("begriff_fallback"):
        return "retrieve_chunks"
    return "finalize"


def route_after_retrieval(state: ChatState) -> str:
    if state.get("sufficiency") == "insufficient" and (state.get("retry_count") or 0) < 2:
        return "retrieve_chunks"
    return "compose_answer"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_chat_graph(checkpointer=None):
    """Build and compile the assistant chat LangGraph.

    Args:
        checkpointer: LangGraph checkpointer. Defaults to MemorySaver (RAM only).
            For production (Iter. 2), pass AsyncPostgresSaver.

    Returns:
        Compiled StateGraph ready for astream_events().
    """
    builder = StateGraph(ChatState)

    builder.add_node("classify_intent",           classify_intent)
    builder.add_node("lemma_lookup",              lemma_lookup)
    builder.add_node("return_begriff_chunk",      return_begriff_chunk)
    builder.add_node("run_authentic_concept_explain", run_authentic_concept_explain)
    builder.add_node("route_retrieval_plan",      route_retrieval_plan)
    builder.add_node("retrieve_chunks",            retrieve_chunks)
    builder.add_node("compose_answer",             compose_answer)
    builder.add_node("attach_citations",          attach_citations)
    builder.add_node("finalize",                  finalize)

    builder.set_entry_point("classify_intent")

    builder.add_conditional_edges(
        "classify_intent",
        route_after_intent,
        {
            "finalize":             "finalize",
            "lemma_lookup":         "lemma_lookup",
            "route_retrieval_plan": "route_retrieval_plan",
        },
    )
    builder.add_conditional_edges(
        "lemma_lookup",
        route_after_lemma_lookup,
        {
            "return_begriff_chunk":      "return_begriff_chunk",
            "run_authentic_concept_explain": "run_authentic_concept_explain",
            "retrieve_chunks":           "retrieve_chunks",
        },
    )
    builder.add_conditional_edges(
        "return_begriff_chunk",
        route_after_return_begriff,
        {"retrieve_chunks": "retrieve_chunks", "finalize": "finalize"},
    )
    builder.add_conditional_edges(
        "run_authentic_concept_explain",
        route_after_ace,
        {
            "retrieve_chunks": "retrieve_chunks",
            "finalize":       "finalize",
        },
    )
    builder.add_edge("route_retrieval_plan", "retrieve_chunks")
    builder.add_conditional_edges(
        "retrieve_chunks",
        route_after_retrieval,
        {
            "retrieve_chunks": "retrieve_chunks",
            "compose_answer":  "compose_answer",
        },
    )
    builder.add_edge("compose_answer",   "attach_citations")
    builder.add_edge("attach_citations", "finalize")
    builder.add_edge("finalize",         END)

    return builder.compile(checkpointer=checkpointer or MemorySaver())
