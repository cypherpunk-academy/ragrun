"""LangGraph: retrieve context → explain a philosophical thought (250–280 words)."""
from __future__ import annotations

import logging
import uuid
from typing import Any, Mapping, Sequence, TypedDict

from langgraph.graph import END, StateGraph

from app.infra.deepseek_client import DeepSeekClient
from app.infra.embedding_client import EmbeddingClient
from app.infra.qdrant_client import QdrantClient
from app.retrieval.models import RetrievedSnippet, ThoughtExplainResult
from app.retrieval.prompts.thought_explain import build_thought_explain_messages, count_words
from app.retrieval.services.action_prompt_service import load_assistant_embedding_prefixes
from app.retrieval.utils.reference_evaluator import evaluate_chunk_relevance
from app.retrieval.utils.retrievers import build_context, dense_retrieve, rerank_by_embedding
from app.retrieval.utils.retry import retry_async

logger = logging.getLogger(__name__)

BOOK_TYPES = ("book", "secondary_book", "chapter_summary")


class ThoughtExplainState(TypedDict, total=False):
    thought_id: str
    text: str
    quote: str
    theme: str
    alignment: str
    writing_style: str
    collection: str
    assistant_slug: str
    incoming_references: list[dict[str, Any]]
    verbose: bool
    llm_retries: int
    snippets: list[RetrievedSnippet]
    explanation: str
    word_count: int
    references: list[dict[str, Any]]
    graph_event_id: str


def _is_retryable_llm_error(exc: Exception) -> bool:
    message = str(exc).lower()
    if "deepseek returned empty content" in message:
        return True
    if "deepseek returned no choices" in message:
        return True
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    return bool(status_code and status_code >= 500)


def build_thought_explain_graph(
    *,
    embedding_client: EmbeddingClient,
    qdrant_client: QdrantClient,
    chat_client: DeepSeekClient,
) -> Any:
    async def retrieve_node(state: ThoughtExplainState) -> dict[str, Any]:
        collection = (state.get("collection") or "philo-von-freisinn").strip()
        slug = (state.get("assistant_slug") or collection).strip()
        query = (state.get("text") or state.get("quote") or "").strip()
        _passage_prefix, query_prefix = load_assistant_embedding_prefixes(slug)
        raw = await dense_retrieve(
            query=query,
            k=12,
            worldview=None,
            book_types=BOOK_TYPES,
            collection=collection,
            embedding_client=embedding_client,
            qdrant_client=qdrant_client,
            query_prefix=query_prefix or None,
        )
        snippets = await rerank_by_embedding(
            query=query,
            snippets=raw,
            embedding_client=embedding_client,
            k_final=8,
            query_prefix=query_prefix or None,
            passage_prefix=_passage_prefix or None,
        )
        return {"snippets": list(snippets or [])}

    async def explain_node(state: ThoughtExplainState) -> dict[str, Any]:
        snippets = list(state.get("snippets") or [])
        context, _refs = build_context(snippets, max_chars=6000)
        messages = build_thought_explain_messages(
            text=state.get("text") or "",
            quote=state.get("quote") or "",
            theme=state.get("theme") or "",
            alignment=state.get("alignment") or "",
            writing_style=state.get("writing_style") or "",
            context=context,
        )
        retries = int(state.get("llm_retries") or 3)
        verbose = bool(state.get("verbose"))

        async def _once() -> str:
            if verbose:
                logger.warning(
                    "Prompt for thought_explain:\n%s",
                    "\n\n".join(f"[{m['role']}] {m['content']}" for m in messages),
                )
            result = await chat_client.chat(list(messages), temperature=0.3, max_tokens=900)
            return (result.content or "").strip()

        explanation = await retry_async(
            _once,
            retries=max(0, retries),
            base_delay=1.0,
            max_delay=8.0,
            jitter=0.2,
            retry_on=_is_retryable_llm_error,
            logger=logger,
            operation="thought_explain",
        )
        return {
            "explanation": explanation,
            "word_count": count_words(explanation),
        }

    async def references_node(state: ThoughtExplainState) -> dict[str, Any]:
        incoming = list(state.get("incoming_references") or [])
        if incoming:
            return {"references": incoming}
        snippets = list(state.get("snippets") or [])
        explanation = state.get("explanation") or ""
        if not snippets or not explanation:
            return {"references": []}
        refs = await evaluate_chunk_relevance(
            generated_text=explanation,
            retrieved_chunks=snippets,
            llm=chat_client,
        )
        normalized: list[dict[str, Any]] = []
        for r in refs or []:
            if not isinstance(r, Mapping):
                continue
            cid = r.get("chunk_id")
            if not isinstance(cid, str) or not cid.strip():
                continue
            normalized.append(
                {
                    "chunk_id": cid.strip(),
                    "description": str(r.get("description") or "").strip(),
                    "relevance": float(r.get("relevance") or 0.0),
                }
            )
        return {"references": normalized}

    builder = StateGraph(ThoughtExplainState)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("explain", explain_node)
    builder.add_node("references", references_node)
    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "explain")
    builder.add_edge("explain", "references")
    builder.add_edge("references", END)
    return builder.compile()


async def run_thought_explain_graph(
    *,
    thought_id: str,
    text: str,
    quote: str,
    theme: str,
    alignment: str,
    collection: str,
    embedding_client: EmbeddingClient,
    qdrant_client: QdrantClient,
    chat_client: DeepSeekClient,
    writing_style: str = "",
    assistant_slug: str | None = None,
    incoming_references: Sequence[Mapping[str, Any]] | None = None,
    verbose: bool = False,
    llm_retries: int = 3,
) -> ThoughtExplainResult:
    event_id = str(uuid.uuid4())
    graph = build_thought_explain_graph(
        embedding_client=embedding_client,
        qdrant_client=qdrant_client,
        chat_client=chat_client,
    )
    initial: ThoughtExplainState = {
        "thought_id": thought_id,
        "text": text,
        "quote": quote,
        "theme": theme,
        "alignment": alignment,
        "writing_style": writing_style,
        "collection": collection,
        "assistant_slug": (assistant_slug or collection).strip(),
        "incoming_references": [dict(r) for r in (incoming_references or [])],
        "verbose": verbose,
        "llm_retries": llm_retries,
        "graph_event_id": event_id,
    }
    final: ThoughtExplainState = await graph.ainvoke(initial)
    explanation = (final.get("explanation") or "").strip()
    if not explanation:
        raise ValueError("thought-explain produced empty explanation")
    return ThoughtExplainResult(
        thought_id=thought_id,
        explanation=explanation,
        word_count=int(final.get("word_count") or count_words(explanation)),
        references=list(final.get("references") or []) or None,
        graph_event_id=event_id,
    )
