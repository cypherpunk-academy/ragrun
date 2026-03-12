"""SSE-Streaming Chat-Endpunkt fuer Assistenten.

Endpunkte:
    POST /api/v1/agent/{assistant_slug}/chat/stream
        Startet einen neuen oder bestehenden Chat-Thread. Streamt Token-Events
        via Server-Sent Events (SSE), gefolgt von einem done-Event mit Citations.

    GET  /api/v1/agent/{assistant_slug}/chat/thread/{thread_id}
        Liest die gespeicherte Konversationshistorie eines Threads.
"""
from __future__ import annotations

import json
import logging
import uuid

from fastapi import APIRouter, Request
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    thread_id: str | None = None
    user_message: str


@router.post("/agent/{assistant_slug}/chat/stream")
async def chat_stream(
    assistant_slug: str,
    body: ChatRequest,
    request: Request,
) -> EventSourceResponse:
    """Startet oder setzt einen Chat-Thread fort. Antwortet als SSE-Stream.

    SSE-Event-Typen:
        {"type": "thread_id", "thread_id": "<uuid>"}
            Erstes Event; liefert die thread_id fuer Folgeanfragen.
        {"type": "status", "step": "<step_id>", "label": "<text>"}
            Zwischenstatus waehrend laengerer Verarbeitungsschritte
            (z.B. authentic_concept_explain). Kein Token, nur Fortschritt.
        {"type": "token", "content": "<text>"}
            Ein Token der LLM-Antwort. Wird Token-fuer-Token geliefert.
        {"type": "done", "citations": [...], "confidence_score": 0.0-1.0,
                         "intent": "<label>", "sufficiency": "<level>"}
            Letztes Event; enthaelt Metadaten der Antwort.
    """
    graph = request.app.state.chat_graph
    thread_id = body.thread_id or str(uuid.uuid4())
    # Collection name = assistant_slug (wie concept_explain_worldviews)
    # Quelle: ragkeep/assistants/{slug}/assistant-manifest.yaml → rag-collection
    collection_name = assistant_slug

    initial_state = {
        "assistant_slug":   assistant_slug,
        "collection_name":  collection_name,
        "user_message":     body.user_message,
        "messages":         [],
        "retry_count":      0,
        "lemma_found":      False,
        "extracted_lemma":  "",
        "retrieval_plan":   [],
        "context_text":     "",
        "context_refs":     [],
        "sufficiency":      "",
        "citations":        [],
        "final_response":   "",
        "confidence_score": 0.0,
    }
    config = {"configurable": {"thread_id": thread_id}}

    async def event_generator():
        yield {"data": json.dumps({"type": "thread_id", "thread_id": thread_id})}
        tokens_sent = False
        try:
            async for event in graph.astream_events(initial_state, config, version="v2"):
                kind = event.get("event")

                if kind == "on_custom_event" and event.get("name") == "ace_progress":
                    payload = event.get("data") or {}
                    yield {
                        "data": json.dumps({
                            "type":  "status",
                            "step":  payload.get("step", ""),
                            "label": payload.get("label", ""),
                        })
                    }

                elif kind == "on_chat_model_stream":
                    token = event["data"]["chunk"].content
                    if token:
                        tokens_sent = True
                        yield {"data": json.dumps({"type": "token", "content": token})}

                elif kind == "on_chain_end" and event.get("name") == "finalize":
                    output = event["data"].get("output") or {}
                    response_text = output.get("final_response", "")
                    # Bei run_authentic_concept_explain/return_begriff_chunk: kein Streaming,
                    # Antwort kommt erst in finalize. Als Token senden, damit die UI den Text anzeigt.
                    if response_text and not tokens_sent:
                        yield {"data": json.dumps({"type": "token", "content": response_text})}
                    yield {
                        "data": json.dumps({
                            "type":             "done",
                            "citations":        output.get("citations", []),
                            "confidence_score": output.get("confidence_score", 0.0),
                            "intent":           output.get("intent", ""),
                            "sufficiency":      output.get("sufficiency", ""),
                            "response":         response_text,
                        })
                    }
        except Exception:
            logger.exception("Error during chat stream (thread_id=%s)", thread_id)
            yield {"data": json.dumps({"type": "error", "message": "Interner Fehler"})}

    return EventSourceResponse(event_generator())


@router.get("/agent/{assistant_slug}/chat/thread/{thread_id}")
async def get_thread(
    assistant_slug: str,
    thread_id: str,
    request: Request,
) -> dict:
    """Liest die Konversationshistorie eines gespeicherten Threads."""
    graph = request.app.state.chat_graph
    snapshot = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    messages = [
        {
            "role":    "human" if isinstance(m, HumanMessage) else "assistant",
            "content": m.content,
        }
        for m in (snapshot.values.get("messages") or [])
    ]
    return {"thread_id": thread_id, "messages": messages}
