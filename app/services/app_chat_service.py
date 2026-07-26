"""Non-streaming chat for /app/chat.

3a.5: nutzt denselben Graph-Kern wie /app/chat/stream (app_chat_stream_service),
nicht mehr den abgespeckten app_search-Pfad. `send_app_chat` bleibt als dünner
Wrapper erhalten, damit `app/api/app_api.py` unverändert bleibt.
"""
from __future__ import annotations

from typing import Any

from app.db.ports import TalksPort
from app.retrieval.services.providers import get_deepseek_chat
from app.services.app_chat_stream_service import run_app_chat_once


async def send_app_chat(
    graph: Any,
    talks: TalksPort,
    *,
    user_id: str,
    user_name: str,
    message: str,
    personality: str,
    talk_id: str | None = None,
    mode: str | None = None,
    model: str | None = None,
    context_mode: str | None = None,
    context_ids: dict[str, Any] | None = None,
    context_paragraph_text: str | None = None,
    linked_document_id: str | None = None,
    document_outline: dict[str, Any] | None = None,
    linked_document_content: str | None = None,
    app_tool_registry: Any = None,
) -> dict[str, str]:
    return await run_app_chat_once(
        graph,
        talks,
        user_id=user_id,
        user_name=user_name,
        message=message,
        personality=personality,
        talk_id=talk_id,
        mode=mode,
        model=model,
        context_mode=context_mode,
        context_ids=context_ids,
        context_paragraph_text=context_paragraph_text,
        linked_document_id=linked_document_id,
        document_outline=document_outline,
        linked_document_content=linked_document_content,
        app_tool_registry=app_tool_registry,
    )


async def summarize_app_talk(talks: TalksPort, *, talk_id: str) -> str:
    rows = await talks.load_talk_turns(talk_id)
    if not rows:
        raise ValueError("talk not found")

    transcript_lines: list[str] = []
    for row in rows:
        transcript_lines.append(f"User: {row['user_message']}")
        transcript_lines.append(f"Assistant: {row['assistant_message']}")
    transcript = "\n".join(transcript_lines)

    chat_client = get_deepseek_chat()
    result = await chat_client.chat(
        [
            {
                "role": "system",
                "content": "Fasse das folgende Gespräch in 3–5 Sätzen auf Deutsch zusammen.",
            },
            {"role": "user", "content": transcript},
        ],
        temperature=0.2,
        max_tokens=400,
    )
    summary = result.content.strip()
    await talks.save_talk_summary(talk_id, summary)
    return summary


async def compress_app_talk(talks: TalksPort, *, talk_id: str) -> dict[str, Any]:
    """Welle 5b — „Verdichten": fasst alle Turns außer den letzten beiden zusammen
    und persistiert `compressed_summary`/`compressed_up_to_turn_index`, um den
    angezeigten Kontextverbrauch zu reduzieren."""
    rows = await talks.load_talk_turns(talk_id)
    if len(rows) < 3:
        raise ValueError("talk too short to compress")

    to_compress = rows[:-2]
    up_to_turn_index = to_compress[-1]["turn_index"]

    transcript_lines: list[str] = []
    for row in to_compress:
        transcript_lines.append(f"User: {row['user_message']}")
        transcript_lines.append(f"Assistant: {row['assistant_message']}")
    transcript = "\n".join(transcript_lines)

    chat_client = get_deepseek_chat()
    result = await chat_client.chat(
        [
            {
                "role": "system",
                "content": (
                    "Fasse den folgenden Gesprächsverlauf knapp auf Deutsch zusammen, "
                    "damit er als Kontext-Gedächtnis für die Fortsetzung des Gesprächs dient."
                ),
            },
            {"role": "user", "content": transcript},
        ],
        temperature=0.2,
        max_tokens=600,
    )
    summary = result.content.strip()
    await talks.save_compressed_summary(talk_id, summary, up_to_turn_index)
    return {"summary": summary, "compressed_up_to_turn_index": up_to_turn_index}
