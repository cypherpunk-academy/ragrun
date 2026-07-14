"""Non-streaming chat for /app/chat."""
from __future__ import annotations

from typing import Any

from app.config import settings
from app.db.ports import TalksPort
from app.retrieval.services.action_prompt_service import (
    _load_personality_system,
    load_assistant_name,
    load_assistant_rag_collection,
)
from app.retrieval.services.providers import get_deepseek_chat
from app.services.app_search_service import app_search


async def send_app_chat(
    talks: TalksPort,
    *,
    user_id: str,
    user_name: str,
    message: str,
    personality: str,
    talk_id: str | None = None,
    context_mode: str | None = None,
    context_ids: dict[str, Any] | None = None,
) -> dict[str, str]:
    msg = message.strip()
    if not msg:
        raise ValueError("message must not be empty")
    personality_slug = personality.strip()
    if not personality_slug:
        raise ValueError("personality must not be empty")

    assistant_slug = settings.app_default_assistant_slug
    collection = load_assistant_rag_collection(assistant_slug)
    assistant_name = load_assistant_name(assistant_slug)

    system_parts: list[str] = []
    personality_system = _load_personality_system(personality_slug)
    if personality_system:
        system_parts.append(personality_system.replace("{assistant_name}", assistant_name))
    else:
        system_parts.append(
            f"Du bist {personality_slug}, ein hilfreicher Gesprächspartner in der App."
        )

    context_block = ""
    ctx = context_ids or {}
    if context_mode == "paragraph" and ctx.get("paragraph_id"):
        context_block = f"\n\nKontext-Absatz: {ctx.get('paragraph_id')}"
    elif context_mode == "segment" and ctx.get("segment_id"):
        context_block = f"\n\nKontext-Segment: {ctx.get('segment_id')}"

    retrieval_block = ""
    try:
        hits = await app_search(query=msg, limit=4, collection=collection)
        if hits:
            lines = []
            for h in hits[:4]:
                lines.append(f"- [{h.get('chunk_id')}] {h.get('snippet', '')}")
            retrieval_block = "\n\nRelevante Quellen:\n" + "\n".join(lines)
    except Exception:
        retrieval_block = ""

    system_prompt = "".join(system_parts) + context_block + retrieval_block
    chat_client = get_deepseek_chat()
    result = await chat_client.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": msg},
        ],
        temperature=0.4,
        max_tokens=1200,
    )

    title = msg[:120]
    saved = await talks.create_talk_turn(
        user_id=user_id,
        user_name=user_name or user_id,
        collection=collection,
        personality=personality_slug,
        title=title,
        user_message=msg,
        assistant_message=result.content,
        talk_id=talk_id,
        kontext_meta=ctx if ctx else None,
        usage={
            "model": settings.deepseek_chat_model or "deepseek-v4-flash",
            **result.usage,
        },
    )
    return {
        "talk_id": saved["talk_id"],
        "turn_id": saved["turn_id"],
        "reply": result.content,
    }


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
