"""API endpoint for the automated problem-solver Socratic dialog."""
from __future__ import annotations

import json
import logging

from fastapi import APIRouter
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from app.retrieval.graphs.problem_solver_graph import run_problem_solver
from app.retrieval.services.action_prompt_service import load_assistant_name
from app.retrieval.services.providers import (
    get_deepseek_chat,
    get_embedding_client,
    get_qdrant_client,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["problem-solver"])


class ProblemSolverRequest(BaseModel):
    assistant_slug: str | None = Field(None, description="Defaults to path param if omitted")
    user_prompt: str = Field(
        default="",
        description="User-supplied problem text. If empty, problem_text is used instead.",
    )
    problem_text: str = Field(
        default="",
        description="TEXT 2 from the last summarize action (used when user_prompt is empty).",
    )
    language: str = Field("de-DE", description="BCP 47 locale")


@router.post("/agent/{assistant_slug}/problem-solver")
async def solve_problem(
    assistant_slug: str,
    body: ProblemSolverRequest,
) -> EventSourceResponse:
    """
    Run the automated problem-solver Socratic dialog.

    Uses *user_prompt* as the problem text if provided; falls back to *problem_text*
    (TEXT 2 from a preceding summarize action). Streams SSE events:
      - {type: "start"}
      - {type: "digest", step, role, digest}  — one per LLM call, max 7
      - {type: "done", agreed, steps, full_dialog, final_answer}
      - {type: "error", message}              — on failure
    """
    effective_slug = body.assistant_slug or assistant_slug
    problem_text = body.user_prompt.strip() or body.problem_text.strip()
    assistant_name = load_assistant_name(effective_slug)

    embedding_client = get_embedding_client()
    qdrant_client = get_qdrant_client()
    chat_client = get_deepseek_chat()

    async def event_generator():
        try:
            async for event in run_problem_solver(
                problem_text=problem_text,
                assistant_slug=effective_slug,
                collection=effective_slug,
                embedding_client=embedding_client,
                qdrant_client=qdrant_client,
                chat_client=chat_client,
            ):
                if event.get("type") == "start":
                    event = {**event, "assistant_name": assistant_name}
                yield {"data": json.dumps(event, ensure_ascii=False)}
        except Exception as exc:
            logger.exception("problem-solver failed: %s", exc)
            yield {"data": json.dumps({"type": "error", "message": str(exc)})}

    return EventSourceResponse(event_generator())
