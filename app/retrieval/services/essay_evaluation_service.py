"""Service layer for essay_evaluation graph."""
from __future__ import annotations

from app.infra.deepseek_client import DeepSeekClient
from app.retrieval.chains.essay_evaluation import run_essay_evaluation_chain
from app.retrieval.models import EssayEvaluationResult


class EssayEvaluationService:
    def __init__(self, *, chat_client: DeepSeekClient) -> None:
        self.chat_client = chat_client

    async def evaluate(
        self,
        *,
        assistant: str,
        essay_slug: str,
        essay_title: str,
        text: str,
        mood_index: int | None,
        mood_name: str | None,
        verbose: bool = False,
        llm_retries: int = 3,
    ) -> EssayEvaluationResult:
        return await run_essay_evaluation_chain(
            assistant=assistant,
            essay_slug=essay_slug,
            essay_title=essay_title,
            text=text,
            mood_index=mood_index,
            mood_name=mood_name,
            chat_client=self.chat_client,
            verbose=verbose,
            llm_retries=llm_retries,
        )
