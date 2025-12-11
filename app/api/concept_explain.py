"""Compatibility wrapper re-exporting the philo-von-freisinn agent router."""
from __future__ import annotations

from app.retrieval.api.philo_von_freisinn import (  # noqa: F401
    AssistantChatRequest,
    AssistantChatResponse,
    ConceptExplainRequest,
    ConceptExplainResponse,
    RetrievedItem,
    concept_explain,
    get_concept_explain_service,
    router,
    assistant_chat,
)
