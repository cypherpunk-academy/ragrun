"""Backwards-compatible re-export of the retrieval ConceptExplainService."""
from __future__ import annotations

from app.retrieval.models import ConceptExplainResult, RetrievedSnippet  # noqa: F401
from app.retrieval.services.concept_explain_service import ConceptExplainService  # noqa: F401
