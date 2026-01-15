"""Shared retrieval data models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, Optional


@dataclass(slots=True)
class RetrievedSnippet:
    text: str
    score: float
    payload: Mapping[str, Any]


@dataclass(slots=True)
class ConceptExplainResult:
    concept: str
    answer: str
    retrieved: List[RetrievedSnippet]
    expanded: List[RetrievedSnippet]


@dataclass(slots=True)
class WorldviewAnswer:
    worldview: str
    main_points: str
    how_details: str
    context1_refs: List[str]
    context2_refs: List[str]
    sufficiency: str = "medium"
    errors: Optional[List[str]] = None


@dataclass(slots=True)
class ConceptExplainWorldviewsResult:
    concept: str
    concept_explanation: str
    worldviews: List[WorldviewAnswer]
    context_refs: List[str]
    graph_event_id: str | None = None

