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


@dataclass(slots=True)
class AuthenticConceptExplainResult:
    concept: str
    steiner_prior_text: str
    verify_refs: List[str]
    verification_report: str
    lexicon_entry: str
    graph_event_id: str | None = None


@dataclass(slots=True)
class TranslateToWorldviewResult:
    input_text: str
    worldviews: List[WorldviewAnswer]
    graph_event_id: str | None = None


@dataclass(slots=True)
class EssayCreateStepInfo:
    step: int
    prompt_file: str


@dataclass(slots=True)
class EssayCreateResult:
    assistant: str
    essay_slug: str
    essay_title: str
    final_text: str
    steps: List[EssayCreateStepInfo]


@dataclass(slots=True)
class EssayFinetuneResult:
    assistant: str
    essay_slug: str
    essay_title: str
    revised_text: str


@dataclass(slots=True)
class EssayCompletionResult:
    assistant: str
    essay_slug: str
    essay_title: str
    mood_index: int
    mood_name: str
    header: str
    draft_header: str
    draft_text: str
    revised_header: str
    revised_text: str
    verify_refs: List[str]  # Primary book references
    all_books_refs: List[str]  # Combined primary + secondary references
    references: Optional[List[Mapping[str, Any]]] = None
    verification_report: str = ""  # Deprecated: No longer used in new pipeline
    graph_event_id: str | None = None


@dataclass(slots=True)
class EssayEvaluationResult:
    assistant: str
    essay_slug: str
    essay_title: str
    mood_index: int | None
    mood_name: str | None
    overall_score: int
    criteria_scores: Mapping[str, int]
    issues: List[str]
    instruction: str
    graph_event_id: str | None = None

