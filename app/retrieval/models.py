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
class AuthenticConceptExplainResult:
    concept: str
    steiner_prior_text: str
    verify_refs: List[str]
    verification_report: str
    lexicon_entry: str
    references: Optional[List[Mapping[str, Any]]] = None
    graph_event_id: str | None = None


@dataclass(slots=True)
class TypologyExplainResult:
    name: str
    members: List[str]
    explanation: str
    aliases: List[str]
    references: Optional[List[Mapping[str, Any]]] = None
    graph_event_id: str | None = None


@dataclass(slots=True)
class ThoughtExplainResult:
    thought_id: str
    explanation: str
    word_count: int
    references: Optional[List[Mapping[str, Any]]] = None
    graph_event_id: str | None = None

