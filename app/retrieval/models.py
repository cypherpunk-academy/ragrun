"""Shared retrieval data models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping


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

