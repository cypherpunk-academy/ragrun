"""Service to retrieve context and generate concept explanations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from app.services.deepseek_client import DeepSeekClient
from app.services.embedding_client import EmbeddingClient
from app.services.qdrant_client import QdrantClient


SUMMARY_TYPES = {
    "chapter_summary",
    "talk_summary",
    "essay_summary",
    "explanation_summary",
}


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


class ConceptExplainService:
    """Retrieval + generation pipeline for concept explanations."""

    def __init__(
        self,
        embedding_client: EmbeddingClient,
        qdrant_client: QdrantClient,
        deepseek_client: DeepSeekClient,
        *,
        collection: str = "philo-von-freisinn",
        k: int = 10,
    ) -> None:
        self.embedding_client = embedding_client
        self.qdrant_client = qdrant_client
        self.deepseek_client = deepseek_client
        self.collection = collection
        self.k = k

    async def _embed(self, text: str) -> Sequence[float]:
        result = await self.embedding_client.embed_texts([text])
        return result.embeddings[0]

    def _qdrant_filter_for_source(self, source_id: str) -> Mapping[str, object]:
        return {"must": [{"key": "metadata.source_id", "match": {"value": source_id}}]}

    async def _expand_summaries(
        self, hits: List[RetrievedSnippet]
    ) -> List[RetrievedSnippet]:
        expanded: List[RetrievedSnippet] = []
        source_ids = {
            str(h.payload.get("payload", {}).get("metadata", {}).get("source_id"))
            for h in hits
            if h.payload.get("payload", {})
            .get("metadata", {})
            .get("chunk_type") in SUMMARY_TYPES
        }
        for src in source_ids:
            if not src:
                continue
            scroll = await self.qdrant_client.scroll_points(
                self.collection,
                filter_=self._qdrant_filter_for_source(src),
                limit=256,
                with_payload=True,
                with_vectors=False,
            )
            for item in scroll:
                payload = item.get("payload", {})
                text = payload.get("text") or ""
                if not text:
                    continue
                expanded.append(
                    RetrievedSnippet(
                        text=text,
                        score=float(item.get("score") or 0.0),
                        payload=item,
                    )
                )
        return expanded

    def _build_prompt(self, concept: str, primary: List[RetrievedSnippet], expanded: List[RetrievedSnippet]) -> List[Mapping[str, str]]:
        context_lines: List[str] = []
        if primary:
            context_lines.append("Primäre Treffer:")
            for hit in primary:
                context_lines.append(f"- {hit.text}")
        if expanded:
            context_lines.append("\nZusätzliche Informationen:")
            for hit in expanded:
                context_lines.append(f"- {hit.text}")
        context_block = "\n".join(context_lines)

        system = "\n".join(
            [
                'Du bist „Philo-von-Freisinn“:',
                "- Philosophischer Assistent, Fokus auf individuelle Freiheit, logisch präzise.",
                "- Ton: klar, zugänglich, lebendig und humorvoll, aber ohne Ironie oder Personalisierungen.",
                "- Keine Fremdworte, die Steiner nicht nutzt.",
                "Aufgabe: Erkläre einen Begriff im Kontext der Sammlung Rudolf Steiner (Primär- und Sekundärliteratur plus Augmentierungen wie summaries, concepts).",
            ]
        )
        user = "\n".join(
            [
                f'Erkläre den Begriff: "{concept}".',
                "Schreibe für eine 16-jährige Leserin.",
                "Nutze nur den gelieferten Kontext; erfinde nichts.",
                "Keine Zitate, sondern erklärend zusammenfassen.",
                "",
                "Kontext:",
                context_block,
            ]
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    async def explain(self, concept: str) -> ConceptExplainResult:
        vector = await self._embed(concept)
        search_results = await self.qdrant_client.search_points(
            self.collection, vector=vector, limit=self.k, with_payload=True
        )
        primary_hits: List[RetrievedSnippet] = []
        for item in search_results:
            payload = item.get("payload", {})
            text = payload.get("text") or ""
            if not text:
                continue
            score = float(item.get("score") or 0.0)
            primary_hits.append(RetrievedSnippet(text=text, score=score, payload=item))

        expanded_hits = await self._expand_summaries(primary_hits)

        messages = self._build_prompt(concept, primary_hits, expanded_hits)
        answer = await self.deepseek_client.chat(messages, temperature=0.2, max_tokens=300)

        return ConceptExplainResult(
            concept=concept,
            answer=answer,
            retrieved=primary_hits,
            expanded=expanded_hits,
        )
