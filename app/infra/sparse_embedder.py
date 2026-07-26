"""BM25 sparse embedder using fastembed."""
from __future__ import annotations

import logging
from typing import Sequence, TypedDict

logger = logging.getLogger(__name__)


class SparseVector(TypedDict):
    indices: list[int]
    values: list[float]


class SparseEmbedder:
    """Wraps fastembed BM25 for computing sparse vectors.

    Used at ingestion time (embed_batch) and query time (embed_query).
    The model is loaded lazily on first call.
    """

    VECTOR_NAME = "text-sparse"

    def __init__(self, model_name: str = "Qdrant/bm25") -> None:
        self._model_name = model_name
        self._model = None  # lazy init

    def _get_model(self):
        if self._model is None:
            from fastembed import SparseTextEmbedding  # type: ignore[import]
            logger.info("Loading sparse embedding model: %s", self._model_name)
            self._model = SparseTextEmbedding(model_name=self._model_name)
        return self._model

    def embed_batch(self, texts: Sequence[str]) -> list[SparseVector]:
        """Compute sparse vectors for a batch of texts (for ingestion)."""
        model = self._get_model()
        results = list(model.embed(list(texts)))
        return [
            {"indices": r.indices.tolist(), "values": r.values.tolist()}
            for r in results
        ]

    def embed_query(self, text: str) -> SparseVector:
        """Compute sparse vector for a single query string."""
        model = self._get_model()
        result = next(model.query_embed(text))
        return {"indices": result.indices.tolist(), "values": result.values.tolist()}
