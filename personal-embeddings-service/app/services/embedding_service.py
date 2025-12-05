import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Union

import numpy as np

from app.config import settings
from app.models.embedding_model import EmbeddingModel

logger = logging.getLogger(__name__)


class LocalEmbeddingService:
    """Asynchronous embedding service using the configured model(s)."""

    def __init__(self) -> None:
        self.model = EmbeddingModel()
        self.executor = ThreadPoolExecutor(max_workers=settings.max_workers)
        self.ready = False

    async def load_model(self, model_name: Optional[str] = None) -> None:
        """Load the default (or specific) model asynchronously."""

        def _load_model() -> bool:
            return self.model.load_model(model_name)

        target = model_name or settings.model_name
        logger.info("Loading %s model asynchronously", target)
        success = await asyncio.get_event_loop().run_in_executor(
            self.executor, _load_model
        )
        if success:
            self.ready = True
            logger.info("Embedding service ready")
        else:
            raise RuntimeError("Failed to load embedding model %s", target)

    def resolve_model_name(self, requested: Optional[str]) -> str:
        """Validate and normalize the requested model name."""

        return self.model.normalize_model_name(requested)

    async def encode_texts(
        self, texts: Union[str, List[str]], model_name: Optional[str] = None
    ) -> np.ndarray:
        """Encode texts to embeddings asynchronously."""
        if not self.ready:
            raise RuntimeError("Service not ready. Call load_model() first.")

        resolved = self.resolve_model_name(model_name)

        def _encode(texts_input: Union[str, List[str]]) -> np.ndarray:
            return self.model.encode(texts_input, resolved)

        embeddings = await asyncio.get_event_loop().run_in_executor(
            self.executor, _encode, texts
        )
        return embeddings

    async def similarity_search(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5,
        model_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Perform similarity search using local embeddings."""
        try:
            query_embedding = await self.encode_texts(query, model_name=model_name)
            doc_embeddings = await self.encode_texts(documents, model_name=model_name)

            if query_embedding.ndim == 2:
                query_embedding = query_embedding[0]

            similarities = np.dot(doc_embeddings, query_embedding) / (
                np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_embedding)
            )

            top_indices = np.argsort(similarities)[::-1][:top_k]

            results = []
            for idx in top_indices:
                results.append(
                    {
                        "index": int(idx),
                        "document": documents[idx],
                        "score": float(similarities[idx]),
                    }
                )

            return results

        except Exception as e:
            logger.error("Failed to perform similarity search: %s", e)
            raise

    def get_service_info(self) -> Dict[str, Any]:
        """Return service information."""
        return {
            "ready": self.ready,
            "models": self.model.get_model_info(),
            "default_model": self.model.default_model_name,
            "max_workers": settings.max_workers,
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform a health check with a test embedding."""
        try:
            if not self.ready:
                return {"status": "unhealthy", "error": "Service not ready"}

            start_time = time.time()
            test_embedding = await self.encode_texts("health check test")
            processing_time = time.time() - start_time

            if test_embedding.ndim == 1:
                dimension = test_embedding.shape[0]
            else:
                dimension = test_embedding.shape[1]

            return {
                "status": "healthy",
                "model": self.model.default_model_name,
                "device": self.model.device,
                "embedding_dimension": int(dimension),
                "test_processing_time": processing_time,
            }

        except Exception as e:
            logger.error("Health check failed: %s", e)
            return {"status": "unhealthy", "error": str(e)}


embedding_service = LocalEmbeddingService()