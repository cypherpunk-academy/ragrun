"""Async client for the personal embedding service."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import httpx

from app.debug_agent_log import agent_log


def _chunk_list(items: Sequence[str], chunk_size: int) -> Iterable[List[str]]:
    """Yield successive slices from a sequence."""

    for idx in range(0, len(items), chunk_size):
        yield list(items[idx : idx + chunk_size])


@dataclass(slots=True)
class EmbeddingBatchResult:
    """Normalized response from the embedding service."""

    embeddings: List[List[float]]
    dimensions: int
    model_name: str


class EmbeddingClient:
    """Simple HTTP client for the personal embedding service."""

    def __init__(self, base_url: str, timeout: float = 60.0, batch_size: int = 64) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.timeout = timeout
        self.batch_size = batch_size

    async def embed_texts(
        self,
        texts: Sequence[str],
        *,
        model_name: str | None = None,
        batch_size: int | None = None,
    ) -> EmbeddingBatchResult:
        """Embed a sequence of texts, chunking requests for throughput."""

        if not texts:
            raise ValueError("at least one text is required for embeddings")

        resolved_batch_size = batch_size or self.batch_size
        all_embeddings: List[List[float]] = []
        resolved_model = model_name
        dimensions = 0

        async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout, connect=10.0)) as client:
            for chunk in _chunk_list(texts, resolved_batch_size):
                payload: dict[str, object] = {"texts": chunk}
                if model_name:
                    payload["model"] = model_name
                target_url = f"{self.base_url}/api/v1/embeddings"
                # region agent log
                agent_log(
                    location="embedding_client.py:embed_texts:before_post",
                    message="embedding request",
                    data={"target_url": target_url, "batch_size": len(chunk)},
                    hypothesis_id="C",
                )
                # endregion
                try:
                    response = await client.post(target_url, json=payload)
                    response.raise_for_status()
                    data = response.json()
                except httpx.ConnectTimeout as exc:
                    # region agent log
                    agent_log(
                        location="embedding_client.py:embed_texts:connect_timeout",
                        message="embedding ConnectTimeout",
                        data={"target_url": target_url, "exc": str(exc)},
                        hypothesis_id="C",
                    )
                    # endregion
                    raise
                chunk_embeddings = data.get("embeddings")
                if not isinstance(chunk_embeddings, list):
                    raise RuntimeError("embedding service returned malformed payload")

                all_embeddings.extend(chunk_embeddings)
                dimensions = int(data.get("dimensions") or 0)
                resolved_model = str(data.get("model") or resolved_model or "")

        if not all_embeddings:
            raise RuntimeError("embedding service returned no embeddings")

        if dimensions <= 0:
            dimensions = len(all_embeddings[0])

        resolved_model = resolved_model or "unknown"

        return EmbeddingBatchResult(
            embeddings=all_embeddings,
            dimensions=dimensions,
            model_name=resolved_model,
        )

