"""Async client for embedding backends (HTTP service or Hugging Face)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import httpx

from app.infra.hf_embedding import DEFAULT_HF_MODEL, HuggingFaceEmbeddingBackend


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
    """HTTP or Hugging Face embedding client with a shared contract."""

    def __init__(
        self,
        base_url: str,
        timeout: float = 60.0,
        batch_size: int = 64,
        *,
        provider: str = "http",
        hf_token: str | None = None,
        hf_model: str = DEFAULT_HF_MODEL,
        hf_api_url: str | None = None,
        hf_max_texts_per_request: int = 8,
        hf_max_retries: int = 4,
        hf_forbid_large_batches: bool = True,
        hf_max_batch_texts: int = 32,
    ) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.timeout = timeout
        self.batch_size = batch_size
        self.provider = (provider or "http").strip().lower()
        self.hf_model = hf_model
        self.hf_forbid_large_batches = hf_forbid_large_batches
        self.hf_max_batch_texts = hf_max_batch_texts
        self._hf: HuggingFaceEmbeddingBackend | None = None
        if self.provider in {"huggingface", "hf"}:
            if not hf_token:
                raise ValueError(
                    "RAGRUN_HF_TOKEN (or HF_TOKEN) is required when "
                    "RAGRUN_EMBEDDINGS_PROVIDER=huggingface"
                )
            self._hf = HuggingFaceEmbeddingBackend(
                token=hf_token,
                model=hf_model,
                api_url=hf_api_url,
                timeout=timeout,
                max_retries=hf_max_retries,
                max_texts_per_request=hf_max_texts_per_request,
            )

    @property
    def is_huggingface(self) -> bool:
        return self._hf is not None

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

        if self._hf is not None:
            if self.hf_forbid_large_batches and len(texts) > self.hf_max_batch_texts:
                raise RuntimeError(
                    "Hugging Face embeddings refuse large batches "
                    f"({len(texts)} > {self.hf_max_batch_texts}). "
                    "Run ingest against a local personal-embeddings-service "
                    "(RAGRUN_EMBEDDINGS_PROVIDER=http, "
                    "RAGRUN_EMBEDDINGS_BASE_URL=http://localhost:8001)."
                )
            vectors = await self._hf.embed_texts(texts)
            dims = len(vectors[0]) if vectors else 0
            return EmbeddingBatchResult(
                embeddings=vectors,
                dimensions=dims,
                model_name=model_name or self.hf_model,
            )

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
                response = await client.post(target_url, json=payload)
                response.raise_for_status()
                data = response.json()
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
