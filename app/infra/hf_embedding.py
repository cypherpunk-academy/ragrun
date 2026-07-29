"""Hugging Face serverless feature-extraction helpers for e5-style embeddings.

Matches the local personal-embeddings-service pipeline:
mean-token pooling + L2 normalization (see EmbeddingModel.encode).
"""
from __future__ import annotations

import asyncio
import logging
import math
from typing import Any, List, Sequence

import httpx

logger = logging.getLogger(__name__)

DEFAULT_HF_MODEL = "intfloat/multilingual-e5-large"
DEFAULT_HF_FEATURE_URL = (
    "https://router.huggingface.co/hf-inference/models/"
    "{model}/pipeline/feature-extraction"
)


def _l2_normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm <= 0.0:
        return vec
    return [x / norm for x in vec]


def _mean_pool(token_vectors: Sequence[Sequence[float]]) -> List[float]:
    if not token_vectors:
        raise ValueError("cannot pool empty token matrix")
    dim = len(token_vectors[0])
    if dim <= 0:
        raise ValueError("empty embedding dimension")
    acc = [0.0] * dim
    for row in token_vectors:
        if len(row) != dim:
            raise ValueError("ragged token embedding matrix")
        for i, v in enumerate(row):
            acc[i] += float(v)
    n = float(len(token_vectors))
    return [v / n for v in acc]


def pool_hf_feature_output(raw: Any) -> List[float]:
    """Normalize HF feature-extraction JSON into one L2-normalized vector.

    Accepts:
    - 1D: already pooled sentence vector
    - 2D: token x hidden (mean-pool)
    - 3D batch of size 1: unwrap then handle as 1D/2D
    """
    if not isinstance(raw, list) or not raw:
        raise ValueError("HF feature-extraction returned empty/non-list payload")

    # Unwrap single-item batch: [[tokens...]] or [[dims...]]
    data: Any = raw
    if isinstance(raw[0], list) and raw and all(isinstance(x, list) for x in raw):
        # Could be batch of vectors or token matrix.
        if raw and isinstance(raw[0][0], (int, float)):
            # 2D: either (tokens, dim) or (batch, dim) if batch of pooled vectors.
            # Heuristic: if len(raw) looks like sequence and dim == 1024-ish, treat as tokens
            # when first element is a list of floats AND there are many rows relative to 1.
            # For a single text, HF usually returns (seq, dim). For batch of N pooled, (N, dim).
            # Caller handles multi-text separately; this function is one embedding.
            first = raw[0]
            if first and isinstance(first[0], (int, float)) and not isinstance(first[0], list):
                # Distinguish pooled batch vs token matrix: token matrices have seq_len >> 1
                # and we always mean-pool 2D float matrices here when called for one input's
                # raw output. If the API already pooled, seq_len is 1 or we get 1D.
                if len(raw) == 1 and isinstance(raw[0][0], (int, float)):
                    # [[f, f, ...]] — one pooled vector wrapped
                    return _l2_normalize([float(x) for x in raw[0]])
                # (seq, dim) token matrix OR (batch, dim). For single-input API calls we
                # always mean-pool 2D matrices (matches local SentenceTransformer pooling).
                return _l2_normalize(_mean_pool(raw))
        if isinstance(raw[0][0], list):
            # 3D: batch x tokens x dim — take first item
            data = raw[0]

    if isinstance(data, list) and data and isinstance(data[0], (int, float)):
        return _l2_normalize([float(x) for x in data])

    if isinstance(data, list) and data and isinstance(data[0], list):
        if data[0] and isinstance(data[0][0], (int, float)):
            return _l2_normalize(_mean_pool(data))

    raise ValueError(f"unsupported HF feature-extraction shape: {type(raw)}")


def pool_hf_batch_output(raw: Any, expected: int) -> List[List[float]]:
    """Pool a HF response that may contain one or many embeddings."""
    if expected <= 0:
        raise ValueError("expected must be positive")

    # Single embedding payloads
    if expected == 1:
        return [pool_hf_feature_output(raw)]

    if not isinstance(raw, list) or not raw:
        raise ValueError("HF batch feature-extraction returned empty payload")

    # Batch of pooled vectors: [[dim], [dim], ...]
    if (
        len(raw) == expected
        and isinstance(raw[0], list)
        and raw[0]
        and isinstance(raw[0][0], (int, float))
    ):
        return [_l2_normalize([float(x) for x in row]) for row in raw]

    # Batch of token matrices: [[[tok, dim], ...], ...]
    if (
        len(raw) == expected
        and isinstance(raw[0], list)
        and raw[0]
        and isinstance(raw[0][0], list)
    ):
        return [pool_hf_feature_output(item) for item in raw]

    # Some endpoints return a flat token matrix for a single string even when we
    # sent a list — fall back to per-item requests at the caller.
    raise ValueError(
        f"unexpected HF batch shape for {expected} texts "
        f"(got top-level len={len(raw) if isinstance(raw, list) else 'n/a'})"
    )


class HuggingFaceEmbeddingBackend:
    """Async HF Inference feature-extraction client with retries."""

    def __init__(
        self,
        *,
        token: str,
        model: str = DEFAULT_HF_MODEL,
        api_url: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 4,
        max_texts_per_request: int = 8,
    ) -> None:
        if not token:
            raise ValueError("Hugging Face token is required")
        self.token = token
        self.model = model
        self.api_url = (api_url or DEFAULT_HF_FEATURE_URL.format(model=model)).rstrip(
            "/"
        )
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_texts_per_request = max(1, max_texts_per_request)

    async def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            raise ValueError("at least one text is required for embeddings")
        out: List[List[float]] = []
        for start in range(0, len(texts), self.max_texts_per_request):
            chunk = list(texts[start : start + self.max_texts_per_request])
            out.extend(await self._embed_chunk(chunk))
        return out

    async def _embed_chunk(self, texts: List[str]) -> List[List[float]]:
        # Prefer one request with list inputs; fall back to per-text on shape errors.
        try:
            raw = await self._post(texts if len(texts) > 1 else texts[0])
            return pool_hf_batch_output(raw, expected=len(texts))
        except ValueError as exc:
            if len(texts) == 1:
                raise
            logger.warning(
                "HF batch pooling failed (%s); falling back to per-text requests",
                exc,
            )
            vectors: List[List[float]] = []
            for text in texts:
                raw = await self._post(text)
                vectors.append(pool_hf_feature_output(raw))
            return vectors

    async def _post(self, inputs: str | List[str]) -> Any:
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        payload = {"inputs": inputs}
        last_error: Exception | None = None
        async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout, connect=10.0)) as client:
            for attempt in range(self.max_retries + 1):
                try:
                    response = await client.post(
                        self.api_url, headers=headers, json=payload
                    )
                    if response.status_code in (429, 503, 502):
                        retry_after = float(response.headers.get("Retry-After") or 0)
                        delay = retry_after or min(2**attempt, 20)
                        logger.warning(
                            "HF embeddings %s (attempt %d/%d); sleeping %.1fs",
                            response.status_code,
                            attempt + 1,
                            self.max_retries + 1,
                            delay,
                        )
                        await asyncio.sleep(delay)
                        last_error = RuntimeError(
                            f"HF embeddings HTTP {response.status_code}: {response.text[:200]}"
                        )
                        continue
                    if response.status_code >= 400:
                        raise RuntimeError(
                            f"HF embeddings HTTP {response.status_code}: {response.text[:400]}"
                        )
                    return response.json()
                except httpx.HTTPError as exc:
                    delay = min(2**attempt, 20)
                    logger.warning(
                        "HF embeddings transport error %s (attempt %d/%d); sleeping %.1fs",
                        exc,
                        attempt + 1,
                        self.max_retries + 1,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    last_error = exc
            assert last_error is not None
            raise RuntimeError(f"HF embeddings failed after retries: {last_error}") from last_error
