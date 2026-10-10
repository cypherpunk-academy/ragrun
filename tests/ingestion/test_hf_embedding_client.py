"""Tests for Hugging Face embedding client path and ingest guards."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.infra.embedding_client import EmbeddingClient


@pytest.mark.asyncio
async def test_hf_client_embeds_via_backend():
    client = EmbeddingClient(
        "http://unused",
        provider="huggingface",
        hf_token="hf_test",
        hf_model="intfloat/multilingual-e5-large",
        hf_max_batch_texts=32,
    )
    fake_vecs = [[0.0] * 1023 + [1.0]]
    assert client._hf is not None
    with patch.object(client._hf, "embed_texts", new=AsyncMock(return_value=fake_vecs)) as mocked:
        result = await client.embed_texts(["query: test"])
    mocked.assert_awaited_once()
    assert result.dimensions == 1024
    assert result.model_name == "intfloat/multilingual-e5-large"
    assert result.embeddings == fake_vecs


@pytest.mark.asyncio
async def test_hf_client_refuses_large_batches():
    client = EmbeddingClient(
        "http://unused",
        provider="huggingface",
        hf_token="hf_test",
        hf_max_batch_texts=2,
    )
    with pytest.raises(RuntimeError, match="refuse large batches"):
        await client.embed_texts(["a", "b", "c"])


def test_hf_client_requires_token():
    with pytest.raises(ValueError, match="HF_TOKEN"):
        EmbeddingClient("http://unused", provider="huggingface", hf_token=None)
