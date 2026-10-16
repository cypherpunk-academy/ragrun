"""Guardrail: block /rag/embed-chunks while embeddings provider is Hugging Face."""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.api.rag import EmbedChunksRequest, _reject_hf_batch_ingest
from app.config import settings


def test_reject_hf_batch_ingest_when_forbidden(monkeypatch):
    monkeypatch.setattr(settings, "embeddings_provider", "huggingface")
    monkeypatch.setattr(settings, "embeddings_hf_forbid_ingest", True)
    request = EmbedChunksRequest(collection_name="philo-von-freisinn")
    with pytest.raises(HTTPException) as exc:
        _reject_hf_batch_ingest(request)
    assert exc.value.status_code == 400
    assert "huggingface" in str(exc.value.detail).lower()


def test_allow_cleanup_only_on_huggingface(monkeypatch):
    monkeypatch.setattr(settings, "embeddings_provider", "huggingface")
    monkeypatch.setattr(settings, "embeddings_hf_forbid_ingest", True)
    request = EmbedChunksRequest(
        collection_name="philo-von-freisinn",
        cleanup_only=True,
    )
    _reject_hf_batch_ingest(request)  # does not raise


def test_allow_ingest_on_http_provider(monkeypatch):
    monkeypatch.setattr(settings, "embeddings_provider", "http")
    monkeypatch.setattr(settings, "embeddings_hf_forbid_ingest", True)
    request = EmbedChunksRequest(collection_name="philo-von-freisinn")
    _reject_hf_batch_ingest(request)


def test_allow_ingest_when_forbid_flag_false(monkeypatch):
    monkeypatch.setattr(settings, "embeddings_provider", "huggingface")
    monkeypatch.setattr(settings, "embeddings_hf_forbid_ingest", False)
    request = EmbedChunksRequest(collection_name="philo-von-freisinn")
    _reject_hf_batch_ingest(request)
