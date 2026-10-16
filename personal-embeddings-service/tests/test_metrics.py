"""Prometheus /metrics without loading a real embedding model."""
import numpy as np
import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from app.config import settings


@pytest.fixture(autouse=True)
def fake_embedding_backend(monkeypatch):
    def fake_load(self, model_name=None):
        return True

    def fake_encode(self, texts, model_name=None):
        if isinstance(texts, str):
            return np.ones(settings.embedding_dimension, dtype=np.float32)
        return np.ones((len(texts), settings.embedding_dimension), dtype=np.float32)

    monkeypatch.setattr(
        "app.models.embedding_model.EmbeddingModel.load_model", fake_load, raising=False
    )
    monkeypatch.setattr(
        "app.models.embedding_model.EmbeddingModel.encode", fake_encode, raising=False
    )


def _sample(name: str, labels: dict | None = None) -> float:
    labels = labels or {}
    total = 0.0
    found = False
    for metric in REGISTRY.collect():
        for sample in metric.samples:
            if sample.name != name:
                continue
            if all(sample.labels.get(key) == value for key, value in labels.items()):
                total += sample.value
                found = True
    return total if found else 0.0


def test_metrics_endpoint_and_embed_counters():
    from app.main import app

    secret = "zz-metrics-secret-phrase-zz"
    with TestClient(app) as client:
        first = client.get("/metrics")
        assert first.status_code == 200
        content_type = first.headers["content-type"]
        assert content_type.startswith("text/plain")
        assert "version=0.0.4" in content_type

        requests_before = _sample(
            "embedder_requests_total",
            {"endpoint": "/api/v1/embeddings", "status": "200"},
        )
        encode_before = _sample("embedder_encode_seconds_count")

        response = client.post("/api/v1/embeddings", json={"texts": [secret]})
        assert response.status_code == 200

        scraped = client.get("/metrics")
        assert scraped.status_code == 200
        body = scraped.text
        assert secret not in body
        assert "embedder_requests_total" in body
        assert "embedder_encode_seconds" in body

    assert (
        _sample(
            "embedder_requests_total",
            {"endpoint": "/api/v1/embeddings", "status": "200"},
        )
        == requests_before + 1
    )
    assert _sample("embedder_encode_seconds_count") == encode_before + 1
