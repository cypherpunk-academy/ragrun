from __future__ import annotations

from typing import Any, Mapping

import pytest
from fastapi.testclient import TestClient

from app.api import concept_explain as concept_router
from app.main import app
from app.services.concept_explain_service import ConceptExplainResult, RetrievedSnippet


class StubConceptExplainService:
    def __init__(self) -> None:
        self.calls = []

    async def explain(self, concept: str) -> ConceptExplainResult:
        self.calls.append(concept)
        stub_hit = RetrievedSnippet(
            text="stub text",
            score=0.9,
            payload={"payload": {"metadata": {"chunk_type": "book", "source_id": "s1"}}},
        )
        return ConceptExplainResult(
            concept=concept,
            answer=f"Erklärung zu {concept}",
            retrieved=[stub_hit],
            expanded=[],
        )


@pytest.fixture
def client_with_stub():
    stub = StubConceptExplainService()
    app.dependency_overrides[concept_router.get_concept_explain_service] = lambda: stub
    client = TestClient(app)
    yield client, stub
    app.dependency_overrides.pop(concept_router.get_concept_explain_service, None)


def test_concept_explain_happy_path(client_with_stub):
    client, stub = client_with_stub
    resp = client.post(
        "/api/v1/agent/philo-von-freisinn/retrieval/concept-explain",
        json={"concept": "Seele"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["concept"] == "Seele"
    assert "Erklärung zu Seele" in body["answer"]
    assert len(body["retrieved"]) == 1
    assert stub.calls == ["Seele"]


def test_concept_explain_requires_concept(client_with_stub):
    client, _ = client_with_stub
    resp = client.post(
        "/api/v1/agent/philo-von-freisinn/retrieval/concept-explain",
        json={"concept": ""},
    )
    assert resp.status_code == 400
    assert "concept is required" in resp.json()["detail"]


def test_chat_endpoint_routes_to_concept_branch(client_with_stub):
    client, stub = client_with_stub
    resp = client.post(
        "/api/v1/agent/philo-von-freisinn/chat",
        json={"prompt": "Was ist Seele?"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["branch"] == "concept-explain"
    assert body["concept"] == "Was ist Seele?"
    assert "Erklärung zu Was ist Seele?" in body["answer"]
    assert stub.calls == ["Was ist Seele?"]


def test_chat_endpoint_rejects_empty_prompt(client_with_stub):
    client, _ = client_with_stub
    resp = client.post("/api/v1/agent/philo-von-freisinn/chat", json={"prompt": ""})
    assert resp.status_code == 400
    assert "prompt is required" in resp.json()["detail"]
