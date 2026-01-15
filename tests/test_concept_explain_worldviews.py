import asyncio
from types import SimpleNamespace

import pytest

from app.retrieval.graphs.concept_explain_worldviews import (
    RetrievalConfig,
    run_concept_explain_worldviews_graph,
)
from app.retrieval.models import WorldviewAnswer


class FakeEmbeddingClient:
    async def embed_texts(self, texts, *, model_name=None, batch_size=None):
        # Return unit vectors of fixed size
        emb = [1.0] * 4
        return SimpleNamespace(embeddings=[emb for _ in texts], dimensions=4, model_name="fake")


class FakeQdrantClient:
    def __init__(self) -> None:
        self.counter = 0

    async def search_points(self, collection, vector, limit=10, filter_=None, with_payload=True):
        self.counter += 1
        # produce distinct text to simulate hits
        hits = []
        for i in range(limit):
            hits.append(
                {
                    "score": 1.0 / (i + 1),
                    "payload": {"text": f"snippet-{self.counter}-{i}", "chunk_id": f"c{self.counter}-{i}"},
                }
            )
        return hits


class FakeDeepSeekClient:
    def __init__(self, tag: str):
        self.tag = tag

    async def chat(self, messages, temperature=0.2, max_tokens=300):
        # echo short content for determinism
        last = messages[-1]["content"] if messages else ""
        return f"{self.tag}:{last[:40]}"

    async def list_models(self):
        return [self.tag]


@pytest.mark.asyncio
async def test_worldviews_graph_basic():
    embedding = FakeEmbeddingClient()
    qdrant = FakeQdrantClient()
    reasoner = FakeDeepSeekClient("reasoner")
    chat = FakeDeepSeekClient("chat")
    cfg = RetrievalConfig(k_base_concept=4, k_base_context1=3, k_base_context2=4, k_final_concept=2, k_final_context1=2, k_final_context2=2)

    result = await run_concept_explain_worldviews_graph(
        concept="Testbegriff",
        worldviews=["WV1", "WV2"],
        collection="test",
        embedding_client=embedding,
        qdrant_client=qdrant,
        reasoning_client=reasoner,
        chat_client=chat,
        cfg=cfg,
        hybrid=False,
        max_concurrency=2,
    )

    assert result.concept == "Testbegriff"
    assert len(result.worldviews) == 2
    assert all(isinstance(w, WorldviewAnswer) for w in result.worldviews)
    # Ensure reducer didn't drop any worldview
    names = sorted([w.worldview for w in result.worldviews])
    assert names == ["WV1", "WV2"]
    # Check sufficiency field exists
    assert all(w.sufficiency for w in result.worldviews)
