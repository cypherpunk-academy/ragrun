"""Optional cross-encoder for GA search. Not part of the API image.

POST /rerank  {"query": "...", "texts": ["...", "..."]}
-> {"scores": [0.1, 0.9]}  raw model scores; ragrun normalizes them.
"""
from __future__ import annotations

import os

from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder

MODEL_NAME = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

app = FastAPI(title="ragrun-ga-rerank")
_model: CrossEncoder | None = None


class RerankRequest(BaseModel):
    query: str
    texts: list[str] = Field(default_factory=list)


@app.on_event("startup")
def _load() -> None:
    global _model
    _model = CrossEncoder(MODEL_NAME)


@app.get("/health")
def health() -> dict[str, object]:
    return {"ok": _model is not None, "model": MODEL_NAME}


@app.post("/rerank")
def rerank(body: RerankRequest) -> dict[str, list[float]]:
    if _model is None:
        raise RuntimeError("rerank model is not loaded")
    if not body.texts:
        return {"scores": []}
    raw = _model.predict([(body.query, text) for text in body.texts])
    return {"scores": [float(score) for score in raw]}
