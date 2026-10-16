"""Unit tests for HF feature-extraction pooling (local SentenceTransformer parity)."""
from __future__ import annotations

import math

import pytest

from app.infra.hf_embedding import pool_hf_batch_output, pool_hf_feature_output


def _norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def test_pool_1d_vector_normalizes_to_unit_length():
    raw = [3.0, 4.0]
    out = pool_hf_feature_output(raw)
    assert len(out) == 2
    assert abs(_norm(out) - 1.0) < 1e-9
    assert abs(out[0] - 0.6) < 1e-9
    assert abs(out[1] - 0.8) < 1e-9


def test_pool_2d_token_matrix_mean_pools_then_normalizes():
    # two tokens, dim 2: mean = [1.5, 3.5] → normalize
    raw = [[1.0, 2.0], [2.0, 5.0]]
    out = pool_hf_feature_output(raw)
    assert len(out) == 2
    assert abs(_norm(out) - 1.0) < 1e-9
    mean = [1.5, 3.5]
    n = _norm(mean)
    assert abs(out[0] - mean[0] / n) < 1e-9
    assert abs(out[1] - mean[1] / n) < 1e-9


def test_pool_wrapped_single_vector():
    raw = [[3.0, 4.0]]
    out = pool_hf_feature_output(raw)
    assert abs(out[0] - 0.6) < 1e-9


def test_pool_batch_of_pooled_vectors():
    raw = [[3.0, 4.0], [0.0, 5.0]]
    out = pool_hf_batch_output(raw, expected=2)
    assert len(out) == 2
    assert abs(_norm(out[0]) - 1.0) < 1e-9
    assert abs(_norm(out[1]) - 1.0) < 1e-9
    assert abs(out[1][1] - 1.0) < 1e-9


def test_pool_batch_of_token_matrices():
    raw = [
        [[1.0, 0.0], [1.0, 0.0]],
        [[0.0, 2.0], [0.0, 2.0]],
    ]
    out = pool_hf_batch_output(raw, expected=2)
    assert len(out) == 2
    assert abs(out[0][0] - 1.0) < 1e-9
    assert abs(out[1][1] - 1.0) < 1e-9


def test_pool_rejects_empty():
    with pytest.raises(ValueError):
        pool_hf_feature_output([])
