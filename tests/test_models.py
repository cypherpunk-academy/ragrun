"""Tests for core ragrun domain models."""
from datetime import datetime

import pytest

from app.shared.models import CHUNK_TYPE_ENUM, ChunkMetadata, ChunkRecord


@pytest.fixture
def base_metadata_dict():
    return {
        "author": "Immanuel Hermann Fichte",
        "source_id": "philo-book1",
        "source_title": "System der Ethik",
        "source_index": 5,
        "segment_id": "chapter-03",
        "segment_title": "Zur Methodik",
        "segment_index": 2,
        "parent_id": None,
        "chunk_id": "philo-book1-chunk-0005",
        "chunk_type": "book",
        "worldviews": ["Idealismus"],
        "importance": 5,
        "text": "Example chunk text",
        "content_hash": "sha256:abc",
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "source_type": "book",
        "language": "de",
        "tags": ["philosophy", "typology"],
    }


def test_chunk_metadata_valid(base_metadata_dict):
    metadata = ChunkMetadata.from_payload(base_metadata_dict)
    assert metadata.chunk_type == "book"
    assert metadata.chunk_id == base_metadata_dict["chunk_id"]


@pytest.mark.parametrize("missing_field", ["source_id", "chunk_id", "content_hash", "language"])
def test_chunk_metadata_required_fields(missing_field, base_metadata_dict):
    payload = base_metadata_dict.copy()
    payload.pop(missing_field)
    with pytest.raises(Exception):
        ChunkMetadata.from_payload(payload)


@pytest.mark.parametrize("invalid_chunk_type", ["invalid", "Book", "chapter"])
def test_chunk_metadata_chunk_type_enum(invalid_chunk_type, base_metadata_dict):
    payload = base_metadata_dict.copy()
    payload["chunk_type"] = invalid_chunk_type
    with pytest.raises(ValueError):
        ChunkMetadata.from_payload(payload)


def test_chunk_record_infers_id_from_metadata(base_metadata_dict):
    chunk_dict = {
        "id": None,
        "text": "Hello",
        "metadata": base_metadata_dict,
        "embedding": [0.1, 0.2],
    }
    record = ChunkRecord.from_dict(chunk_dict)
    assert record.id == base_metadata_dict["chunk_id"]


def test_chunk_record_requires_metadata(base_metadata_dict):
    chunk_dict = {
        "id": "chunk-1",
        "text": "Hello",
    }
    with pytest.raises(ValueError):
        ChunkRecord.from_dict(chunk_dict)
