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


def test_chunk_metadata_lecture_quote_fields(base_metadata_dict):
    payload = base_metadata_dict.copy()
    payload["chunk_type"] = "quote"
    payload["lecture_id"] = "19190525"
    payload["paragraph"] = 7
    payload["body_source_id"] = "a1b2c3d4-e5f6-4789-a012-3456789abcde"
    payload["paragraph_id"] = "para-uuid-1"
    payload["quote_span"] = {"start": 10, "end": 42}
    payload["quote_verified"] = True
    metadata = ChunkMetadata.from_payload(payload)
    dumped = metadata.model_dump(mode="json")
    assert dumped["lecture_id"] == "19190525"
    assert dumped["paragraph"] == 7
    assert dumped["body_source_id"] == "a1b2c3d4-e5f6-4789-a012-3456789abcde"
    assert dumped["paragraph_id"] == "para-uuid-1"
    assert dumped["quote_span"] == {"start": 10, "end": 42}
    assert dumped["quote_verified"] is True


def test_chunk_metadata_catalog_quotes_enum(base_metadata_dict):
    payload = base_metadata_dict.copy()
    payload["quotes"] = "quoted"
    payload["quote_marks"] = [
        {
            "id": "q1",
            "author": "Goethe",
            "paragraph_id": "p1",
            "span": {"start": 0, "end": 5},
        }
    ]
    metadata = ChunkMetadata.from_payload(payload)
    assert metadata.quotes == "quoted"
    assert metadata.quote_marks is not None
    assert len(metadata.quote_marks) == 1


def test_chunk_metadata_migrates_legacy_quotes_array(base_metadata_dict):
    payload = base_metadata_dict.copy()
    payload["author"] = "Rudolf Steiner"
    payload["quotes"] = [
        {
            "id": "q1",
            "author": "Rudolf Steiner",
            "paragraph_id": "p1",
            "span": {"start": 0, "end": 5},
        }
    ]
    metadata = ChunkMetadata.from_payload(payload)
    assert metadata.quotes == "author"
    assert metadata.quote_marks is not None
    assert metadata.quote_marks[0]["id"] == "q1"


@pytest.fixture
def typology_member_payload(base_metadata_dict):
    payload = base_metadata_dict.copy()
    payload.update(
        {
            "chunk_type": "typology",
            "typology_slug": "soziale-gesundheit",
            "typology_role": "member",
            "group_id": "feld-1",
            "group_title": "Geist -> Recht",
            "health": "gesund",
            "counterpart_chunk_id": "counterpart-uuid",
            "segment_index": 0,
        }
    )
    return payload


def test_chunk_metadata_typology_member_fields(typology_member_payload):
    dumped = ChunkMetadata.from_payload(typology_member_payload).model_dump(mode="json")
    assert dumped["typology_slug"] == "soziale-gesundheit"
    assert dumped["typology_role"] == "member"
    assert dumped["group_id"] == "feld-1"
    assert dumped["group_title"] == "Geist -> Recht"
    assert dumped["health"] == "gesund"
    assert dumped["counterpart_chunk_id"] == "counterpart-uuid"
    assert "short_text" not in dumped


def test_chunk_metadata_typology_without_health(typology_member_payload):
    payload = typology_member_payload.copy()
    for key in ("health", "counterpart_chunk_id", "group_id", "group_title"):
        payload.pop(key)
    metadata = ChunkMetadata.from_payload(payload)
    assert metadata.health is None
    assert metadata.counterpart_chunk_id is None


def test_chunk_metadata_typology_empty_fields_are_absent(typology_member_payload):
    payload = typology_member_payload.copy()
    payload["health"] = ""
    payload["counterpart_chunk_id"] = "  "
    metadata = ChunkMetadata.from_payload(payload)
    assert metadata.health is None
    assert metadata.counterpart_chunk_id is None


@pytest.mark.parametrize(
    "field,value",
    [("typology_role", "leaf"), ("health", "neutral"), ("health", "Gesund")],
)
def test_chunk_metadata_typology_rejects_unknown_values(typology_member_payload, field, value):
    payload = typology_member_payload.copy()
    payload[field] = value
    with pytest.raises(ValueError):
        ChunkMetadata.from_payload(payload)


def test_chunk_metadata_kurzfassung_omitted_when_empty(base_metadata_dict):
    dumped = ChunkMetadata.from_payload(base_metadata_dict).model_dump(mode="json")
    assert "kurzfassung" not in dumped
    payload = base_metadata_dict.copy()
    payload["kurzfassung"] = "  Der Gedanke wird am anderen wach.  "
    kept = ChunkMetadata.from_payload(payload).model_dump(mode="json")
    assert kept["kurzfassung"] == "Der Gedanke wird am anderen wach."
