"""Tests for the App-Chat streaming adapter (Welle 3a)."""
from __future__ import annotations

import json
from typing import Any

import pytest

from app.services.app_chat_stream_service import (
    _history_messages_from_turns,
    _strip_usage_comment,
    _thinking_type_for_mode,
    run_app_chat_once,
    stream_app_chat,
)


class _FakeChunk:
    def __init__(self, content: str = "", additional_kwargs: dict | None = None, usage_metadata: dict | None = None) -> None:
        self.content = content
        self.additional_kwargs = additional_kwargs or {}
        self.usage_metadata = usage_metadata


class _FakeGraph:
    def __init__(self, events: list[dict[str, Any]]) -> None:
        self._events = events

    async def astream_events(self, initial_state: dict, config: dict, version: str = "v2"):
        for event in self._events:
            yield event


class _FakeTalks:
    def __init__(self, history: list[dict[str, Any]] | None = None) -> None:
        self.history = history or []
        self.created: list[dict[str, Any]] = []
        self.saved_refs: list[tuple[str, list[dict[str, Any]]]] = []

    async def load_talk_turns(self, talk_id: str):
        return self.history

    async def create_talk_turn(self, **kwargs: Any) -> dict[str, str]:
        self.created.append(kwargs)
        return {"talk_id": "talk-1", "turn_id": "turn-1"}

    async def save_talk_summary(self, talk_id: str, summary: str) -> None:
        raise AssertionError("not used by this test")

    async def save_turn_references(self, turn_id: str, references) -> None:
        self.saved_refs.append((turn_id, list(references)))


class _FakeUsageRecorder:
    def __init__(self) -> None:
        self.recorded: list[dict[str, Any]] = []

    async def record(self, **kwargs: Any) -> None:
        self.recorded.append(kwargs)


def _happy_path_events() -> list[dict[str, Any]]:
    return [
        {
            "event": "on_custom_event",
            "name": "ace_progress",
            "data": {"step": "retrieve", "label": "Suche Quellen…"},
        },
        {"event": "on_chat_model_stream", "data": {"chunk": _FakeChunk(content="Hallo")}},
        {"event": "on_chat_model_stream", "data": {"chunk": _FakeChunk(content=" Welt")}},
        {
            "event": "on_chain_end",
            "name": "finalize",
            "data": {
                "output": {
                    "final_response": "Hallo Welt",
                    "citations": [
                        {"chunk_id": "c1", "source_title": "Buch", "segment_title": "Kap 1"}
                    ],
                    "confidence_score": 0.8,
                    "intent": "retrieval",
                    "sufficiency": "sufficient",
                    "usage_metadata": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                }
            },
        },
    ]


def test_thinking_type_for_mode() -> None:
    assert _thinking_type_for_mode(None) == "disabled"
    assert _thinking_type_for_mode("chat") == "disabled"
    assert _thinking_type_for_mode("nachdenken") == "enabled"


def test_strip_usage_comment_removes_trailing_marker() -> None:
    raw = 'Die Antwort.\n<!-- usage {"prompt_tokens": 10, "model": "deepseek-v4-flash"} -->'
    assert _strip_usage_comment(raw) == "Die Antwort."


def test_strip_usage_comment_noop_without_marker() -> None:
    assert _strip_usage_comment("Die Antwort.") == "Die Antwort."


def test_history_messages_from_turns_alternates_and_caps() -> None:
    rows = [
        {"user_message": f"U{i}", "assistant_message": f"A{i}"} for i in range(10)
    ]
    messages = _history_messages_from_turns(rows, max_turns=3)
    assert len(messages) == 6
    assert messages[0].content == "U7"
    assert messages[-1].content == "A9"


@pytest.mark.asyncio
async def test_stream_app_chat_emits_events_and_persists_done() -> None:
    graph = _FakeGraph(_happy_path_events())
    talks = _FakeTalks()
    usage_recorder = _FakeUsageRecorder()

    collected = []
    async for sse in stream_app_chat(
        graph,
        talks,
        user_id="u1",
        user_name="User",
        message="Hi",
        personality="philo",
        talk_id=None,
        mode="chat",
        model=None,
        usage_recorder=usage_recorder,
    ):
        collected.append(json.loads(sse["data"]))

    assert [c["type"] for c in collected] == ["status", "token", "token", "done"]

    done = collected[-1]
    assert done["turn_id"] == "turn-1"
    assert done["talk_id"] == "talk-1"
    assert done["assistant_message"] == "Hallo Welt"
    assert done["citations"] == [{"chunk_id": "c1", "source_title": "Buch", "segment_title": "Kap 1"}]
    assert done["usage"] == {
        "model": talks.created[0]["usage"]["model"],
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }
    assert done["tool_results"] == []

    assert talks.created[0]["assistant_message"] == "Hallo Welt"
    assert talks.created[0]["talk_id"] is None
    assert talks.saved_refs == [
        ("turn-1", [{"chunk_id": "c1", "source_title": "Buch", "segment_title": "Kap 1"}])
    ]

    assert len(usage_recorder.recorded) == 1
    recorded = usage_recorder.recorded[0]
    assert recorded["turn_id"] == "turn-1"
    assert recorded["talk_id"] == "talk-1"
    assert recorded["endpoint"] == "app_chat"
    assert recorded["prompt_tokens"] == 10
    assert recorded["completion_tokens"] == 5
    assert recorded["total_tokens"] == 15


@pytest.mark.asyncio
async def test_stream_app_chat_rejects_empty_message() -> None:
    graph = _FakeGraph([])
    talks = _FakeTalks()

    with pytest.raises(ValueError):
        async for _ in stream_app_chat(
            graph, talks, user_id="u1", user_name="User", message="   ", personality="philo"
        ):
            pass


@pytest.mark.asyncio
async def test_stream_app_chat_yields_error_event_on_graph_failure() -> None:
    class _BrokenGraph:
        async def astream_events(self, *args: Any, **kwargs: Any):
            raise RuntimeError("boom")
            yield  # pragma: no cover - makes this an async generator

    talks = _FakeTalks()
    collected = []
    async for sse in stream_app_chat(
        _BrokenGraph(), talks, user_id="u1", user_name="User", message="Hi", personality="philo"
    ):
        collected.append(json.loads(sse["data"]))

    assert collected == [{"type": "error", "message": "Interner Fehler"}]
    assert talks.created == []


@pytest.mark.asyncio
async def test_run_app_chat_once_returns_final_turn() -> None:
    graph = _FakeGraph(_happy_path_events())
    talks = _FakeTalks()

    result = await run_app_chat_once(
        graph,
        talks,
        user_id="u1",
        user_name="User",
        message="Hi",
        personality="philo",
        usage_recorder=_FakeUsageRecorder(),
    )

    assert result == {"talk_id": "talk-1", "turn_id": "turn-1", "reply": "Hallo Welt"}
