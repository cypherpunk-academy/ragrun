"""Unit-Tests für _build_retrieval_query (Teil A — Retrieval-Query-Kontext)."""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from app.retrieval.graphs.assistant_chat_graph import _build_retrieval_query


def _state(
    user_message: str = "Was meint Steiner damit?",
    context_paragraph_text: str = "",
    messages: list = [],
) -> dict:
    return {
        "user_message": user_message,
        "context_paragraph_text": context_paragraph_text,
        "messages": messages,
    }


def test_only_user_message():
    """Ohne Bezugstext und ohne AI-Turn: nur die User-Message."""
    result = _build_retrieval_query(_state(user_message="Hallo"))
    assert result == "Hallo"


def test_paragraph_body_appended():
    """Bezugstext wird nach dem Header-Strip angehängt."""
    para = "Absatz 3 · Kapitel I\n\nDer Mensch ist ein dreigliedriges Wesen."
    result = _build_retrieval_query(_state(context_paragraph_text=para))
    assert "Bezugstext:" in result
    assert "dreigliedriges Wesen" in result
    # Header darf nicht im Ergebnis stehen
    assert "Absatz 3 · Kapitel I" not in result


def test_last_ai_message_appended():
    """Die letzte AIMessage wird als 'Vorherige Antwort' angehängt."""
    messages = [
        HumanMessage(content="Was ist der Geist?"),
        AIMessage(content="Der Geist ist das Höchste im Menschen."),
    ]
    result = _build_retrieval_query(_state(messages=messages))
    assert "Vorherige Antwort:" in result
    assert "Höchste im Menschen" in result


def test_both_blocks_present():
    """Bezugstext und letzte AI-Antwort werden beide angehängt."""
    para = "Absatz 1\n\nDie Seele strebt nach oben."
    messages = [AIMessage(content="Steiner beschreibt die Seele als Mittler.")]
    result = _build_retrieval_query(_state(context_paragraph_text=para, messages=messages))
    assert "Bezugstext:" in result
    assert "Vorherige Antwort:" in result
    assert "Seele strebt" in result
    assert "Mittler" in result


def test_paragraph_without_header():
    """Bezugstext ohne Header-Zeile wird vollständig übernommen."""
    para = "Reiner Absatztext ohne Header."
    result = _build_retrieval_query(_state(context_paragraph_text=para))
    assert "Reiner Absatztext" in result


def test_only_last_ai_no_paragraph():
    """Nur AI-Turn, kein Bezugstext: kein 'Bezugstext:'-Block."""
    messages = [AIMessage(content="Kurze Antwort.")]
    result = _build_retrieval_query(_state(messages=messages))
    assert "Bezugstext:" not in result
    assert "Vorherige Antwort:" in result


def test_truncation_paragraph():
    """Langer Bezugstext wird auf ≤400 Zeichen getrimmt."""
    long_body = "Ein Satz. " * 100  # ~1000 Zeichen
    para = f"Header\n\n{long_body}"
    result = _build_retrieval_query(_state(context_paragraph_text=para))
    # Der Bezugstext-Block allein darf ≤400+len("Bezugstext:\n") Zeichen haben
    bezug_block = [p for p in result.split("\n\n") if p.startswith("Bezugstext:")][0]
    assert len(bezug_block) <= 420  # 400 + len("Bezugstext:\n")


def test_truncation_ai():
    """Lange AI-Antwort wird auf ≤400 Zeichen getrimmt."""
    long_ai = "Steiner schreibt. " * 100
    messages = [AIMessage(content=long_ai)]
    result = _build_retrieval_query(_state(messages=messages))
    ai_block = [p for p in result.split("\n\n") if p.startswith("Vorherige Antwort:")][0]
    assert len(ai_block) <= 420


def test_non_ai_messages_ignored():
    """Human-Messages in messages werden nicht als AI-Turn gewertet."""
    messages = [HumanMessage(content="Eine Frage."), HumanMessage(content="Noch eine.")]
    result = _build_retrieval_query(_state(messages=messages))
    assert "Vorherige Antwort:" not in result


def test_only_user_message_returned_when_empty_para_and_no_ai():
    """Leerer Bezugstext und keine AI: Ergebnis == user_message."""
    msg = "Frage ohne Kontext"
    result = _build_retrieval_query(_state(user_message=msg, context_paragraph_text="   "))
    assert result == msg
