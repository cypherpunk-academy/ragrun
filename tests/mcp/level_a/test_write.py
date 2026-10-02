"""Ebene A, schreibende Tools und Nutzertrennung."""
from __future__ import annotations

import uuid

import pytest

from tests.mcp.mcp_live_support import db_execute, db_fetchone, has_stacktrace


def _hits(body):
    if isinstance(body, dict) and "error" in body:
        raise AssertionError(f"{body.get('code')}: {body.get('error')}")
    if not isinstance(body, list):
        raise AssertionError(f"erwartete Liste, bekam {type(body).__name__}")
    return body


def _created_id(body: dict) -> str:
    note_id = body.get("id") or body.get("note_id")
    assert body.get("ok") is True and note_id, body.get("error") or body
    return str(note_id)


def _open_enum(stored: bool, error_body: dict | None) -> str:
    if stored:
        return "gespeichert"
    if error_body and error_body.get("error") and not has_stacktrace(error_body):
        return "abgelehnt: " + str(error_body.get("error"))[:180]
    raise AssertionError(error_body or "weder gespeichert noch strukturiert abgelehnt")


def test_A_01(live):
    """A-01: insight auf dem leeren Segment legt das Protokoll an."""
    created = live.call(
        "a",
        "append_to_protocol",
        {
            "source_id": live.source_id,
            "segment_slug": live.segment_empty,
            "entry_type": "insight",
            "content": "[TEST] A-01 Erkenntnis",
        },
    )
    assert created.get("ok") is True
    body = live.call(
        "a",
        "get_protocol",
        {"source_id": live.source_id, "segment_slug": live.segment_empty},
    )
    contents = [entry.get("content") for entry in body.get("entries") or []]
    assert any(str(content).startswith("[TEST] A-01") for content in contents)
    assert any(entry.get("entry_type") == "insight" for entry in body["entries"])


def test_A_02(live):
    """A-02: note, question und summary werden mit diesem Typ gespeichert."""
    for entry_type in ("note", "question", "summary"):
        created = live.call(
            "a",
            "append_to_protocol",
            {
                "source_id": live.source_id,
                "segment_slug": live.segment,
                "entry_type": entry_type,
                "content": f"[TEST] A-02 {entry_type}",
            },
        )
        assert created.get("ok") is True
    body = live.call("a", "get_protocol", {"source_id": live.source_id, "segment_slug": live.segment})
    found = {entry["entry_type"] for entry in body.get("entries") or [] if str(entry.get("content") or "").startswith("[TEST] A-02")}
    assert found == {"note", "question", "summary"}


def test_A_03(live, remark):
    """A-03: entry_type foo. Plan offen — gespeichert oder abgelehnt, ohne Stacktrace."""
    created = live.call(
        "a",
        "append_to_protocol",
        {
            "source_id": live.source_id,
            "segment_slug": live.segment,
            "entry_type": "foo",
            "content": "[TEST] A-03 foo",
        },
    )
    stored = False
    if created.get("ok") is True:
        body = live.call("a", "get_protocol", {"source_id": live.source_id, "segment_slug": live.segment})
        stored = any(
            entry.get("entry_type") == "foo" and str(entry.get("content") or "").startswith("[TEST] A-03")
            for entry in body.get("entries") or []
        )
    remark(_open_enum(stored, None if stored else created))


def test_A_04(live):
    """A-04: Eintrag mit paragraph_id bleibt mit dem Absatz verknüpft."""
    created = live.call(
        "a",
        "append_to_protocol",
        {
            "source_id": live.source_id,
            "segment_slug": live.segment,
            "entry_type": "note",
            "content": "[TEST] A-04 Absatz",
            "paragraph_id": live.para_id,
        },
    )
    assert created.get("ok") is True
    body = live.call("a", "get_protocol", {"source_id": live.source_id, "segment_slug": live.segment})
    ours = [entry for entry in body.get("entries") or [] if str(entry.get("content") or "").startswith("[TEST] A-04")]
    assert ours and ours[0].get("paragraph_id") == live.para_id


def test_A_05(live):
    """A-05: conversation_url wird am Eintrag gespeichert."""
    url = "https://dev.example/philo/test-conversation"
    created = live.call(
        "a",
        "append_to_protocol",
        {
            "source_id": live.source_id,
            "segment_slug": live.segment,
            "entry_type": "note",
            "content": "[TEST] A-05 URL",
            "conversation_url": url,
        },
    )
    assert created.get("ok") is True and created.get("entry_id")
    row = db_fetchone(
        "SELECT conversation_url FROM protocol_entries WHERE id = CAST(:id AS uuid)",
        {"id": created["entry_id"]},
    )
    assert row is not None and row[0] == url


def test_A_06(live):
    """A-06: zwei aufeinanderfolgende Einträge bleiben beide erhalten."""
    for label in ("eins", "zwei"):
        created = live.call(
            "a",
            "append_to_protocol",
            {
                "source_id": live.source_id,
                "segment_slug": live.segment,
                "entry_type": "note",
                "content": f"[TEST] A-06 {label}",
            },
        )
        assert created.get("ok") is True
    body = live.call("a", "get_protocol", {"source_id": live.source_id, "segment_slug": live.segment})
    contents = [str(entry.get("content") or "") for entry in body.get("entries") or []]
    assert any(item.startswith("[TEST] A-06 eins") for item in contents)
    assert any(item.startswith("[TEST] A-06 zwei") for item in contents)


def test_A_07(live):
    """A-07: leerer content liefert eine Fehlermeldung."""
    body = live.call(
        "a",
        "append_to_protocol",
        {
            "source_id": live.source_id,
            "segment_slug": live.segment,
            "entry_type": "note",
            "content": "",
        },
    )
    if body.get("ok") is True and body.get("entry_id"):
        db_execute("DELETE FROM protocol_entries WHERE id = CAST(:id AS uuid)", {"id": body["entry_id"]})
    assert body.get("error"), body
    assert not has_stacktrace(body)


def test_A_08(live):
    """A-08: ungültiges Segment liefert einen Fehler und kein verwaistes Protokoll."""
    source_id = str(uuid.uuid4())
    body = live.call(
        "a",
        "append_to_protocol",
        {
            "source_id": source_id,
            "segment_slug": "kein-kapitel",
            "entry_type": "note",
            "content": "[TEST] A-08",
        },
    )
    assert body.get("error")
    assert not has_stacktrace(body)
    row = db_fetchone(
        """
        SELECT count(*) FROM protocols
        WHERE user_id = CAST(:uid AS uuid) AND source_id = :sid
        """,
        {"uid": live.user_a, "sid": source_id},
    )
    assert row is not None and row[0] == 0


def test_L_01(live):
    """L-01: list_work_texts ohne Parameter liefert bis zu 20 eigene Texte."""
    rows = _hits(live.call("a", "list_work_texts"))
    assert len(rows) <= 20
    ids = {row["note_id"] for row in rows}
    assert live.note_b not in ids
    for row in rows:
        assert row.get("display_title") and row.get("title_source")


def test_L_02(live):
    """L-02: text_type essay liefert nur Essays, einschließlich des Fixture-Essays."""
    rows = _hits(live.call("a", "list_work_texts", {"text_type": "essay", "limit": 50}))
    assert rows
    assert all(row.get("text_type") == "essay" for row in rows)
    assert any(row["note_id"] == live.essay_a for row in rows)


def test_L_03(live, remark):
    """L-03: limit 1 und 50 begrenzen, 0 und 51 werden still auf 1 bzw. 50 gezogen."""
    created = live.call(
        "a",
        "create_work_text",
        {"title": "[TEST] L-03", "content": "[TEST] Limit"},
    )
    assert created.get("ok") is True
    one = _hits(live.call("a", "list_work_texts", {"limit": 1}))
    fifty = _hits(live.call("a", "list_work_texts", {"limit": 50}))
    zero = _hits(live.call("a", "list_work_texts", {"limit": 0}))
    over = _hits(live.call("a", "list_work_texts", {"limit": 51}))
    assert len(one) == 1
    assert len(zero) == 1
    assert len(fifty) <= 50
    assert len(over) <= 50
    assert len(over) == len(fifty)
    remark(f"1→{len(one)} 50→{len(fifty)} 0→{len(zero)} 51→{len(over)}")


def test_L_04(live):
    """L-04: Nutzer ohne Texte erhält eine leere Liste."""
    own = _hits(live.call("a", "list_work_texts", {"limit": 1}))
    other = _hits(live.call("b", "list_work_texts", {"limit": 1}))
    if not own:
        assert own == []
        return
    if not other:
        assert other == []
        return
    pytest.skip("Kein Konto ohne Arbeitstexte in den Fixtures (User A und User B haben Texte).")


def test_L_05(live):
    """L-05: User A listet keinen Arbeitstext von User B."""
    rows = _hits(live.call("a", "list_work_texts", {"limit": 50}))
    assert all(row["note_id"] != live.note_b for row in rows)


def test_T_01(live):
    """T-01: User A liest NOTE_A mit title, content, version, display_title und title_source."""
    body = live.call("a", "get_work_text", {"note_id": live.note_a})
    assert body.get("error") is None
    assert body.get("title") is not None
    assert "content" in body and "version" in body
    assert body.get("display_title") and body.get("title_source")


def test_T_02(live):
    """T-02: unbekannte note_id liefert note not found."""
    body = live.call("a", "get_work_text", {"note_id": "test-missing-" + uuid.uuid4().hex})
    assert body.get("error") == "note not found"


def test_T_03(live):
    """T-03: User A liest NOTE_B nicht und erhält keine Inhalte."""
    own = live.call("b", "get_work_text", {"note_id": live.note_b})
    foreign = live.call("a", "get_work_text", {"note_id": live.note_b})
    assert foreign.get("error") == "note not found"
    assert own.get("content"), own.get("error") or "NOTE_B ohne Inhalt"
    assert own["content"] not in str(foreign)


def test_C_01(live):
    """C-01: title und content legen eine Notiz mit text_type note und Version 1 an."""
    created = live.call(
        "a",
        "create_work_text",
        {"title": "[TEST] C-01", "content": "[TEST] Inhalt C-01"},
    )
    note_id = _created_id(created)
    assert created.get("version") == 1
    body = live.call("a", "get_work_text", {"note_id": note_id})
    assert body.get("text_type") == "note"
    assert body.get("version") == 1


def test_C_02(live):
    """C-02: text_type essay und draft werden übernommen."""
    for text_type in ("essay", "draft"):
        created = live.call(
            "a",
            "create_work_text",
            {"title": f"[TEST] C-02 {text_type}", "content": f"[TEST] {text_type}", "text_type": text_type},
        )
        body = live.call("a", "get_work_text", {"note_id": _created_id(created)})
        assert body.get("text_type") == text_type


def test_C_03(live, remark):
    """C-03: text_type foo. Plan offen — gespeichert oder abgelehnt, ohne Stacktrace."""
    created = live.call(
        "a",
        "create_work_text",
        {"title": "[TEST] C-03", "content": "[TEST] foo-typ", "text_type": "foo"},
    )
    stored = False
    if created.get("ok") is True:
        body = live.call("a", "get_work_text", {"note_id": _created_id(created)})
        stored = body.get("text_type") == "foo"
    remark(_open_enum(stored, None if stored else created))


def test_C_05(live):
    """C-05: conversation_url wird am Arbeitstext gespeichert."""
    url = "https://dev.example/philo/note-conversation"
    created = live.call(
        "a",
        "create_work_text",
        {
            "title": "[TEST] C-05",
            "content": "[TEST] mit URL",
            "conversation_url": url,
        },
    )
    body = live.call("a", "get_work_text", {"note_id": _created_id(created)})
    assert body.get("conversation_url") == url


def test_C_06(live):
    """C-06: leerer title oder leerer content liefert eine Fehlermeldung."""
    empty_title = live.call(
        "a",
        "create_work_text",
        {"title": "", "content": "[TEST] C-06 ohne Titel"},
    )
    empty_content = live.call(
        "a",
        "create_work_text",
        {"title": "[TEST] C-06 ohne Inhalt", "content": ""},
    )
    assert empty_title.get("error"), empty_title
    assert empty_content.get("error"), empty_content
    assert not has_stacktrace(empty_title)
    assert not has_stacktrace(empty_content)


@pytest.mark.xfail(strict=True, reason="Befund F-4")
def test_C_07(live):
    """C-07: content über 1 MB UTF-8 liefert content_too_long."""
    content = "[TEST] " + ("x" * 1_048_576)
    body = live.call(
        "a",
        "create_work_text",
        {"title": "[TEST] C-07 zu lang", "content": content},
        timeout=120,
    )
    assert body.get("code") == "content_too_long"
    assert not has_stacktrace(body)


def test_C_08(live):
    """C-08: ein neuer Text erscheint in list_work_texts."""
    created = live.call(
        "a",
        "create_work_text",
        {"title": "[TEST] C-08", "content": "[TEST] sichtbar"},
    )
    note_id = _created_id(created)
    rows = _hits(live.call("a", "list_work_texts", {"limit": 50}))
    assert any(row["note_id"] == note_id for row in rows)


def test_U_01(live):
    """U-01: update mit der gelesenen Version liefert ok und new_version = alt + 1."""
    created = live.call(
        "a",
        "create_work_text",
        {"title": "[TEST] U-01", "content": "[TEST] vorher"},
    )
    note_id = _created_id(created)
    current = live.call("a", "get_work_text", {"note_id": note_id})
    updated = live.call(
        "a",
        "update_work_text",
        {"note_id": note_id, "content": "[TEST] nachher", "expected_version": current["version"]},
    )
    assert updated.get("ok") is True
    assert updated.get("new_version") == current["version"] + 1


def test_U_02(live):
    """U-02: veraltete Version liefert conflict, der Inhalt bleibt."""
    created = live.call(
        "a",
        "create_work_text",
        {"title": "[TEST] U-02", "content": "[TEST] v1"},
    )
    note_id = _created_id(created)
    current = live.call("a", "get_work_text", {"note_id": note_id})
    live.call(
        "a",
        "update_work_text",
        {"note_id": note_id, "content": "[TEST] v2", "expected_version": current["version"]},
    )
    stale = live.call(
        "a",
        "update_work_text",
        {"note_id": note_id, "content": "[TEST] veraltet", "expected_version": current["version"]},
    )
    assert stale.get("error") == "conflict"
    again = live.call("a", "get_work_text", {"note_id": note_id})
    assert again["content"] == "[TEST] v2"
    assert again["version"] == current["version"] + 1


def test_U_03(live):
    """U-03: status wechselt von final nach draft."""
    created = live.call(
        "a",
        "create_work_text",
        {"title": "[TEST] U-03", "content": "[TEST] status"},
    )
    note_id = _created_id(created)
    current = live.call("a", "get_work_text", {"note_id": note_id})
    final = live.call(
        "a",
        "update_work_text",
        {
            "note_id": note_id,
            "content": "[TEST] status",
            "expected_version": current["version"],
            "status": "final",
        },
    )
    assert final.get("ok") is True
    assert live.call("a", "get_work_text", {"note_id": note_id})["status"] == "final"
    draft = live.call(
        "a",
        "update_work_text",
        {
            "note_id": note_id,
            "content": "[TEST] status",
            "expected_version": final["new_version"],
            "status": "draft",
        },
    )
    assert draft.get("ok") is True
    assert live.call("a", "get_work_text", {"note_id": note_id})["status"] == "draft"


def test_U_04(live, remark):
    """U-04: status foo. Plan offen — gespeichert oder abgelehnt, ohne Stacktrace."""
    created = live.call(
        "a",
        "create_work_text",
        {"title": "[TEST] U-04", "content": "[TEST] status foo"},
    )
    note_id = _created_id(created)
    current = live.call("a", "get_work_text", {"note_id": note_id})
    updated = live.call(
        "a",
        "update_work_text",
        {
            "note_id": note_id,
            "content": "[TEST] status foo",
            "expected_version": current["version"],
            "status": "foo",
        },
    )
    stored = False
    if updated.get("ok") is True:
        stored = live.call("a", "get_work_text", {"note_id": note_id}).get("status") == "foo"
    remark(_open_enum(stored, None if stored else updated))


def test_U_05(live):
    """U-05: update ohne title lässt den Titel stehen."""
    created = live.call(
        "a",
        "create_work_text",
        {"title": "[TEST] U-05 Titel", "content": "[TEST] alt"},
    )
    note_id = _created_id(created)
    current = live.call("a", "get_work_text", {"note_id": note_id})
    updated = live.call(
        "a",
        "update_work_text",
        {"note_id": note_id, "content": "[TEST] neu", "expected_version": current["version"]},
    )
    assert updated.get("ok") is True
    assert live.call("a", "get_work_text", {"note_id": note_id})["title"] == "[TEST] U-05 Titel"


def test_U_06(live):
    """U-06: update mit title ändert den Titel."""
    created = live.call(
        "a",
        "create_work_text",
        {"title": "[TEST] U-06 alt", "content": "[TEST] text"},
    )
    note_id = _created_id(created)
    current = live.call("a", "get_work_text", {"note_id": note_id})
    updated = live.call(
        "a",
        "update_work_text",
        {
            "note_id": note_id,
            "content": "[TEST] text",
            "expected_version": current["version"],
            "title": "[TEST] U-06 neu",
        },
    )
    assert updated.get("ok") is True
    assert live.call("a", "get_work_text", {"note_id": note_id})["title"] == "[TEST] U-06 neu"


def test_U_07(live):
    """U-07: User A kann NOTE_B nicht ändern."""
    before = live.call("b", "get_work_text", {"note_id": live.note_b})
    denied = live.call(
        "a",
        "update_work_text",
        {
            "note_id": live.note_b,
            "content": "fremder Eingriff",
            "expected_version": before["version"],
        },
    )
    if denied.get("ok") is True:
        live.call(
            "b",
            "update_work_text",
            {
                "note_id": live.note_b,
                "content": before["content"],
                "expected_version": denied["new_version"],
                "title": before.get("title"),
            },
        )
    after = live.call("b", "get_work_text", {"note_id": live.note_b})
    assert denied.get("error")
    assert after["content"] == before["content"]
    assert after["version"] == before["version"]


def test_U_08(live):
    """U-08: zwei Clients mit derselben Version — genau einer gewinnt, der andere conflict."""
    created = live.call(
        "a",
        "create_work_text",
        {"title": "[TEST] U-08", "content": "[TEST] ausgang"},
    )
    note_id = _created_id(created)
    current = live.call("a", "get_work_text", {"note_id": note_id})
    left, right = live.hub.race_updates(
        note_id,
        current["version"],
        "[TEST] U-08 a",
        "[TEST] U-08 b",
    )
    winners = [body for body in (left, right) if body.get("ok") is True]
    losers = [body for body in (left, right) if body.get("error") == "conflict"]
    assert len(winners) == 1 and len(losers) == 1
    after = live.call("a", "get_work_text", {"note_id": note_id})
    assert after["version"] == current["version"] + 1
    assert after["content"] in {"[TEST] U-08 a", "[TEST] U-08 b"}
