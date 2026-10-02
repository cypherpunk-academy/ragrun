"""Ebene A, lesende Tools und übergreifende Prüfungen."""
from __future__ import annotations

import uuid

import pytest

from tests.mcp.mcp_live_support import (
    CONCEPT_CHUNK_TYPES,
    QUOTE_CHUNK_TYPES,
    TEXT_CHUNK_TYPES,
    citation_problem,
    find_mcp_error,
    has_stacktrace,
    http_initialize_status,
)

QUERY = "Freiheit und Erkenntnis"


def _hits(body):
    if isinstance(body, dict) and "error" in body:
        raise AssertionError(f"{body.get('code')}: {body.get('error')}")
    if not isinstance(body, list):
        raise AssertionError(f"erwartete Liste, bekam {type(body).__name__}")
    return body


def _types(hits) -> set[str]:
    return {str(hit.get("chunk_type")) for hit in hits}


def test_W_01(live):
    """W-01: whoami mit gültigem Token liefert user_id und E-Mail von User A."""
    body = live.call("a", "whoami")
    assert body["user_id"] == live.user_a
    assert body["email"] == live.email_a
    assert body["authenticated"] is True


def test_W_02(server_url: str):
    """W-02: whoami ohne Token endet mit Auth-Fehler, nicht mit einem Tool-Body."""
    from mcp import Client
    from mcp.client.streamable_http import streamable_http_client
    from mcp.shared._httpx_utils import create_mcp_http_client

    assert http_initialize_status(server_url, None) == 401

    async def _open():
        http = create_mcp_http_client(headers={"ngrok-skip-browser-warning": "true"})
        await http.__aenter__()
        try:
            transport = streamable_http_client(server_url, http_client=http, terminate_on_close=False)
            async with Client(transport, mode="legacy"):
                return "connected"

        finally:
            await http.__aexit__(None, None, None)

    import asyncio

    with pytest.raises(BaseException) as caught:
        asyncio.run(_open())
    assert find_mcp_error(caught.value) is not None


def test_W_03(server_url: str):
    """W-03: manipuliertes Token endet mit Auth-Fehler."""
    from mcp import Client
    from mcp.client.streamable_http import streamable_http_client
    from mcp.shared._httpx_utils import create_mcp_http_client

    forged = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIwIn0.not-a-valid-signature"
    assert http_initialize_status(server_url, forged) == 401

    async def _open():
        http = create_mcp_http_client(
            headers={"Authorization": f"Bearer {forged}", "ngrok-skip-browser-warning": "true"}
        )
        await http.__aenter__()
        try:
            transport = streamable_http_client(server_url, http_client=http, terminate_on_close=False)
            async with Client(transport, mode="legacy"):
                return "connected"
        finally:
            await http.__aexit__(None, None, None)

    import asyncio

    with pytest.raises(BaseException) as caught:
        asyncio.run(_open())
    assert find_mcp_error(caught.value) is not None


def test_S_01(live: Live):
    """S-01: query Freiheit und Erkenntnis liefert bis zu 10 Treffer mit chunk_id und Snippet."""
    hits = _hits(live.call("a", "search_corpus", {"query": QUERY}))
    assert 1 <= len(hits) <= 10
    for hit in hits:
        assert hit.get("chunk_id")
        assert "snippet" in hit
        assert "is_primary" not in hit and "ga" not in hit and "zyklus" not in hit
        if hit.get("paragraph_id"):
            problem = citation_problem(hit.get("citation"))
            assert problem is None, problem


def test_S_02(live: Live, remark):
    """S-02: englische Query liefert 0 Treffer (Embeddings matchen deutsch, Stand 1.10)."""
    hits = _hits(live.call("a", "search_corpus", {"query": "freedom and knowledge"}))
    remark(f"{len(hits)} Treffer")
    assert hits == []


def test_S_03(live: Live):
    """S-03: types text liefert nur book, secondary_book oder talk."""
    hits = _hits(live.call("a", "search_corpus", {"query": QUERY, "types": ["text"]}))
    assert hits
    assert _types(hits) <= TEXT_CHUNK_TYPES
    assert "text" not in _types(hits)


def test_S_04(live: Live, remark):
    """S-04: concept, quote und chapter_summary filtern auf die internen chunk_types."""
    concept = _hits(live.call("a", "search_corpus", {"query": "Freiheit", "types": ["concept"]}))
    quote = _hits(live.call("a", "search_corpus", {"query": "Freiheit", "types": ["quote"]}))
    summary = _hits(live.call("a", "search_corpus", {"query": "Freiheit", "types": ["chapter_summary"]}))
    assert concept and _types(concept) <= CONCEPT_CHUNK_TYPES
    assert quote and _types(quote) <= QUOTE_CHUNK_TYPES
    assert summary and _types(summary) <= {"chapter_summary"}
    remark(f"concept={sorted(_types(concept))} quote={sorted(_types(quote))} summary={len(summary)}")


def test_S_05(live: Live, remark):
    """S-05: types concept und quote mischen nur diese internen Typen."""
    allowed = CONCEPT_CHUNK_TYPES | QUOTE_CHUNK_TYPES
    hits = _hits(live.call("a", "search_corpus", {"query": "Freiheit", "types": ["concept", "quote"], "limit": 20}))
    assert hits
    assert _types(hits) <= allowed
    remark("Typen: " + ", ".join(sorted(_types(hits))))


def test_S_06(live: Live):
    """S-06: unbekannter types-Wert liefert invalid_types und die erlaubte Liste."""
    body = live.call("a", "search_corpus", {"query": QUERY, "types": ["bogus_type"]})
    assert isinstance(body, dict)
    assert body.get("code") == "invalid_types"
    assert "allowed" in str(body.get("error"))
    assert not has_stacktrace(body)


def test_S_07(live: Live):
    """S-07: limit 1 liefert genau einen Treffer, limit 20 höchstens 20."""
    one = _hits(live.call("a", "search_corpus", {"query": "Freiheit", "limit": 1}))
    twenty = _hits(live.call("a", "search_corpus", {"query": "Freiheit", "limit": 20}))
    assert len(one) == 1
    assert 1 <= len(twenty) <= 20


def test_S_08(live: Live, remark):
    """S-08: limit 0 und 21 werden still auf 1 bzw. 20 begrenzt."""
    zero = _hits(live.call("a", "search_corpus", {"query": "Freiheit", "limit": 0}))
    broad = _hits(live.call("a", "search_corpus", {"query": "Freiheit", "limit": 20}))
    over = _hits(live.call("a", "search_corpus", {"query": "Freiheit", "limit": 21}))
    assert len(zero) == 1
    assert len(over) <= 20
    assert len(over) == len(broad)
    remark(f"limit 0 → {len(zero)}, limit 21 → {len(over)}")


def test_S_09(live: Live):
    """S-09: leere query liefert empty_query, keinen Serverfehler."""
    body = live.call("a", "search_corpus", {"query": ""})
    assert isinstance(body, dict)
    assert body.get("code") == "empty_query"
    assert not has_stacktrace(body)


def test_S_10(live: Live):
    """S-10: Suchbegriff aus dem Lessig-Band trifft source_id_bg."""
    hits = _hits(live.call("a", "search_corpus", {"query": live.search_bg, "limit": 10}))
    assert any(str(hit.get("source_id") or "").split(":")[0] == live.source_id_bg for hit in hits)


def test_S_11(live: Live):
    """S-11: paragraph_id aus der Suche lässt sich mit get_passage lesen."""
    hits = _hits(live.call("a", "search_corpus", {"query": QUERY}))
    chosen = next((hit for hit in hits if hit.get("paragraph_id")), None)
    assert chosen is not None, "kein Treffer mit paragraph_id (F-5)"
    passage = live.call("a", "get_passage", {"paragraph_id": chosen["paragraph_id"]})
    assert passage.get("error") is None
    assert passage.get("paragraph_id") == chosen["paragraph_id"]
    assert passage.get("text")


def test_S_12(live: Live):
    """S-12: Umlaute und ß liefern Treffer ohne Kodierungsfehler."""
    hits = _hits(live.call("a", "search_corpus", {"query": "Verhältnis und Größe"}))
    assert hits
    for hit in hits:
        assert isinstance(hit.get("snippet"), str)


def test_P_01(live: Live):
    """P-01: get_passage liefert Volltext, source_id, Segment und vollständiges citation."""
    body = live.call("a", "get_passage", {"paragraph_id": live.para_id})
    assert body.get("error") is None
    assert body.get("source_id") == live.source_id
    assert body.get("text")
    plain = lambda value: str(value).replace("\u00ad", "")
    assert plain(live.preview[:24]) in plain(body["text"])
    assert body.get("segment_title") or body.get("segment_index") is not None
    problem = citation_problem(body.get("citation"))
    assert problem is None, problem


def test_P_02(live: Live):
    """P-02: Segment aus get_passage trifft in get_protocol dasselbe Kapitel wie der Slug."""
    passage = live.call("a", "get_passage", {"paragraph_id": live.para_id})
    by_index = live.call(
        "a",
        "get_protocol",
        {"source_id": passage["source_id"], "segment_slug": str(passage["segment_index"])},
    )
    by_slug = live.call(
        "a",
        "get_protocol",
        {"source_id": live.source_id, "segment_slug": live.segment},
    )
    assert by_index.get("code") != "segment_not_found"
    assert by_index.get("protocol_id") == by_slug.get("protocol_id")
    assert by_index.get("message") == by_slug.get("message")


def test_P_03(live: Live):
    """P-03: gültige unbekannte UUID liefert paragraph not found."""
    missing = str(uuid.uuid4())
    body = live.call("a", "get_passage", {"paragraph_id": missing})
    assert body.get("error") == "paragraph not found"
    assert "code" not in body


def test_P_04(live: Live):
    """P-04: ungültiges Format liefert paragraph not found, keinen Validierungsfehler."""
    body = live.call("a", "get_passage", {"paragraph_id": "abc"})
    assert body.get("error") == "paragraph not found"
    assert not has_stacktrace(body)


def test_V_01(live: Live):
    """V-01: list_volumes liefert source_id, display_name und source_type."""
    rows = _hits(live.call("a", "list_volumes"))
    assert rows
    for row in rows:
        assert row.get("source_id") and row.get("display_name") and row.get("source_type")


def test_V_02(live: Live, remark):
    """V-02: Anzahl der Bände entspricht volumes.count."""
    rows = _hits(live.call("a", "list_volumes"))
    remark(f"gezählt {len(rows)}, YAML {live.volume_count}")
    assert len(rows) == live.volume_count


def test_V_03(live: Live):
    """V-03: nackte Body-source_ids aus der Suche stehen in list_volumes."""
    hits = _hits(live.call("a", "search_corpus", {"query": QUERY, "types": ["text"], "limit": 10}))
    volumes = {row["source_id"] for row in _hits(live.call("a", "list_volumes"))}
    bare = []
    for hit in hits:
        source = str(hit.get("source_id") or "")
        if not source or source.endswith(":quotes") or source.endswith(":summary"):
            continue
        bare.append(source.split(":")[0])
    assert bare
    missing = [source for source in bare if source not in volumes]
    assert not missing, missing


@pytest.mark.xfail(strict=True, reason="Befund F-3")
def test_V_04(live: Live):
    """V-04: display_name ist ein lesbarer Titel, kein Author#Title#Index und ohne GA-Nummer."""
    import re

    rows = _hits(live.call("a", "list_volumes"))
    for row in rows:
        name = str(row.get("display_name") or "")
        assert name.strip()
        assert "#" not in name
        assert re.search(r"GA\s*\d", name, re.IGNORECASE) is None, name


def test_V_05(live: Live):
    """V-05: jedes Band-Element hat is_primary, ga und zyklus."""
    rows = _hits(live.call("a", "list_volumes"))
    for row in rows:
        assert "is_primary" in row and "ga" in row and "zyklus" in row


def test_LCT_01(live: Live):
    """LCT-01: list_lectures liefert die Katalogzahl, Titel und zitierform ohne GA."""
    body = live.call("a", "list_lectures", {"ga": live.lecture_ga})
    assert body.get("lecture_count") == live.lecture_count
    assert len(body.get("lectures") or []) == live.lecture_count
    for lecture in body["lectures"]:
        title = str(lecture.get("vortragstitel") or lecture.get("display_title") or "")
        zitier = str(lecture.get("zitierform") or "")
        assert "GA " not in title and "GA " not in zitier


def test_LCT_02(live: Live):
    """LCT-02: UUID und Katalog-ID liefern dieselbe Vortragszeile."""
    by_uuid = live.call("a", "get_lecture_info", {"source_id": live.lecture_source_id})
    by_id = live.call("a", "get_lecture_info", {"source_id": live.lecture_id})
    assert by_uuid.get("lecture_id") == by_id.get("lecture_id") == live.lecture_id
    for body in (by_uuid, by_id):
        assert body.get("ort")
        assert body.get("datum")
        assert "zyklus" in body
        assert "has_chunks" in body


def test_LCT_03(live: Live):
    """LCT-03: leere source_id bzw. ga liefern empty_source_id bzw. empty_ga."""
    info = live.call("a", "get_lecture_info", {"source_id": ""})
    lectures = live.call("a", "list_lectures", {"ga": ""})
    assert info.get("code") == "empty_source_id"
    assert lectures.get("code") == "empty_ga"


def test_LCT_04(live: Live):
    """LCT-04: unbekannte UUID liefert lecture_not_found."""
    body = live.call("a", "get_lecture_info", {"source_id": str(uuid.uuid4())})
    assert body.get("code") == "lecture_not_found"


def test_G_01(live: Live):
    """G-01: Protokoll kommt zeitlich sortiert, mit entry_type, citation und chapter_citation."""
    created = live.call(
        "a",
        "append_to_protocol",
        {
            "source_id": live.source_id,
            "segment_slug": live.segment,
            "entry_type": "note",
            "content": "[TEST] G-01 Eintrag",
            "paragraph_id": live.para_id,
        },
    )
    assert created.get("ok") is True
    body = live.call("a", "get_protocol", {"source_id": live.source_id, "segment_slug": live.segment})
    entries = body.get("entries") or []
    assert entries
    stamps = [entry["created_at"] for entry in entries]
    assert stamps == sorted(stamps)
    ours = [entry for entry in entries if str(entry.get("content") or "").startswith("[TEST] G-01")]
    assert ours and ours[0]["entry_type"] == "note"
    assert citation_problem(ours[0].get("citation")) is None
    assert citation_problem(body.get("chapter_citation")) is None


def test_G_02(live: Live):
    """G-02: leeres Kapitel liefert protocol null und die feste Meldung. Läuft vor A-01."""
    body = live.call(
        "a",
        "get_protocol",
        {"source_id": live.source_id, "segment_slug": live.segment_empty},
    )
    assert body == {"protocol": None, "message": "No protocol found for this chapter."}


def test_G_03(live: Live):
    """G-03: Segment-Index liefert dasselbe Protokoll wie der Slug."""
    slug = live.call("a", "get_protocol", {"source_id": live.source_id, "segment_slug": live.segment})
    index = live.call("a", "get_protocol", {"source_id": live.source_id, "segment_slug": live.segment_idx})
    assert index.get("code") != "segment_not_found"
    assert index.get("protocol_id") == slug.get("protocol_id")
    assert [entry.get("entry_id") for entry in index.get("entries") or []] == [
        entry.get("entry_id") for entry in slug.get("entries") or []
    ]


def test_G_04(live: Live):
    """G-04: unbekannte source_id liefert einen Fehler."""
    body = live.call(
        "a",
        "get_protocol",
        {"source_id": str(uuid.uuid4()), "segment_slug": live.segment},
    )
    assert body.get("error")
    assert not has_stacktrace(body)


def test_G_05(live: Live):
    """G-05: User B sieht den Protokolleintrag von User A nicht."""
    marker = "[TEST] G-05 nur Anton"
    created = live.call(
        "a",
        "append_to_protocol",
        {
            "source_id": live.source_id,
            "segment_slug": live.segment,
            "entry_type": "insight",
            "content": marker,
        },
    )
    assert created.get("ok") is True
    foreign = live.call("b", "get_protocol", {"source_id": live.source_id, "segment_slug": live.segment})
    blob = str(foreign.get("entries") or "")
    assert marker not in blob
    if foreign.get("entries"):
        assert all(marker not in str(entry.get("content")) for entry in foreign["entries"])


def test_H_01(live: Live):
    """H-01: frischer Handoff liefert markierten Text, Frage, Absatz und citation."""
    body = live.call("a", "get_handoff", {"handoff_id": live.handoff_id})
    assert str(body.get("marked_text") or "").startswith("[TEST]")
    assert str(body.get("user_question") or "").startswith("[TEST]")
    assert body.get("paragraph_id") == live.para_id
    assert citation_problem(body.get("citation")) is None


def test_H_02(live: Live, remark):
    """H-02: abgelaufener Handoff liefert Teildaten und expired true."""
    body = live.call("a", "get_handoff", {"handoff_id": live.handoff_old})
    assert body.get("expired") is True
    assert body.get("paragraph_id") == live.para_id
    remark("Felder: " + ", ".join(sorted(body.keys())))


def test_H_03(live: Live):
    """H-03: unbekannte 5-stellige ID liefert handoff not found."""
    unknown = "zzzzz"
    if unknown in {live.handoff_id, live.handoff_old, live.handoff_b}:
        unknown = "qqqqq"
    body = live.call("a", "get_handoff", {"handoff_id": unknown})
    assert body.get("error") == "handoff not found"


def test_H_04(live: Live):
    """H-04: 4-stellige ID liefert handoff not found, keinen Validierungsfehler."""
    body = live.call("a", "get_handoff", {"handoff_id": "abcd"})
    assert body.get("error") == "handoff not found"
    assert not has_stacktrace(body)


def test_H_05(live: Live):
    """H-05: derselbe Handoff ist zweimal gleich lesbar."""
    first = live.call("a", "get_handoff", {"handoff_id": live.handoff_id})
    second = live.call("a", "get_handoff", {"handoff_id": live.handoff_id})
    assert first == second


def test_H_06(live: Live):
    """H-06: Anton liest Antonias Handoff nicht, Antonia liest Antons Handoff nicht."""
    as_anton = live.call("a", "get_handoff", {"handoff_id": live.handoff_b})
    as_antonia = live.call("b", "get_handoff", {"handoff_id": live.handoff_id})
    assert as_anton.get("error") == "handoff not found"
    assert as_antonia.get("error") == "handoff not found"
    assert "[TEST]" not in str(as_anton.get("marked_text") or "")
    assert "[TEST]" not in str(as_antonia.get("marked_text") or "")


def test_H_07(live: Live):
    """H-07: Absatz aus dem Handoff ist über get_passage lesbar."""
    handoff = live.call("a", "get_handoff", {"handoff_id": live.handoff_id})
    passage = live.call("a", "get_passage", {"paragraph_id": handoff["paragraph_id"]})
    assert passage.get("text")
    assert passage.get("paragraph_id") == live.para_id


def test_CITE_01(live: Live):
    """CITE-01: Suche und get_passage liefern ein übernehmbares zitierform ohne GA."""
    hits = _hits(live.call("a", "search_corpus", {"query": QUERY}))
    cited = next((hit for hit in hits if hit.get("paragraph_id")), None)
    assert cited is not None
    assert citation_problem(cited.get("citation")) is None
    passage = live.call("a", "get_passage", {"paragraph_id": live.para_id})
    assert citation_problem(passage.get("citation")) is None


def test_CITE_02(live: Live, remark):
    """CITE-02: get_passage eines Vortrags hat segment_kind vortrag sowie ort und datum."""
    info = live.call("a", "get_lecture_info", {"source_id": live.lecture_id})
    title = info.get("vortragstitel") or info.get("display_title")
    hits = _hits(live.call("a", "search_corpus", {"query": title, "types": ["text"], "limit": 10}))
    chosen = next((hit for hit in hits if hit.get("chunk_type") == "talk" and hit.get("paragraph_id")), None)
    assert chosen is not None, "kein Vortragsabsatz mit paragraph_id (F-5)"
    passage = live.call("a", "get_passage", {"paragraph_id": chosen["paragraph_id"]})
    citation = passage.get("citation") or {}
    assert citation.get("segment_kind") == "vortrag"
    assert citation.get("ort") and citation.get("datum")
    remark(citation.get("zitierform") or "")


def test_CITE_03(live: Live):
    """CITE-03: Handoff und Protokolleintrag mit Absatz tragen citation."""
    handoff = live.call("a", "get_handoff", {"handoff_id": live.handoff_id})
    assert citation_problem(handoff.get("citation")) is None
    live.call(
        "a",
        "append_to_protocol",
        {
            "source_id": live.source_id,
            "segment_slug": live.segment,
            "entry_type": "note",
            "content": "[TEST] CITE-03",
            "paragraph_id": live.para_id,
        },
    )
    protocol = live.call("a", "get_protocol", {"source_id": live.source_id, "segment_slug": live.segment})
    ours = [entry for entry in protocol.get("entries") or [] if str(entry.get("content") or "").startswith("[TEST] CITE-03")]
    assert ours
    assert citation_problem(ours[0].get("citation")) is None


def test_X_01(live: Live, remark):
    """X-01: search_corpus warm unter 3 s; der erste Aufruf wird getrennt notiert."""
    _, cold = live.timed_call("a", "search_corpus", {"query": QUERY})
    _, warm = live.timed_call("a", "search_corpus", {"query": QUERY})
    remark(f"erster Aufruf {cold:.2f}s, zweiter {warm:.2f}s")
    assert warm < 3


def test_X_02(live: Live, remark):
    """X-02: übrige Tools unter 1 s."""
    calls = [
        ("whoami", {}),
        ("get_passage", {"paragraph_id": live.para_id}),
        ("list_volumes", {}),
        ("get_lecture_info", {"source_id": live.lecture_id}),
        ("list_lectures", {"ga": live.lecture_ga}),
        ("get_protocol", {"source_id": live.source_id, "segment_slug": live.segment}),
        ("list_work_texts", {}),
        ("get_work_text", {"note_id": live.note_a}),
        ("get_handoff", {"handoff_id": live.handoff_id}),
    ]
    slow = []
    notes = []
    for tool, args in calls:
        _, elapsed = live.timed_call("a", tool, args)
        notes.append(f"{tool} {elapsed:.2f}s")
        if elapsed >= 1:
            slow.append(f"{tool} {elapsed:.2f}s")
    remark("; ".join(notes))
    assert not slow, ", ".join(slow)


def test_X_04(live: Live):
    """X-04: Fehler sind Objekte mit error und optional code, ohne Stacktrace."""
    samples = [
        live.call("a", "search_corpus", {"query": ""}),
        live.call("a", "get_passage", {"paragraph_id": "abc"}),
        live.call("a", "get_handoff", {"handoff_id": "abcd"}),
        live.call("a", "get_lecture_info", {"source_id": ""}),
    ]
    for body in samples:
        assert isinstance(body, dict) and isinstance(body.get("error"), str)
        if "code" in body:
            assert isinstance(body["code"], str) and body["code"]
        assert not has_stacktrace(body)
