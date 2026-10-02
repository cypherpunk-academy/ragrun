# Testplan: Philo-MCP-Server

Stand: 1.10.2026 · Gilt für: Philo Dev (ngrok) und Philo Staging · Branch `philo-claude-integration`

**Gemeinsame Datei:** Cursor (Repo) schreibt hier. Fixtures nur in `ragrun/tests/mcp/mcp_testdata.yaml`. Vor dem Überschreiben den aktuellen Stand lesen.

## Worum geht es?

Dieser Plan prüft die **13 Tools** des Philo-MCP-Servers direkt (pytest, Cursor-Proxy, MCP Inspector). Sprachmodelle (Claude u. a.) sind nicht Teil der automatisierten Suite. App-Sichtprüfungen (A-09, C-04) bleiben manuell.

**Fixtures:** Alle Platzhalter kommen ausschließlich aus der Datei im **ragrun**-Repo (nicht ragkeep): [`tests/mcp/mcp_testdata.yaml`](https://github.com/cypherpunk-academy/ragrun/blob/philo-claude-integration/tests/mcp/mcp_testdata.yaml) — lokal `ragrun/tests/mcp/mcp_testdata.yaml`. Keine IDs hier festschreiben. Handoffs erzeugt `scripts/mcp_tests/test_handoffs.py` vor jedem Lauf neu. Dauerhafte Notizen: `scripts/mcp_tests/seed_fixtures.py` (`[FIXTURE]`). Aufräumen: `scripts/mcp_tests/test_cleanup.py` (nur `[TEST]`).

## Welche Tools werden geprüft?

Registriert in `app/mcp_server/server.py` → `_register_tools()` (Stand 1.10.2026). Namen sind die MCP-`name=`-Strings.

| Tool | Parameter | Art | Code |
|---|---|---|---|
| `whoami` | – | lesen | `server.py` |
| `search_corpus` | `query`, `limit` (1–20, Default 10), `types` (`text`, `concept`, `quote`, `chapter_summary`) | lesen | `tools.py` |
| `get_passage` | `paragraph_id` | lesen | `tools.py` |
| `list_volumes` | – | lesen | `tools.py` |
| `get_lecture_info` | `source_id` (Katalog-UUID oder `YYYYMMDD[a-z]`) | lesen | `tools.py` |
| `list_lectures` | `ga` (intern, z. B. `"151"`, `"337b"`) | lesen | `tools.py` |
| `get_protocol` | `source_id`, `segment_slug` | lesen | `tools.py` |
| `list_work_texts` | `limit` (1–50, Default 20), `text_type` | lesen | `tools.py` |
| `get_work_text` | `note_id` | lesen | `tools.py` |
| `create_work_text` | `title`, `content`, `text_type` (note, essay, draft; Default note), `paragraph_id`, `conversation_url` | schreiben | `tools.py` |
| `update_work_text` | `note_id`, `content`, `expected_version`, `title`, `status` (draft, final) | schreiben | `tools.py` |
| `append_to_protocol` | `source_id`, `segment_slug`, `entry_type` (note, question, insight, summary), `content`, `paragraph_id`, `conversation_url` | schreiben | `tools.py` |
| `get_handoff` | `handoff_id` (5 Zeichen, keine Längenprüfung) | lesen | `tools.py` |

`get_lecture_info` / `list_lectures` liegen im Branch. Ein bereits geöffneter Claude-Chat sieht sie erst nach **Neuverbinden** des MCP.

### Lecture-Tools — Schema (Code)

`get_lecture_info(source_id)` → `{ lecture_id, source_id, ga, vortragstitel, display_title, datum, ort, reihe, zyklus, zyklus_titel, has_chunks, zitierform }` oder `{error, code}` (`empty_source_id` / `lecture_not_found`). `ga` intern; Titel und `zitierform` ohne GA-Nummer.

`list_lectures(ga)` → `{ ga, lecture_count, lectures: [...] }` (gleiche Felder je Eintrag, ohne Top-Level-`ga` am Lecture). Unbekannte GA → `lecture_count: 0`, keine Exception. Leer → `empty_ga`.

## Was muss vor dem Test bereitstehen?

Zwei Testkonten (YAML `users.a` / `users.b`): Anton Testo und Antonia Testa, Dev. Schreibtests nur unter diesen Konten.

Platzhalter → YAML-Pfad:

| Platzhalter | YAML |
|---|---|
| `<USER_A>` / `<USER_B>` | `users.a.user_id` / `users.b.user_id` |
| `<SOURCE_ID>` | `volumes.source_id.id` (PdF) |
| `<SOURCE_ID_BG>` / `<SEARCH_BG>` | `volumes.source_id_bg.id` / `.search_term` |
| `<SEGMENT>` / `<SEGMENT_IDX>` / `<SEGMENT_EMPTY>` | `segments.segment` / `.segment_idx` / `.segment_empty` |
| `<PARA_ID>` | `paragraph.para_id` |
| `<NOTE_A>` / `<ESSAY_A>` / `<NOTE_B>` | `work_texts.*` |
| `<HANDOFF_ID>` / `<HANDOFF_OLD>` / `<HANDOFF_B>` | `handoffs.*` (nach `test_handoffs.py`) |
| `<LECTURE_GA>` / `<LECTURE_ID>` / `<LECTURE_SOURCE_ID>` | `lectures.ga` / `.lecture_id` / `.source_id` |
| `<VOLUME_COUNT>` | `volumes.count` (Dev, ~333) |

Alle Testeinträge aus einem Lauf: Präfix `[TEST]` im Titel oder Inhalt. Dauerhafte Fixture-Notizen: Präfix `[FIXTURE]` (überlebt Cleanup).

## Ebene A: Funktioniert jedes Tool für sich?

### whoami

| ID | Aufruf | Erwartung |
|---|---|---|
| W-01 | whoami mit gültigem Token | `user_id` und E-Mail von User A |
| W-02 | whoami ohne Token | Auth-Fehler (oft Transport, nicht Tool-Body) |
| W-03 | abgelaufenes oder manipuliertes Token | Auth-Fehler |

### search_corpus

`types` filtert App-Typen. Die **ausgegebenen** `chunk_type`-Werte sind intern: `text` → `book` / `secondary_book` / `talk`; `concept` → `begriff` / `typology`; `quote` → `quote` / (intern) `quote_explanation`; `chapter_summary` → `chapter_summary`. Treffer mit `paragraph_id` tragen ein `citation`-Objekt. **Nicht** in der Suchantwort: `is_primary`, `ga`, `zyklus`.

| ID | Aufruf | Erwartung |
|---|---|---|
| S-01 | query="Freiheit und Erkenntnis" | bis zu 10 Treffer; `chunk_id`, Snippet; bei `paragraph_id` vollständiges `citation` (siehe Beleg) |
| S-02 | englische query ("freedom and knowledge") | **live 1.10:** 0 Treffer. Embeddings matchen deutsch; kein Soll mehr „englisch → deutsch“. Optional später: eigene EN-Query oder als bekannt offen markieren |
| S-03 | types=["text"] | nur `book`, `secondary_book` oder `talk` — nie der String `text` |
| S-04 | types=["concept"], dann ["quote"], dann ["chapter_summary"] | nur `begriff`/`typology` bzw. `quote` bzw. `chapter_summary` |
| S-05 | types=["concept","quote"] | Mischung aus diesen internen Typen, sonst nichts |
| S-06 | types=["unbekannt"] oder `["bogus_type"]` | `{error, code: invalid_types}` plus erlaubte Liste |
| S-07 | limit=1 und limit=20 | genau 1 bzw. höchstens 20 |
| S-08 | limit=0 und limit=21 | kein Fehler, stille Begrenzung 1 bzw. 20 |
| S-09 | query="" | `{error, code: empty_query}`, kein Serverfehler |
| S-10 | query=`<SEARCH_BG>` (YAML: `code is law`, nicht `neutral search`) | Treffer aus `<SOURCE_ID_BG>` (Lessig) |
| S-11 | `paragraph_id` aus S-01 an `get_passage` | Absatz gefunden, wenn die Treffer-Zeile eine `paragraph_id` hat. Vortrags-Body ohne `paragraph_id` → **F-5**, Kette bricht |
| S-12 | Umlaute, ß, Sonderzeichen | korrekte Treffer, keine Kodierungsfehler |

### get_passage

| ID | Aufruf | Erwartung |
|---|---|---|
| P-01 | `<PARA_ID>` | Volltext, `source_id`, Segment; `citation` vollständig; Join `rag_sources`/`rag_lecture_catalog` wirft keinen Typfehler |
| P-02 | Metadaten aus P-01 an `get_protocol` | richtiges Kapitel |
| P-03 | gültige, unbekannte UUID | `{error: "paragraph not found"}` |
| P-04 | ungültiges Format (`"abc"`) | `{error: "paragraph not found"}` — **kein** Validierungsfehler (live 1.10) |

### list_volumes

| ID | Aufruf | Erwartung |
|---|---|---|
| V-01 | list_volumes | `source_id`, `display_name`, `source_type` |
| V-02 | Anzahl | YAML `volumes.count` (Dev ~333; Bücher + Vorträge + Sekundär) |
| V-03 | source_ids aus S-01 | nur **nackte** Body-`source_id`s müssen in `list_volumes` stehen. Suffix `:quotes` / `:summary` sind eigene Partitionen (**F-7**), nicht in der Bandliste |
| V-04 | display_name | lesbare Titel, kein `Author#Title#Index`; GA-Nummer nicht im Titel |
| V-05 | Metadaten | jedes Element hat `is_primary`, `ga`, `zyklus` (Code `list_volumes` / `app_catalog_repository.py`). Fehlen sie im Client: MCP neu verbinden, nicht den Test streichen |

### get_lecture_info / list_lectures

| ID | Aufruf | Erwartung |
|---|---|---|
| LCT-01 | `list_lectures(<LECTURE_GA>)` | `lecture_count` = YAML `lectures.lecture_count`; Titel/`zitierform` ohne `GA ` |
| LCT-02 | `get_lecture_info(<LECTURE_SOURCE_ID>)` und `(<LECTURE_ID>)` | dieselbe Vortragszeile; `ort`, `datum`, `zyklus`, `has_chunks` |
| LCT-03 | `get_lecture_info("")` / `list_lectures("")` | `empty_source_id` / `empty_ga` |
| LCT-04 | unbekannte UUID | `lecture_not_found` |

### get_protocol

| ID | Aufruf | Erwartung |
|---|---|---|
| G-01 | `<SOURCE_ID>`, `<SEGMENT>` mit Protokoll | Einträge zeitlich, `entry_type`; Einträge mit `paragraph_id` haben `citation`; Antwort hat `chapter_citation` |
| G-02 | `<SOURCE_ID>`, `<SEGMENT_EMPTY>` (vor A-01) | `{"protocol": null, "message": "No protocol found for this chapter."}` (live 1.10) |
| G-03 | `<SEGMENT_IDX>` statt Slug | dasselbe Protokoll wie G-01 (`resolve_segment_slug`) |
| G-04 | unbekannte source_id | Fehler / Segment nicht gefunden |
| G-05 | User B liest Protokoll von User A | User B sieht nur eigene Einträge |

### append_to_protocol

| ID | Aufruf | Erwartung |
|---|---|---|
| A-01 | entry_type="insight", `<SEGMENT_EMPTY>` | Protokoll angelegt, Eintrag drin |
| A-02 | je note, question, summary | alle drei mit korrektem Typ |
| A-03 | entry_type="foo" | **offen:** MCP prüft das Enum nicht (nur nicht-leer). Live unter Anton: speichert der Server `foo` oder lehnt die DB ab? |
| A-04 | mit paragraph_id | Eintrag verknüpft |
| A-05 | mit conversation_url | URL gespeichert (Anzeige in der App → Ebene B) |
| A-06 | zweimal hintereinander | beide Einträge bleiben |
| A-07 | content="" | Fehlermeldung (Soll; Code prüft content-Länge nicht explizit) |
| A-08 | ungültige source_id oder segment_slug | Fehler, kein verwaistes Protokoll |

### list_work_texts

| ID | Aufruf | Erwartung |
|---|---|---|
| L-01 | ohne Parameter | bis zu 20 eigene Texte; `display_title`, `title_source` |
| L-02 | text_type="essay" | nur Essays |
| L-03 | limit=1, 50, 0, 51 | Anzahl; 0→1, 51→50 still |
| L-04 | Nutzer ohne Texte | leere Liste |
| L-05 | User A listet | kein Text von User B |

### get_work_text

| ID | Aufruf | Erwartung |
|---|---|---|
| T-01 | `<NOTE_A>` als User A | title, content, version; `display_title`, `title_source`; bei Anker `citation` |
| T-02 | unbekannte note_id | „note not found“ |
| T-03 | `<NOTE_B>` als User A | nicht gefunden / kein Zugriff, keine Inhalte |

### create_work_text

| ID | Aufruf | Erwartung |
|---|---|---|
| C-01 | nur title und content | text_type="note", Version 1 |
| C-02 | text_type="essay" und "draft" | Typ übernommen |
| C-03 | text_type="foo" | **offen:** MCP prüft das Enum nicht. Live unter Anton, nicht unter dem Produktkonto |
| C-05 | mit conversation_url | URL gespeichert |
| C-06 | leerer title oder content | Fehlermeldung (Soll) |
| C-07 | content > 1 MB (UTF-8) | `{error, code: content_too_long}` |
| C-08 | danach list_work_texts | neuer Text erscheint |

### update_work_text

| ID | Aufruf | Erwartung |
|---|---|---|
| U-01 | get, dann update mit dieser Version | `{ok, new_version = alt + 1}` |
| U-02 | veraltete Version | `{error: "conflict"}`, Inhalt unverändert |
| U-03 | status="final", danach "draft" | Status wechselt |
| U-04 | status="foo" | **offen:** MCP prüft das Enum nicht. Live unter Anton |
| U-05 | ohne title | Titel bleibt |
| U-06 | mit neuem title | Titel geändert |
| U-07 | `<NOTE_B>` als User A | Fehler, keine Änderung |
| U-08 | zwei Clients, dieselbe Version | genau einer gewinnt, der andere conflict |

### get_handoff

| ID | Aufruf | Erwartung |
|---|---|---|
| H-01 | frischer `<HANDOFF_ID>` | markierter Text, Frage, Absatz; `citation` wenn `paragraph_id` |
| H-02 | `<HANDOFF_OLD>` | Teildaten, `expired: true` (`expires_at < now()`, TTL 24 h) |
| H-03 | unbekannte ID mit 5 Zeichen | `{error: "handoff not found"}` |
| H-04 | ID mit 4 Zeichen (`"abcd"`) | `{error: "handoff not found"}` — **kein** Validierungsfehler (live 1.10) |
| H-05 | `<HANDOFF_ID>` zweimal | dieselben Daten |
| H-06 | `<HANDOFF_B>` als Anton, `<HANDOFF_ID>` als Antonia | jeweils kein Zugriff |
| H-07 | Absatz aus H-01 an `get_passage` | Absatz gefunden |

### Beleg (`citation`)

Pflichtfelder (`citation.py` `row_to_citation`): `paragraph_id`, `band`, `segment_kind` (`kapitel` \| `vortrag`), `segment_title`, `paragraph_number`, `ort`, `datum`, `zitierform`, `return_url`. Live 1.10 an `search_corpus` und `get_passage` bestätigt.

| ID | Aufruf | Erwartung |
|---|---|---|
| CITE-01 | S-01 / P-01 | Objekt vorhanden; `zitierform` wörtlich übernehmbar; kein `GA ` in `zitierform` |
| CITE-02 | Vortrags-`get_passage` | `segment_kind=vortrag`; `ort`/`datum` gesetzt |
| CITE-03 | H-01 / G-01 mit Absatz | `citation` am Handoff bzw. am Eintrag |

## Manuell in der App

| ID | Prüfung | Erwartung |
|---|---|---|
| A-09 | Eintrag aus A-01 in der App | richtiges Kapitel |
| C-04 | create mit `paragraph_id`, in der App | Verknüpfung sichtbar |

## Was gilt übergreifend?

| ID | Prüfung | Erwartung |
|---|---|---|
| X-01 | `search_corpus` | Warm unter 3 s anstreben; Kalt (Embedding/Qdrant) darf länger sein — getrennt notieren |
| X-02 | übrige Tools | unter 1 s (ohne Hybrid) |
| X-03 | Server weg (ngrok aus) | Client meldet den Ausfall verständlich |
| X-04 | Fehlerformat | `{error, code?}`; kein Stacktrace. Infra: `db_unavailable`, `qdrant_unavailable`, `embeddings_unavailable`, `upstream_unavailable` (`_tool_infra_error`) |
| X-05 | Tool-Beschreibungen | Defaults und erlaubte `types` stimmen; `chunk_type` in Treffern = interne Namen |
| X-06 | Dev und Staging | gleiche **Verträge**; Staging ohne `019_lecture_catalog` hat andere Mengen (V-02, Lecture-Tools) — kein Serverbug |

## Fehlervertrag

| Situation | `error` (Auszug) | `code` | Live / Code |
|---|---|---|---|
| leere Suche | query is required | `empty_query` | Code |
| unbekannte `types` | invalid types: …; allowed: … | `invalid_types` | live (S-06) |
| Absatz unbekannt / `"abc"` | paragraph not found | — | live (P-03, P-04) |
| Handoff unbekannt / `"abcd"` | handoff not found | — | live (H-03, H-04) |
| leeres Protokoll | message wie G-02 | — | live (G-02) |
| content > 1 MB | exceeds maximum length … | `content_too_long` | Code + RPC 020 |
| Lecture leer / unbekannt | … | `empty_source_id` / `empty_ga` / `lecture_not_found` | Code |
| Segment unbekannt | segment not found: … | `segment_not_found` | Code |
| Infra Postgres / Typvergleich | Exception-Name + Meldung | `db_unavailable` | Code |
| Infra Hybrid/HTTP | ReadError / Timeout | `upstream_unavailable` o. ä. | Code |

## Was ist aus dem Code festgelegt?

| Frage | Festlegung | Quelle | Betrifft |
|---|---|---|---|
| Ungültiges `types` | Fehler `invalid_types` | `tools.py` `_validate_search_types` (~70) | S-06 |
| `types` → `chunk_type` | Mapping in `_APP_TYPE_TO_CHUNK_TYPES` | `app_search_service.py` ~19–24 | S-03–S-05 |
| limit außerhalb | still 1–20 / 1–50 | `tools.py` 148, 569 | S-08, L-03 |
| Kapitel ohne Protokoll | `protocol: null` + message | `tools.py` 525 | G-02 |
| Slug = Index | `resolve_segment_slug` | `citation.py` | G-03 |
| Handoff | mehrfach; abgelaufen `expires_at < now()`; TTL 24 h; keine Längenprüfung | `tools.py` 929–988; `018_claude_integration.sql` | H-02, H-04, H-05 |
| content | max 1 MB UTF-8 | `tools.py` `_CONTENT_MAX_BYTES`; `020_note_content_max_length.sql` | C-07 |
| `entry_type` / `text_type` / `status` | **kein** Enum-Check im MCP | `append_to_protocol` / `create_work_text` / `update_work_text` | A-03, C-03, U-04 |
| `list_volumes`-Metadaten | `is_primary`, `ga`, `zyklus` | `tools.py` 349–357; `app_catalog_repository.py` 23–47 | V-05 |
| Suche ohne Band-Metadaten | `search_corpus` reicht `is_primary`/`ga`/`zyklus` nicht durch | `tools.py` 157–172 | kein S-13 auf Treffern |
| `rag_paragraphs.id` auf Dev | **text**, nicht uuid; Compare `p.id::text` | `citation.py` `fetch_citation` | P-01 |

## Welche Befunde stehen?

| ID | Stand | Rest |
|---|---|---|
| F-1 | **erledigt** (S-06 live) | — |
| F-2 | **erledigt** (G-03 Code) | Live G-03 unter Anton noch einmal |
| F-3 | **teilweise** | Titel/337b-Disambiguierung im Sync; Oberschlesien bewusst doppelt; Band und Einzelvortrag bleiben beide in `list_volumes` bis Produktentscheidung |
| F-4 | **erledigt** (1 MB) | C-07 live unter Anton |
| F-5 | **Fix im Code** (1.10, lokal: `Erster Vortrag` → pid + citation) | Titeltreffer umgingen `app_paragraph_chunk`. `app_search_service.py` reichert jetzt auch Title-Hits an. Live-MCP neu starten, dann CITE-02 / S-11 |
| F-6 | **erledigt auf Dev** (2.10) | LLM-Präambel in `quote_explanation` (36 Zeilen). Repair: `ragprep/scripts/strip-quote-explanation-preamble.ts`. Staging/Prod: derselbe Befehl mit `--env`. |
| F-7 | **neu (live 1.10)** | `quote` / `chapter_summary` haben `source_id` mit `:quotes` / `:summary` — in `list_volumes` nicht. V-03 gilt nur für Body-UUIDs (Absicht, nicht Bug der Liste) |

## Wie wird aufgeräumt?

`scripts/mcp_tests/test_cleanup.py` löscht `[TEST]`-Arbeitstexte und -Protokolleinträge. `[FIXTURE]`-Notizen bleiben. Staging per Skript, Dev nicht per Reset. Abgelaufene Handoffs verfallen.

## Wo stehen die Ergebnisse?

| Lauf | Datum | Umgebung | Client | bestanden | fehlgeschlagen | Bemerkung |
|---|---|---|---|---|---|---|
| 1 | 1.10.2026 | Dev | MCP live + Code | S-03/S-06, P-04, H-04, G-02 | alter Connector ohne Lecture-Tools | — |
| 1b | 1.10.2026 | Dev, 13 Tools nach Neuverbinden | Claude Live-MCP | P-01/03/04; H-03/04; G-02; S-01, S-03–S-09, S-11, S-12; V-01/04/05; LCT-01–04 | S-02 (EN→0 Treffer); S-10 alter Fixture-Term | F-5, F-6, F-7. `list_lectures(151)` einmal „No approval received“, Retry ok. Blockiert ohne Anton: G-01/03, A/C/U, H-01/02/05–07, V-02 |
