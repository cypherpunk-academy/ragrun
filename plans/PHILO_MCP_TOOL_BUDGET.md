# Philo MCP: Tool-Budget und Grenzen

**Stand:** 2026-10-19  
**Code:** `app/mcp_server/server.py` → `_register_tools()`  
**Testplan:** `tests/mcp/TESTPLAN_MCP.md`

Dieses Dokument ist die Leitplanke für neue MCP-Tools am Philo-Server. Es ersetzt keine Produktentscheidung, aber: Was hier als „nicht in MCP“ steht, gehört in Scripts, Admin-API oder ragprep — nicht in den Assistenten-Connector.

---

## Ausgangslage

Philo ist ein **spezialisierter** Assistent (Steiner-Korpus + Struktur + persönliche Notizen). Das Modell wählt bei jedem Schritt ein Tool. MCP liefert keine Tool-Gruppen oder „Hide“ — alles Registrierte erscheint im Menü.

**Aktuell ~18 Tools** (Stand 19.10.2026), grob in fünf Jobs:

| Job | Tools (Beispiele) |
|-----|-------------------|
| Korpus lesen & belegen | `search_corpus`, `get_passage` |
| GA-Wortlaut / Forschung | `search_ga`, `search_ga_summaries` |
| Typologien (Struktur) | `list_typologies`, `list_typology`, `search_typology` |
| Katalog | `list_volumes`, `list_lectures`, `get_lecture_info` |
| Nutzer & App-Kontext | `get_protocol`, `append_to_protocol`, `list_work_texts`, `get_work_text`, `create_work_text`, `update_work_text`, `get_handoff` |
| Auth | `whoami` |

---

## Weiche und harte Grenze

| | Richtwert |
|---|-----------|
| **Weiche Obergrenze** | **~25 Tools** — darüber nur mit klaren Clustern in `_SERVER_INSTRUCTIONS` und ohne überlappende Such-Tools |
| **Harte Regel** | Kein Tool für **eine** Typologie, **ein** Buch oder **einen** Pipeline-Schritt (Import, Embed, Delete, Sync) |
| **Such-Tools** | Maximal **~4** Varianten mit unterschiedlichem Korpus und Antwort-Stil (Korpus, GA, GA-Summaries, Typologie) — nicht weiter aufsplitten |

Unter ~25 Tools ist die **Anzahl** selten das Problem; kritisch sind **Namens- und Job-Überlappung** (vier `search_*`) und **Schreib-Tools ohne expliziten Nutzerauftrag**.

---

## Wann ist ein neues Tool sinnvoll?

**Ja**, wenn:

- ein **neuer Datentyp** oder **neue Partition** (andere Tabelle, andere Zitierregeln) — z. B. Typologie vs. `rag_paragraphs`
- **Lesen vs. Schreiben** klar getrennt bleiben soll
- Parameter an einem bestehenden Tool **unübersichtlich** werden (selten; Merge hat Vorrang)

**Nein / lieber anders**, wenn:

- nur ein **Filter** auf bestehende Suche (→ Parameter an `search_corpus` / `search_typology`)
- **Ops** (ragprep, embed, deprecate, Supabase)
- **Admin** oder einmalige Wartung
- das Modell dasselbe mit **zwei Tool-Aufrufen** ohne neue Surface schon kann

---

## Merge-Kandidaten (sparsam einsetzen)

| Heute | Alternative | Trade-off |
|-------|-------------|-----------|
| `list_typologies` + `list_typology` | ein Tool, `name` optional (leer = Katalog) | −1 Tool, +Komplexität in Docstring |
| `search_ga` + `search_ga_summaries` | ein Tool mit `mode: wording \| summary` | nur wenn Verwechslungsrate hoch wird |

**Getrennt lassen:** `search_typology` ≠ `search_corpus` (Metadaten, Volltext, **kein wörtliches Zitat** vs. Beleg mit `citation`).

Geplante Erweiterungen, die **noch ins Budget passen** (~20 Tools gesamt):

- `get_passage`: optionale Nachbar-Absätze (Snippet, gleiches Segment)
- optional ein **`get_typology_chunk(chunk_id)`**, falls Lesen ohne Volltreffer-Suche nötig wird — nicht pro Typologie ein eigenes Tool

---

## Antwort-Stil vs. Tool-Anzahl

Regeln, die **kein** extra Tool brauchen, gehören in `_SERVER_INSTRUCTIONS` (siehe Abschnitt Typologien: Strukturhilfe, paraphrasieren, Steiner-Belege nur über Korpus).

Technische Reduktion von Abschreib-Tendenz (optional später):

- Typology-Treffer in `search_corpus` nur Metadaten + Verweis auf `search_typology` — **kein** zusätzliches Tool, Verhalten in `app_search_service` / `search_corpus`.

---

## Symptome „zu viel“ oder „falsch designt“

- Mehrere `search_*`-Aufrufe für **dieselbe** Nutzerfrage hintereinander
- Verwechslung `paragraph_id` (Buch) vs. `chunk_id` (Typologie)
- Schreib-Tools ohne ausdrückliche Bitte des Nutzers
- Neue Features als **+1 Tool** statt Parameter oder Instruction

Dann zuerst Instructions und Docstrings schärfen; erst danach mergen oder Tools entfernen.

---

## Checkliste vor Merge eines neuen Tools

1. Welchem der **fünf Jobs** ordnet es sich zu? (Wenn keinem → nicht MCP.)
2. Erhöht es die Zahl der **Such-**Tools über vier? (Wenn ja → Parameter statt neues Tool.)
3. Ist es **schreibend**? → In Instructions: nur auf Nutzeranfrage.
4. Steht die **Zitier-/Paraphrase-Regel** im Server-Instruction-Block, nicht nur im Tool-Docstring?
5. Ist `tests/mcp/TESTPLAN_MCP.md` um **mindestens einen** Live- oder Unit-Hinweis ergänzt?

---

## Verwandte Pläne

- Typologie-MCP (Stufe 5): `plans/TYPOLOGY_GROUPS_HEALTH.md`
- Live-Tests: `tests/mcp/TESTPLAN_MCP.md`
