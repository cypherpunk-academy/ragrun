
### AUTHENTIC_CONCEPT_RETRIEVAL — Plan

### Ziel
`ragrun/app/retrieval/graphs/concept_explain_worldviews.py` wird funktional **aufgeteilt** in zwei neue Retrieval-Endpunkte:

- **`authentic_concept_explain.py`**: erzeugt zuerst eine **“Steiner-Start-Erklärung” ohne Qdrant**, verifiziert sie danach **mit Qdrant (chunk_type=book)** gegen Steiner-Textstellen, und generiert daraus einen **lexikonartigen Standard-Chunk**.
- **`translate_to_worldview.py`**: nimmt einen **beliebigen Standard-Chunk** (z. B. Output von `authentic_concept_explain`) und produziert pro Weltanschauung die Schritte **what?** (Verstehen/Essenz) und **how?** (Neuformulierung/Interpretation) analog zur Logik aus `concept_explain_worldviews`.

### Leitprinzipien / Constraints
- **Step 1.1 startet ohne Retrieval**: keine Qdrant-Suche im ersten Schritt; stattdessen LLM mit Philo-Systemprompt + neuem User-Prompt (“Steiner-Vorwissen”).
- **Standard-Chunk-Länge**: Ziel ~220–280 Wörter (max ~320) wie im bestehenden Prompt `ragkeep/assistants/philo-von-freisinn/prompts/concept-explain-user.prompt`.
- **Best-effort Persistenz/Telemetry**: wie in `GraphEventRecorder` und `retrieval_telemetry`: Fehler sollen nie den Endpunkt brechen.
- **Prompt-Quelle**: Prompt-Loader lesen aus `ragkeep/assistants/...` (konfigurierbar via `RAGRUN_ASSISTANTS_ROOT`; Default: `ragkeep/assistants`). Docker muss diese Pfade bereitstellen.

---

### Teil 1: Endpoint `authentic_concept_explain.py`

### 1) API-Surface (FastAPI)
- **Router-Ort**: `ragrun/app/retrieval/api/authentic_concept_explain.py`
- **Route** (Vorschlag, konsistent zu bestehenden Graph-Routen):
  - `POST /api/v1/agent/philo-von-freisinn/graphs/authentic-concept-explain`
- **Request (Vorschlag)**:
  - `concept: str` (required)
  - `collection: str = "philo-von-freisinn"` (optional; default wie bisher)
  - `verbose: bool = False` (optional; loggt Prompt/Debug ähnlich wie `_chat_with_retry(..., verbose=True)`)
  - optional: `k_verify: int` (z. B. 8–20), `hybrid: bool | None`
- **Response (Vorschlag)**:
  - `concept: str`
  - `steiner_prior_text: str` (Step 1.1 Output)
  - `verify_refs: List[str]` (Qdrant refs via `build_context`)
  - `verification_report: str` (Step 1.2 LLM Output: Bewertung + evtl. Korrekturhinweise)
  - `lexicon_entry: str` (Step 1.3 Output, Standard-Chunk)
  - `graph_event_id: str | None`

### 2) Orchestrierung (Chain/Graph)
- **Chain-Ort**: `ragrun/app/retrieval/chains/authentic_concept_explain.py`
- **Service-Ort**: `ragrun/app/retrieval/services/authentic_concept_explain_service.py`
- **Graph-Name** für Event Recorder: `authentic_concept_explain`
- **Steps**:

#### Step 1.1 — “Steiner-Prior” ohne Qdrant
**Ziel**: Ein **chunk-langer**, in sich geschlossener Text “was meint Steiner mit {concept}?” als Suchanker für spätere Retrievals.

- **Inputs**: `concept`
- **LLM**: DeepSeek Reasoner, “reasoning_client” je nach Provider-Konzept.
- **Prompts**:
  - **System**: `ragkeep/assistants/philo-von-freisinn/prompts/instruction.prompt`
  - **User (neu)**: Template “Steiner Prior Explain” (siehe Prompt-Sektion unten)
- **Output**: `steiner_prior_text` (≈ Standard-Chunk-Länge, sauber abgeschlossen)
- **Event**:
  - `step="steiner_prior"`
  - `prompt_messages`, `response_text`

#### Step 1.2 — Retrieval + Verifikation gegen Steiner-Chunks
**Ziel**: Mit `steiner_prior_text` (oder `concept`) **Steiner-Chunks** aus Qdrant holen und die Prior-Antwort **kritisch bewerten**: stimmt das mit Steiner überein?

- **Retrieval**:
  - **Query**: primär `steiner_prior_text` (weil semantisch reich), optional zusätzlich `concept` als Nebenquery (Hybrid/2-pass).
  - **Filter**: `chunk_type = "book"` (entspricht `book_types=["primary"]` in `payload_filter`)
  - **Retriever**: i. d. R. **dense** (lange Query), optional hybrid wenn `settings.use_hybrid_retrieval`.
  - **Postprocessing**: `rerank_by_embedding` + `build_context` wie im bestehenden Graph.
- **LLM-Verifikation** (neu):
  - Input: `steiner_prior_text` + `retrieved_context`
  - Output: **Bewertung** + **konkrete Abweichungen** + optional **Korrekturvorschlag**.
- **Event**:
  - `step="steiner_verify_retrieval"`: `query_text`, `context_refs`, `context_text`, `retrieval_mode`, `metadata` (k/widen/k_final)
  - `step="steiner_verify_reasoning"`: `prompt_messages`, `response_text`

**Definition “Bewertung” (Vorschlag, maschinenlesbar optional)**:
- Score 0–100 oder Labels {`high`, `medium`, `low`, `insufficient`}
- Liste “Behauptungen, die nicht im Kontext gestützt sind”
- Liste “Formulierungen, die Steiner untypisch sind”
- “Korrigierte Version” (optional; kann Step 1.3 speisen)

#### Step 1.3 — Lexikon-Eintrag aus verifiziertem Material
**Ziel**: Einen **lexikonartigen** Eintrag in Standard-Chunk-Länge erzeugen, der “stringt” erklärt (dein Wort: “stringt”).

- **Inputs**:
  - `concept`
  - `verification_report` (oder daraus extrahierte “Korrektur/Empfehlungen”)
  - `retrieved_context` (als Grounding)
- **LLM**:
  - bevorzugt Chat-Model (stabiler Stil), oder Reasoner wenn du “strenger” willst.
- **Prompts (neu)**:
  - “Lexikon Entry Builder” (siehe Prompt-Sektion)
- **Output**:
  - `lexicon_entry` (≈ 220–280 Wörter; max ~320; sauberer Abschluss)
- **Event**:
  - `step="steiner_lexicon"`

---

### Teil 2: Endpoint `translate_to_worldview.py`

### 1) API-Surface (FastAPI)
- **Router-Ort**: `ragrun/app/retrieval/api/translate_to_worldview.py`
- **Route (Vorschlag)**:
  - `POST /api/v1/agent/sigrid-von-gleich/graphs/translate-to-worldview`
- **Request (Vorschlag)**:
  - `text: str` (required; Standard-Chunk, z. B. `lexicon_entry`)
  - `worldviews: List[str]` (min 1; validiert via `ALLOWED_WORLDVIEWS`)
  - `concept: str | None` (optional; nur für GraphEventRecorder/Persistenz)
  - `collection: str = "sigrid-von-gleich"`
  - `verbose: bool = False`
  - optional: `max_concurrency: int = 4`
- **Response (Vorschlag)**:
  - `input_text: str`
  - `worldviews: List[{worldview, main_points, how_details, context1_refs, context2_refs, sufficiency, errors}]`
  - `graph_event_id: str | None`

### 2) Orchestrierung
Analog zur bestehenden `concept_explain_worldviews`-Kette, aber mit `text` als Eingang und optionaler Parallelisierung über `worldviews`.

**Wichtig:** Das ist Teil des **Assistenten `sigrid-von-gleich`**. Prompts liegen **pro Weltanschauung** unter `ragkeep/assistants/sigrid-von-gleich/worldviews/<WV>/prompts/`.

- **Fail-fast**: Wenn für eine angeforderte Weltanschauung `concept-explain-what.prompt` oder `concept-explain-how.prompt` fehlt, soll der Endpunkt **hart fehlschlagen** (HTTP 400). Die fehlenden Prompts werden später ergänzt.

- **Chain-Ort**: `ragrun/app/retrieval/chains/translate_to_worldview.py`
- **Service-Ort**: `ragrun/app/retrieval/services/translate_to_worldview_service.py`
- **Graph-Name** für Event Recorder: `translate_to_worldview`

#### Step 2.1 — what? und how? pro Weltanschauung (RAG-gestützt)
**Ziel**: Den Eingangstext erst **klar verstehen** (what), dann **weltanschauungs-gerecht neu formulieren** (how).

- **Input**: `text` (Standard-Chunk), `worldviews[]`
- **Retrieval (wie im Graph)**:
  - context1: `book_types=["primary"]`, `worldview=<wv>`, query = `text`
  - context2: `book_types=["primary","secondary"]`, `worldview=<wv>`, query = `text`
  - Postprocessing: `rerank_by_embedding` + `build_context`
- **LLM Steps**:
  - what: **Sigrid Prompt** `ragkeep/assistants/sigrid-von-gleich/worldviews/<WV>/prompts/concept-explain-what.prompt`
  - how: **Sigrid Prompt** `ragkeep/assistants/sigrid-von-gleich/worldviews/<WV>/prompts/concept-explain-how.prompt`
- **Outputs**:
  - `main_points` (what)
  - `how_details` (how)
  - `context1_refs`, `context2_refs`, `sufficiency`, `errors`
- **Events** (pro worldview):
  - `step="wv_context1_retrieval"` / `step="wv_what"` / `step="wv_context2_retrieval"` / `step="wv_how"` (gleiches Schema wie im bestehenden Graph, aber `query_text=text`)

---

### Prompt-Plan (neu)

**Ablage-Ort (Vorschlag)**: `ragrun/app/retrieval/prompts/authentic_concept_explain.py`

#### Prompt A — Step 1.1 “Steiner Prior Explain” (neu)
- **System**: unverändert aus Philo: `instruction.prompt`
- **User Template (neu)**:
  - Aufgabe: “Erkläre {concept} aus Sicht Rudolf Steiners.”
  - Länge: 220–280 Wörter (max ~320)
  - Form: zusammenhängend, lexikonartig
  - Keine Meta-Referenzen (kein “wie oben”, kein “Kontext”)
  - Abschluss: sauberer Schlusssatz
  - Wichtig: hier **kein** “Nutze ausschließlich folgenden Kontext”, da absichtlich ohne Retrieval.

#### Prompt B — Step 1.2 “Steiner Verify Against Context” (neu)
- **Input**: `steiner_prior_text`, `retrieved_context`
- **Output**: Bewertung + Abweichungen + optional korrigierte Version.
- **User Template (neu)** (Kernpunkte):
  - “Nutze **nur** den Kontext zur Prüfung.”
  - “Liste Behauptungen, die nicht durch Kontext gestützt sind.”
  - “Bewerte Übereinstimmung (Score/Label).”
  - “Wenn möglich: gib eine korrigierte Version, die strikt am Kontext bleibt.”

#### Prompt C — Step 1.3 “Lexikon Entry Builder” (neu)
- **Input**: `concept`, `retrieved_context`, optional `corrections`/`verification_report`
- **Output**: finaler Lexikon-Text in Standard-Chunk-Länge
- **User Template (neu)** (Kernpunkte):
  - “Schreibe einen Lexikon-Eintrag zu {concept} in Rudolf Steiners Begrifflichkeit.”
  - “Nutze ausschließlich Kontext (+ evtl. Korrekturanweisungen).”
  - “Keine Zitate/Quellenangaben im Text, sondern fließender Eintrag.”

#### Prompt D — Step 2.1 what/how
- **Quelle**: Prompt-Files unter `ragkeep/assistants/sigrid-von-gleich/worldviews/<WV>/prompts/`
- **Loader/Renderer**: `ragrun/app/retrieval/prompts/sigrid_von_gleich_worldviews.py` ersetzt Platzhalter:
  - `{{CONCEPT_EXPLANATION}}`, `{{CONTEXT1_K5}}`, `{{CONTEXT2_K10}}`, `{{MAIN_POINTS}}`, `{{WORLDVIEW_DESCRIPTION}}`

---

### Code-Änderungen (konkret, als Umsetzungs-Checklist)

#### Neue Dateien
- `ragrun/app/retrieval/api/authentic_concept_explain.py`
- `ragrun/app/retrieval/api/translate_to_worldview.py`
- `ragrun/app/retrieval/chains/authentic_concept_explain.py`
- `ragrun/app/retrieval/chains/translate_to_worldview.py`
- `ragrun/app/retrieval/services/authentic_concept_explain_service.py`
- `ragrun/app/retrieval/services/translate_to_worldview_service.py`
- `ragrun/app/retrieval/prompts/authentic_concept_explain.py` (Prompts A/B/C)

#### Anpassungen bestehender Dateien
- `ragrun/app/retrieval/api/__init__.py`
  - Router include:
    - `/agent/philo-von-freisinn`: `concept-explain-worldviews`, `authentic-concept-explain`
    - `/agent/sigrid-von-gleich`: `translate-to-worldview`
- (Optional, aber empfohlen) Prompt-Root-Fix:
  - `ragrun/app/retrieval/prompts/philo_von_freisinn.py`
  - `ragrun/app/retrieval/prompts/concept_explain_worldviews.py`
  - Ziel: Assistants-Pfade konsistent machen (siehe “Prompt-Quelle” oben).

#### Telemetry / Persistenz
- Reuse `GraphEventRecorder` Schema:
  - `authentic_concept_explain`: `steiner_prior`, `steiner_verify_retrieval`, `steiner_verify_reasoning`, `steiner_lexicon`
  - `translate_to_worldview`: reuses worldview steps aus dem bestehenden Graph
- Optional: `retrieval_telemetry` erweitern um ein generisches `record_graph(name, metadata)`; ansonsten bleibt Telemetry erstmal wie bisher (nur worldviews/standard retrieval).

---

### Offene Fragen (damit wir den Plan “hart” machen)
- **Steiner-Filter**: Gibt es in Qdrant-Payload ein zuverlässiges Feld wie `author="Rudolf Steiner"` oder `source_id`-Prefix? Falls ja, wird Step 1.2 deutlich präziser.

Antwort: Nein.

- **Modellwahl**: Step 1.1/1.2 eher Reasoner (streng) vs Chat (stilsauber)? (Plan empfiehlt: 1.1 Reasoner, 1.2 Reasoner, 1.3 Chat.)

Ja!

- **Chunk-Länge**: Settings-Option