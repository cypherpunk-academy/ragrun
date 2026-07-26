# Implementierungsplan V2: Assistenten-Chat – Zwei-Endpoint-Architektur

> Paradigmenwechsel gegenüber `ASSITANTS_CHAT_PLAN.md`: Der Client erhält den
> mit Qdrant-Daten befüllten Prompt **vor** dem LLM-Aufruf und kann ihn
> anzeigen, anpassen oder direkt ausführen lassen.
> Fundgrube für Intents und Chunk-Typen: `ASSITANTS_CHAT_PLAN.md`,
> `ASSISTANT_CHAT_INTENT_DESIGN.md`.

---

## 1. Übersicht: Warum dieses „Frontend“ für ein LLM?

### 1.1 Sinnhaftigkeit der Zwei-Phasen-Architektur

| Aspekt | Bewertung | Begründung |
|--------|-----------|------------|
| **Transparenz** | ⭐⭐⭐ | Der Nutzer sieht, welche Quellen und welcher Kontext dem LLM vorgelegt werden. Kein „Black Box“-Gefühl. |
| **Kontrolle** | ⭐⭐⭐ | Nutzer kann den Prompt anpassen, bevor Tokens verbraucht werden – z.B. irrelevante Chunks entfernen oder die Frage präzisieren. |
| **Vertrauen** | ⭐⭐ | Gerade bei philosophischen/anthroposophischen Themen: Nachvollziehbarkeit der Quellenbasis ist wertvoll. |
| **Kosten** | ⭐⭐ | Prompt-Änderung vor dem LLM-Call spart unnötige Token-Verbräuche bei „falschen“ Abfragen. |
| **UX-Komplexität** | ⭐ | Zwei Schritte (generate → prüfen → execute) können für routinierte Nutzer als Overhead wirken. |

**Fazit:** Das Modell lohnt sich besonders für:

- Nutzer, die auf Quellengenauigkeit achten (Forscher, Studierende)
- Anwendungen, in denen falsche oder aus dem Kontext extrapolierte Antworten problematisch sind
- Debugging und Qualitätssicherung der RAG-Pipeline

**Risiko:** Bei rein konversationellen Nutzern könnte die zusätzliche Bestätigungsschleife stören. Daher: **Beide Modi anbieten** – „Schnellantwort“ (direkt execute) und „Mit Prüfung“ (generate → prüfen → execute).

---

## 2. Architektur: Zwei Endpoints

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Client (Frontend)                                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│  1. User schreibt Prompt + wählt Aktion (Dropdown / erste Frage)             │
│  2. POST /generate-prompt  →  befüllter Prompt + Metadaten                    │
│  3. Client zeigt Prompt an: „So wird die Anfrage beantwortet. Ändern?“       │
│  4. User entscheidet: direkt ausführen ODER Prompt anpassen und dann         │
│  5. POST /execute-prompt   →  LLM-Antwort (Streaming)                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Endpoint 1: `generate-prompt`

**Request:**
```json
{
  "assistant_slug": "philo-von-freisinn",
  "action_id": "general-question",
  "user_prompt": "Was bedeutet der Begriff Ich bei Steiner?",
  "thread_id": "optional-for-context",
  "language": "de-DE"
}
```

**Response:**
```json
{
  "prompt_id": "uuid",
  "action_id": "general-question",
  "filled_prompt": "System: [instruction.prompt des Assistenten] + [prompt.prompt befüllt]\n\nUser: Was bedeutet...",
  "query_results": {
    "primary-books": { "chunk_count": 8, "chunk_ids": ["..."], "preview": "..." },
    "secondary-books": { "chunk_count": 8, ... },
    "concepts": { "chunk_count": 5, ... }
  },
  "citations_metadata": [...],
  "estimated_tokens": 4200,
  "expires_at": "ISO8601"
}
```

### Endpoint 2: `execute-prompt`

**Request:**
```json
{
  "prompt_id": "uuid-from-generate",
  "modified_prompt": "optional – wenn User den Prompt geändert hat",
  "stream": true
}
```

**Response:** SSE-Stream (wie bisher) mit Token-Events und abschließendem `done`-Event inkl. Citations.

### LangGraph-Integration

- **generate-prompt** = Subgraph bis Retrieval: Intent/Klassifikation (falls nötig), Queries ausführen, Embedding-Vektoren cachen, Prompt befüllen → Response mit `prompt_id` und gespeichertem State.
- **execute-prompt** = Rest des Graphen: `compose_answer`, `attach_citations`, ggf. `finalize`. Nutzt gecachten State (inkl. Embedding-Vektoren), sodass keine erneute Qdrant-Anfrage nötig ist.

---

## 3. Aktionen (Prompttypen)

### 3.1 Kern-Aktionen (aus User-Vorgabe)

| Aktion | ID | Kurzbeschreibung | UI-Label |
|--------|-----|------------------|----------|
| General Question | `general-question` | Broad explanation, comparison, deepening | General Question |
| Find Quote | `find-quote` | Quote or citation for a topic | Find Quote |
| Locate in Works | `locate-in-works` | Where does a topic appear? (Book, lecture, series) | Locate in Works |
| Dialectic Dialogue | `dialectic-dialogue` | Dialectical engagement, check counter-thesis | Dialectic Dialogue |
| Clarify Concept | `clarify-concept` | Definition of a single concept | Clarify Concept |
| Thanks or Feedback | `thanks-feedback` | No retrieval; reinforcement feedback on answer quality | Thanks or Feedback |

### 3.2 Empfohlene Reihenfolge im Dropdown

1. General Question *(Default for first question)*  
2. Clarify Concept  
3. Find Quote  
4. Locate in Works  
5. Dialectic Dialogue  
6. Thanks or Feedback  

---

## 4. Dateistruktur: Ordner pro Aktion

**System-Prompt pro Assistent:** Jeder Assistent hat seinen eigenen System-Prompt in
`ragkeep/assistants/<assistant_slug>/prompts/instruction.prompt`. Dieser wird für alle
Aktionen verwendet. Beispiel: `ragkeep/assistants/philo-von-freisinn/prompts/instruction.prompt`.

Der `prompt.prompt` in jedem Aktionsordner enthält **nur** den action-spezifischen Teil:
Kontext-Slots (`{primary-books}` usw.), Konversationskontext und Nutzeranfrage. Der
assistant-eigene System-Prompt wird separat geladen und dem LLM vorangestellt.

```
app/retrieval/actions/
├── general-question/
│   ├── action-manifest.yaml
│   └── prompt.prompt
├── clarify-concept/
│   ├── action-manifest.yaml
│   └── prompt.prompt
├── find-quote/
│   ├── action-manifest.yaml
│   └── prompt.prompt
├── locate-in-works/
│   ├── action-manifest.yaml
│   └── prompt.prompt
├── dialectic-dialogue/
│   ├── action-manifest.yaml
│   └── prompt.prompt
└── thanks-feedback/
    ├── action-manifest.yaml
    └── prompt.prompt
```

### Sonderfall: thanks-feedback (Reinforcement)

`thanks-feedback` hat **kein Retrieval** – die Nutzeräußerung (Dank, Lob, Kritik) ist das Signal.  
**Reinforcement-Flow:** Nach dem Senden triggert das Backend im Hintergrund eine **zweite LLM-Anfrage**:

- **Input:** Kontext der letzten Antwort (User-Frage, AI-Antwort, Nutzer-Feedback wie „Danke, hat geholfen!“)
- **Output:** strukturierte Bewertung (z.B. helpfulness, accuracy, clarity als Scores oder Kategorien)
- **Zweck:** Reinforcement-Daten für Qualitätsverbesserung, Fine-Tuning, A/B-Tests oder Metriken

Das Ergebnis wird nicht an den Client zurückgegeben, sondern gespeichert (event_metadata, separater reinforcement-Tabelle o.ä.).

**Beispiel action-manifest.yaml für thanks-feedback:**

```yaml
id: thanks-feedback
label: Dank oder Bewertung
requires_retrieval: false
reinforcement_evaluation:
  prompt_ref: evaluate_feedback.prompt
  output_schema: feedback_scores   # z.B. { helpfulness: 1-5, accuracy: 1-5, sentiment: positive|neutral|negative }
```

---

## 5. action-manifest.yaml: Schema und Attribute

### 5.1 Vollständiges Schema

```yaml
# action-manifest.yaml – vollständiges Schema

id: general-question
label: Allgemeine Frage
description: Breite Erklärung, Vergleich, Vertiefung zu einem Thema.

# Ob Retrieval überhaupt nötig ist (false z.B. bei thanks-feedback)
requires_retrieval: true

# Nur bei thanks-feedback: Hintergrund-LLM zur Bewertung der vorherigen Antwort (Reinforcement)
reinforcement_evaluation: null   # oder: { prompt_ref: "evaluate_feedback.prompt", output_schema: "feedback_scores" }

# Qdrant-Anfragen: jede hat einen Namen, der im Prompt als {name} erscheint
queries:
  - name: primary-books
    chunk_types: [book]
    k: 8
    method: hybrid          # hybrid | dense | sparse
    # optionale Filter (werden an payload_filter übergeben)
    author: null           # z.B. "Rudolf Steiner"
    worldview: null        # z.B. "Rationalismus" (oder leer für allgemein)

  - name: secondary-books
    chunk_types: [secondary_book, chapter_summary]
    k: 8
    method: dense

  - name: concepts
    chunk_types: [begriff_list]
    k: 5
    method: sparse

# Query-spezifische Erweiterungen (optional)
query_options:
  # Für find-quote: parallele Suche quote + book (wie hybrid_retrieve_quote_parallel)
  parallel_branches: null   # oder z.B. [{quote}, {book, secondary_book}]
  # Query-Vorverarbeitung
  query_extract: null      # lemma | quote_text | raw (default)

# Prompt-Placeholders: welche query-Ergebnisse im Template vorkommen
prompt_placeholders:
  - primary-books
  - secondary-books
  - concepts

# Folge-Aktionen (siehe Abschnitt 7)
follow_ups:
  - type: detail
    question: "Soll ich detaillierter antworten?"
    condition: null        # optional: z.B. "answer_length < 200"
  - type: summary
    question: "Das Gespräch scheint rund. Soll ich eine Zusammenfassung machen?"
    condition: "turn_count >= 3"
```

### 5.2 Ergänzende Query-Attribute (Vorschläge)

| Attribut | Typ | Beschreibung | Beispiel |
|----------|-----|--------------|----------|
| `k` | int | Anzahl Treffer | 8 |
| `chunk_types` | list[str] | Filter auf CHUNK_TYPE_ENUM | [book, talk] |
| `method` | enum | hybrid, dense, sparse | hybrid |
| `author` | str \| null | Payload-Filter author | "Rudolf Steiner" |
| `worldview` | str \| null | Payload-Filter worldviews | "Rationalismus" |
| `min_score` | float \| null | Nur Treffer ab Score | 0.5 |
| `query_source` | str | Woher kommt die Query? | user_prompt \| extracted_lemma \| extracted_quote |
| `deduplicate_with` | list[str] | RRF-Fusion mit anderen Queries | [quotes] |
| `rerank` | bool | Nach Retrieval reranken | false |

---

## 6. Übersicht: Queries je action-manifest

| Aktion | Queries (name → chunk_types, k, method) |
|--------|----------------------------------------|
| **general-question** | primary-books: [book], 8, hybrid; secondary-books: [secondary_book, chapter_summary], 8, dense; concepts: [begriff_list], 5, sparse |
| **clarify-concept** | lemma-lookup: [begriff_list], 1, exact (Lemma-Gleichheit); fallback-books: [book, explanation], 8, dense |
| **find-quote** | quotes: [quote], 5, hybrid; books: [book, secondary_book] + author Steiner, 5, hybrid (parallel, RRF-fusioniert) |
| **locate-in-works** | works: [chapter_summary, talk, talk_summary], 8, hybrid |
| **dialectic-dialogue** | thesis: [book, talk], 6, hybrid; counter: [book, essay], 6, dense |
| **thanks-feedback** | *(keine Queries)* Nach dem execute: Hintergrund-LLM-Anfrage zur Bewertung der vorherigen Antwort (Reinforcement-Feedback). |

---

## 7. follow-ups: Philosophische Überlegungen

### 7.1 Konzept

Das LLM gibt nach seiner Antwort nicht nur Text aus, sondern wählt optional aus einer **Liste von Folge-Aktionen**, die sinnvoll erscheinen. Der Client zeigt diese als Buttons/Quick-Replies.

Beispiel-Output des LLM (Structured Output oder Nachverarbeitung):

```json
{
  "response": "...",
  "suggested_follow_ups": ["detail", "summary"]
}
```

### 7.2 Definition im action-manifest

```yaml
follow_ups:
  - type: detail
    question: "Soll ich detaillierter antworten?"
    # condition: optional, wird vom Backend ausgewertet
    condition: null

  - type: summary
    question: "Das Gespräch scheint rund. Soll ich eine Zusammenfassung machen?"
    condition: "turn_count >= 3"

  - type: source_deepen
    question: "Soll ich eine der genannten Quellen vertiefen?"
    condition: "citation_count >= 2"

  - type: compare
    question: "Möchtest du einen Vergleich mit einem anderen Konzept?"
    condition: null

  - type: quote_request
    question: "Soll ich ein passendes Zitat dazu suchen?"
    condition: "action_id == clarify-concept || action_id == general-question"
```

### 7.3 Wie wählt das LLM?

**Option A: LLM wählt explizit**  
Das Prompt enthält am Ende:

```
Nach deiner Antwort: Wähle aus den folgenden Folge-Aktionen die sinnvollen aus (JSON-Array).
Verfügbare Aktionen: {follow_ups_json}
Antworte mit: {"suggested_follow_ups": ["detail", "source_deepen"]}
```

**Option B: Heuristiken im Backend**  
Das Backend wertet `condition` aus (z.B. `turn_count >= 3`) und filtert die Liste. Das LLM erhält nur die kontextuell sinnvollen Optionen und kann sie noch einmal filtern.

**Option C: Hybrid**  
Backend filtert grob (z.B. „summary“ nur nach ≥3 Turns), LLM wählt fein („summary“ nur wenn das Gespräch thematisch abgeschlossen wirkt).

### 7.4 Sinnvolle follow-up-Typen (gesamt)

| type | question | Typischer Kontext |
|------|----------|-------------------|
| `detail` | "Soll ich detaillierter antworten?" | Kurze Antwort, Nutzer könnte mehr wollen |
| `summary` | "Soll ich eine Zusammenfassung machen?" | Langes Gespräch, viele Turns |
| `source_deepen` | "Soll ich eine Quelle vertiefen?" | Mehrere Zitate genannt |
| `compare` | "Möchtest du einen Vergleich?" | Einzelnes Konzept erklärt |
| `quote_request` | "Soll ich ein Zitat dazu suchen?" | Begriff/Konzept erklärt, aber ohne Zitat |
| `related` | "Soll ich verwandte Themen finden?" | Thema abgehandelt |
| `restart` | "Mit neuem Thema starten?" | Nach Dank/Meta oder Abschluss |

---

## 8. Beispiel: action-manifest und prompt für general-question

### action-manifest.yaml

```yaml
id: general-question
label: Allgemeine Frage
description: Breite Erklärung, Vergleich, Vertiefung zu einem Thema.

requires_retrieval: true

queries:
  - name: primary-books
    chunk_types: [book]
    k: 8
    method: hybrid

  - name: secondary-books
    chunk_types: [secondary_book, chapter_summary]
    k: 8
    method: dense

  - name: concepts
    chunk_types: [begriff_list]
    k: 5
    method: sparse

follow_ups:
  - type: detail
    question: "Soll ich detaillierter antworten?"
  - type: quote_request
    question: "Soll ich ein passendes Zitat dazu suchen?"
  - type: summary
    question: "Das Gespräch scheint rund. Soll ich eine Zusammenfassung machen?"
    condition: "turn_count >= 3"
```

### prompt.prompt

Der System-Prompt kommt aus `ragkeep/assistants/<assistant_slug>/prompts/instruction.prompt`
(z.B. Philo von Freisinn). Der `prompt.prompt` enthält nur den **action-spezifischen**
Teil – Kontext-Slots und Nutzeranfrage:

```
Nutze ausschließlich die folgenden Quellen. Zitiere sie wo angebracht.

--- Primäre Buchquellen ---
{primary-books}

--- Sekundäre Werke und Kapitelzusammenfassungen ---
{secondary-books}

--- Begriffslexikon ---
{concepts}

---

Konversationskontext (falls vorhanden):
{conversation_context}

Nutzeranfrage: {user_prompt}

Antworte prägnant und quellengestützt.
```

**Zusammensetzung für das LLM:** `[instruction.prompt]` + `[prompt.prompt befüllt]` → System-Nachricht; `[user_prompt]` → User-Nachricht.

---

## 9. Beispiel: find-quote (mit parallel branches)

### action-manifest.yaml

```yaml
id: find-quote
label: Zitat suchen
description: Zitat oder Belegstelle zu einem Thema finden.

requires_retrieval: true

queries:
  - name: quotes
    chunk_types: [quote]
    k: 5
    method: hybrid
    # Spezieller Modus: parallel zu books, dann RRF-Fusion
    parallel_with: steiner-books

  - name: steiner-books
    chunk_types: [book, secondary_book]
    k: 5
    method: hybrid
    author: "Rudolf Steiner"
    parallel_with: quotes

follow_ups:
  - type: detail
    question: "Soll ich ein Zitat genauer erklären?"
  - type: source_deepen
    question: "Soll ich den Kontext eines Zitats vertiefen?"
```

*(Hinweis: `parallel_with` erfordert Backend-Logik, die beide Queries parallel ausführt und fusioniert – analog zu `hybrid_retrieve_quote_parallel`.)*

---

## 10. Implementierungs-Roadmap (Vorschlag)

| Phase | Inhalt |
|-------|--------|
| **P0** | Endpoints `generate-prompt` und `execute-prompt` anlegen |
| **P0** | action-manifest Loader + Query-Executor für eine Aktion (z.B. `general-question`) |
| **P1** | Alle Aktionen mit manifest + prompt anlegen |
| **P1** | Client-API: prompt_id als Session-Objekt, optional modified_prompt |
| **P2** | follow_ups: Backend filtert nach condition, LLM wählt aus |
| **P2** | UI: Dropdown Aktionen, Preview des befüllten Prompts, „Schnellantwort“ vs. „Mit Prüfung“ |
| **P3** | Caching von generate-prompt (expires_at), Token-Schätzung |

---

## 11. Entscheidungen (vormals Offene Punkte)

- [x] **Embedding-Vector cachen:** Ja. `generate-prompt` speichert die vorberechneten Embedding-Vektoren (pro Query) mit dem prompt_id. `execute-prompt` nutzt diese, sodass keine erneute Embedding-Berechnung nötig ist.
- [x] **modified_prompt:** Kompletter Ersatz. Der Client überschreibt den befüllten Prompt vollständig, kein Diff.
- [x] **Aktionen pro Assistant:** Die Standard-Definition (action-manifest pro Aktion) gilt als Default. Im `assistant-manifest.yaml` können Aktionen selektiv überschrieben werden (z.B. andere Queries, anderes prompt-Template).
- [x] **LangGraph-Integration:** Ja. `generate-prompt` = Subgraph bis Retrieval (inkl. Query-Ergebnisse, befüllter Prompt). `execute-prompt` = Rest: compose_answer, attach_citations, ggf. finalize.
