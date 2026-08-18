# Intent-Classifier: Architektur und Erweiterung

## Überblick

Der Intent-Classifier ist die erste Stufe im RAG-Graph (`assistant_chat_graph.py`). Er entscheidet, welchen Verarbeitungspfad eine Nutzeranfrage nimmt — ob Quellen gesucht oder direkt geantwortet wird. Begriffsdefinitionen laufen über `erklaerung`, nicht über einen eigenen Intent.

```
User-Message
    │
    ▼
classify_intent          ← DeepSeek (JSON-Mode), kein Streaming
    │
    ├─ "skip"            → finalize (direkte Antwort ohne Retrieval)
    └─ "erklaerung" / "quelle_suchen" / "sonstiges"
                         → route_retrieval_plan → retrieve_chunks → compose_answer → attach_citations → finalize
```

## Prompt

**Datei:** `app/retrieval/prompts/intent_classify.prompt`

Der Prompt definiert 4 Intents und gibt dem LLM den Gesprächskontext (letzte Nutzerfragen) mit. Das LLM antwortet als JSON mit 3 Feldern:

| Feld | Typ | Beschreibung |
|------|-----|-------------|
| `intent` | string | Eines der 4 Labels |
| `confidence` | float | 0.0–1.0 |
| `reasoning` | string | 1–2 Sätze Begründung |

### Aktuelle Intents

| Intent | Zweck | Routing-Ziel |
|--------|-------|-------------|
| `quelle_suchen` | Zitat oder Belegstelle finden | `route_retrieval_plan` → hybrides Quote-Retrieval |
| `erklaerung` | Thema erklären, definieren, vertiefen, vergleichen, auflisten | `route_retrieval_plan` → Standard-Retrieval |
| `skip` | Grüße, Dank, Meta-Fragen, Off-Topic | `finalize` (keine Quellensuche) |
| `sonstiges` | Fallback | `route_retrieval_plan` → breites Retrieval ohne chunk_type-Filter |

## Beteiligte Funktionen

### 1. `classify_intent` (Node)

**Zeile ~260–341** in `assistant_chat_graph.py`

- Lädt den Prompt aus `intent_classify.prompt`
- Injiziert die letzten 3 User-Nachrichten als `{conversation_context}`
- Ruft DeepSeek im JSON-Mode auf (`with_structured_output(IntentResult)`)
- Gibt `intent` und `intent_confidence` als State zurück
- Kosten werden über `UsageRecorder` getrackt

### 2. `route_after_intent` (Routing-Funktion)

Entscheidet basierend auf `state["intent"]`:
- `skip` → `"finalize"` (überspringe Retrieval)
- alles andere → `"route_retrieval_plan"` (Standard-RAG)

### 3. `route_retrieval_plan` (Node)

Setzt den `retrieval_plan` basierend auf dem Intent via `INTENT_CHUNK_TYPE_MAP`:

```python
INTENT_CHUNK_TYPE_MAP = {
    "quelle_suchen":      ["quote", "book", "talk"],
    "erklaerung":         ["book", "talk", "chapter_summary", "secondary_book", "typology"],
    "sonstiges":          [],   # kein Filter → alle chunk_types
}
```

### 4. `retrieve_chunks` (Node)

**Zeile ~526–570+**

Hybrid-Retrieval (Dense + Sparse) aus Qdrant. Bei `quelle_suchen` wird `hybrid_retrieve_quote_parallel` verwendet (optimiert für Zitate). Retry-Logik: bei `sufficiency == "insufficient"` bis zu 2 Wiederholungen mit breiteren Parametern.

## Konfigurationsdateien

| Datei | Zweck |
|-------|-------|
| `app/retrieval/prompts/intent_classify.prompt` | Prompt-Template für den Classifier |
| `app/retrieval/graphs/intents.py` | Intent-Labels, Chunk-Type-Map, Skip-Set |
| `app/retrieval/graphs/assistant_chat_graph.py` | Graph-Definition, Nodes, Routing |

## Erweiterungsmöglichkeiten

### A. Neue Intents hinzufügen

In `intents.py` sind bereits gestrichene Intents dokumentiert, die bei Bedarf aktiviert werden können:

| Möglicher Intent | Aktuell gemappt auf | Nutzen |
|-------------------|---------------------|--------|
| `werk_lokalisieren` | `erklaerung` | Gezieltes Retrieval nach GA-Nummer/Werk |
| `zitat_suchen` | `quelle_suchen` | Differenzierung: wörtliches Zitat vs. Belegstelle |
| `vergleich` | `erklaerung` | Spezial-Prompt für Gegenüberstellungen |
| `zusammenfassung` | `erklaerung` | Optimiert auf `chapter_summary`-Chunks |
| `follow_up` | `sonstiges` | Rückbezug auf vorherige Antwort (braucht erweiterten Kontext) |

**Schritte für einen neuen Intent:**

1. Label in `INTENT_LABELS` (intents.py) hinzufügen
2. Chunk-Type-Mapping in `INTENT_CHUNK_TYPE_MAP` definieren
3. Beispiele im Prompt (`intent_classify.prompt`) ergänzen
4. Optional: eigene Routing-Logik in `route_after_intent` und eigenen Node
5. Optional: eigene Retrieval-Strategie in `retrieve_chunks`

### B. Intent-spezifische Retrieval-Strategien

Aktuell unterscheidet `retrieve_chunks` nur zwischen `quelle_suchen` (Quote-Parallel-Retrieval) und allem anderen (Standard-Hybrid). Mögliche Erweiterungen:

- **`zusammenfassung`**: Nur `chapter_summary`-Chunks, höheres k für breitere Abdeckung
- **`vergleich`**: Zwei getrennte Retrieval-Durchläufe (einen pro Vergleichsgegenstand)
- **`werk_lokalisieren`**: Metadata-Filter auf `source_id` statt Embedding-Suche

### C. Confidence-basiertes Routing

Aktuell wird `confidence` nur geloggt. Mögliche Nutzung:

```python
# Beispiel: bei niedriger Confidence auf breites Retrieval zurückfallen
def route_after_intent(state):
    intent = state.get("intent", "sonstiges")
    confidence = state.get("intent_confidence", 0.0)
    if confidence < 0.6:
        return "route_retrieval_plan"  # breites Retrieval statt Spezial-Branch
    ...
```

### D. Multi-Intent / Compound Queries

Aktuell: genau 1 Intent pro Nachricht. Erweiterung auf mehrere Intents (z.B. "Erkläre den Begriff Ich und finde ein Zitat dazu") würde erfordern:

- Prompt anpassen: Liste von Intents statt einem
- Routing: parallele Branches oder sequenzielle Verarbeitung
- Merge-Node: Ergebnisse zusammenführen

### E. Kontext-abhängige Intent-Erkennung

Der Classifier sieht aktuell die letzten 3 User-Nachrichten als Kontext. Erweiterungen:

- **Assistant-Antworten mitgeben**: Bessere Follow-up-Erkennung
- **Arbeitstext-Status**: "Hat verknüpftes Dokument" als Signal für schreibbezogene Intents
- **Aktive Quelle**: Wenn der User in einem Buch liest, könnte das die Retrieval-Strategie beeinflussen

### F. Evaluation und Monitoring

Aktuell gibt es kein systematisches Tracking der Intent-Qualität. Mögliche Maßnahmen:

- Intent + Confidence in der Datenbank loggen (pro Talk)
- Fehlklassifikationen sammeln (User korrigiert sich, stellt gleiche Frage anders)
- Confusion Matrix aus Chat-Logs erstellen
- A/B-Test zwischen Prompt-Varianten
