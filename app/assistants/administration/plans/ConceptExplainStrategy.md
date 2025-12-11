## Ziel
- Neues Kommando `rag:augment:concepts:explain` generiert Konzept-Erklärungen pro Buch auf Basis `books/<BOOK>/results/augmentation/concepts.jsonl`, angereichert mit Primär-/Sekundärbüchern und vorhandenen Augmentierungen (summaries, concepts etc.) des Assistenten `philo-von-freisinn`. Parameter: bookDir.
- Zwei Pfade:
  - Spezieller Endpoint nur für `rag:augment:concepts:explain` (deterministisch, ohne Intent-Gate).
  - Allgemeiner Assistant-Endpoint mit Intent-Gate, das bei Ein-Begriff-Prompts automatisch den Konzept-Branch zieht.
- Ergebnis ist eine über `rag:upload` ladbare JSONL-Datei unter `ragkeep/books/<bookDir>/results/rag-chunks/concepts-chunks.jsonl`.

## Inputs und Datenquellen
- `ragkeep/books/<bookDir>/results/augmentation/concepts.jsonl` je Buch: enthält Konzepte, die mit Erklärungen ergänzt werden sollen.
- `primary-books` / `secondary-books` aus Manifest des Assistenten (Option: --assistant): steuern bevorzugte Quellen/Filter im Retrieval.
- Augmentierungen in Qdrant: `chapter_summary`, `talk_summary`, `essay_summary`, `explanation_summary`, `concepts`, ggf. Quotes.
- Collection: `philo-von-freisinn` (Manifest-gesteuert), k=10 für initiale Treffer.

## Neues CLI/Endpoint
- CLI: `rag:augment:concepts:explain <book-id>` (Wrapper ruft spezialisierten Endpoint) und schreibt die Resultate nach `ragkeep/books/<bookDir>/results/rag-chunks/concepts-chunks.jsonl`.
- Spezial-Endpoint (ragrun): POST `/agent/philo-von-freisinn/retrieval/concept-explain` mit Payload `{ book_id, concept, trace? }`; setzt den Branch deterministisch auf „concept-explain“, streamt Antwort + Quellen.
- Allgemeiner Assistant-Endpoint: nutzt Intent-Gate (siehe unten) und wählt „concept-explain“, wenn Ein-Begriff-Prompt erkannt wird; sonst Standard-RAG-Branch.

## Branch-Auswahl (Intent-Gate) – nur im allgemeinen Assistant-Endpoint
- Triggert, wenn
  - Prompt ≤ ~8 Token und keine Mehrfach-Entitäten, oder
  - Frageform mit genau einem benannten Begriff („Was ist die Seele?“, „was bedeutet Ichheit?“).
- Heuristiken:
  - NER/Keyword-Extractor → 1 Kernbegriff.
  - Länge/Stopwords-Check (keine Liste mehrerer Items, kein „und/oder“).
  - Fallback: Wenn mehrere Begriffe → Standard-RAG-Branch.

## LangGraph-Skizze (Retrieval-Branch „concept-explain“)
- Supervisor Node: wählt Branch „concept-explain“ bei obigem Gate; sonst Standard.
- Nodes:
  1) **Parse/Normalize Concept**: Kleinbuchstaben, Lemma, Varianten (z. B. „menschliche Seele“ → „Seele“ + „menschliche Seele“).
  2) **Retriever (k=10)**: Qdrant-Suche über gesamte Collection (books + aug); Filter aus Manifest (collection, evtl. worldview). Scores speichern.
  3) **Summary-Expander**: Für Treffer mit `chunk_type` in `{chapter_summary,talk_summary,essay_summary,explanation_summary}` → lade alle Chunks desselben `source_id`/Kapitel (`chapter_id`/`talk_id`). Label als „Zusätzliche Information“.
  4) **Context Assembler**: 
     - Primäre Sektion: Top-10 Treffer (sort by score, dedupe per Quelle).
     - Sekundäre Sektion: Expanded Chapter/Talk Chunks (geordnet).
     - Kurzer Fußabdruck (<3k Tokens) durch Trimmen nach Score/Länge.
  5) **DeepSeek Chat Call**: Prompt unten; Output-Limit 300 Tokens.
  6) **Trace/Return**: Rückgabe mit verwendeten Quellen + Scores; Logging zu LangFuse.

## Prompt für DeepSeek Chat (Konzept-Erklärung)
```
System:
Du bist „Philo-von-Freisinn“ (s. prompts/instruction.md & prompts/chat-instructions.md):
- Philosophischer Assistent, Fokus auf individuelle Freiheit, logisch präzise.
- Ton: klar, zugänglich, lebendig und humorvoll, aber ohne Ironie oder Personalisierungen.
- Keine Fremdworte, die Steiner nicht nutzt (z. B. kein „Determinismus“).
Aufgabe: Erkläre einen Begriff im Kontext der Sammlung Rudolf Steiner (Primär- und Sekundärliteratur plus Augmentierungen wie summaries, concepts).

User:
Erkläre den Begriff: "<CONCEPT>".

Context (RAG):
- Primäre Treffer: Top-10 gemischte Fundstellen (Buchauszüge, Konzepte, Summaries).
- Zusätzliche Information: Alle Chunks der Kapitel/Vorträge, zu denen Summary-Treffer gehören.

Anweisungen:
- Schreibe für eine 16-jährige Leserin.
- Maximal ca. 300 Tokens.
- Erkläre so, dass die Bedeutung im Philo-von-Freisinn-Kontext klar wird.
- Nutze nur den gelieferten Kontext; erfinde nichts.
- Keine Zitate, sondern erklärend zusammenfassen.
```

## Verarbeitung pro Buch (rag:augment:concepts:explain)
- Lade `concepts.jsonl`.
- Für jeden Eintrag:
  - `concept` normalisieren, optional Varianten aus dem Datensatz nutzen.
  - Rufe LangGraph-Branch „concept-explain“ mit `concept` auf.
  - Speichere Erklärung zurück in dieselbe JSONL (Feld `explanation`) oder neues Feld `explanation_de`.
  - Optional: Confidence/Score dokumentieren.
- Parallelisierung nach Buch erlaubt; Rate-Limit berücksichtigen.

## Output-Format `concepts-chunks.jsonl` (ChunkMetadata)
- Alle Begriffe eines Buchs gehören zu einer Begriffe-Liste (`begriff_list`).
- Pflichtfelder pro Chunk:
  - `source_id`: `<id begriff_list>` (identifiziert die Liste für das Buch)
  - `chunk_type`: `begriff_list`
  - `source_title`: `Begriffe von <Name des Assistenten>`
  - `source_index`: Index des Begriffs in der Liste (0-based)
  - `segment_title`: `<Begriff>`
  - `parent_id`: `<book_id>`
  - `text`: erzeugte Erklärung (ca. 300 Tokens)
- Ablagepfad: `ragkeep/books/<bookDir>/results/rag-chunks/concepts-chunks.jsonl` (rag:upload-fähig).

## Test-/Validierungsplan
- Unit: Intent-Gate (Ein-Begriff vs. Multi-Begriff) mit Beispielen.
- Integration: Anfrage „Seele“, „Was ist die Menschliche Seele?“ → Branch „concept-explain“, k=10 Treffer, Summary-Expansion ausgelöst, Antwort ≤300 Tokens.
- Regression: Multi-Begriff („Seele und Geist“) fällt korrekt auf Standard-RAG zurück.

## Offene Punkte / Risiken
- Token-Budget: Chapter-Expansion kann groß werden – Heuristik zum Kürzen nötig (Score-basierte Kürzung, Priorität Summaries > Book-Chunks).
- Mehrsprachige Varianten: Lemmatizer/Normalizer muss Umlaute/ß korrekt behandeln.
- Caching: Wiederholte Erklärungen desselben Begriffs sollten aus Cache/Persistenz bedient werden.
