# LangChain/LangGraph Architektur

Ziel: Jede Anfrage wird als Graph/Chain mit stabilen IDs, Request-Trace und nachvollziehbaren Entscheidungen ausgeführt. Ergebnisse, Context, LLM-Calls und Branches werden in der DB protokolliert. Struktur bleibt übersichtlich auch bei vielen Chains/Graphs (eine Datei pro Graph/Chain, klarer Ordnerbaum).

## Ordnerstruktur (vorgeschlagen)
- `app/core/` – Provider, Telemetrie (LangFuse), Logging-Hooks, ID-Generatoren.
- `app/infra/` – Adapters (Qdrant, Embeddings, LLMs, DB, Cache).
- `app/shared/` – Domänenmodelle, DTOs, Prompt-Bausteine, Utility-Funktionen (IDs/Hashing).
- `app/ingestion/` – Pipelines/Graphs für Upload/Delete, Repos (Mirror), API.
- `app/retrieval/` – Chains/Graphs/Prompts/Services für Agents & Tools.
  - `chains/` → eine Datei pro Chain (reine LangChain-Runnable/Tool-Graph-Knoten)
  - `graphs/` → eine Datei pro LangGraph-Graph (Steuerlogik, Branches, Retries)
  - `prompts/` → Prompt-Templates je Agent/Weltanschauung
  - `services/` → Use-Cases, Logging, Provider-Bridging
  - `telemetry/` → LangFuse/DB Hooks für Spans und Events

## IDs, Tracing, Persistenz
- `request_id`: vom API-Layer generiert (z.B. UUIDv4), pro HTTP-Request.
- `graph_id`: pro Graph-Instanziierung (UUIDv7) – verbindet alle Spans/Nodes einer LangGraph-Ausführung.
- `run_id`/`node_run_id`: pro Chain/Tool/LLM-Call (UUIDv7) – für feingranulare Nachvollziehbarkeit.
- Persistenz in DB (bestehend `retrieval_events`/`retrieval_chunks`) um Spalten erweitern: `request_id`, `graph_id`, `run_id`, `node`, `branch`, `llm_model`, `prompt_checksum`, `context_checksum`.
- Telemetrie: LangFuse-Spans (optional) mit denselben IDs; falls LangFuse down → best-effort noop.
- Logging-Hooks: zentrale Callback-Handler, die LangChain/LangGraph Events konsumieren und in Postgres schreiben (ähnlich `RetrievalLoggingRepository`).

## Beispiel-Graph: Drei Perspektiven + Konsistenzcheck + Bewertung
**Use-Case:** Begriff wird in drei Perspektiven beantwortet (neutral, Mathematismus, Individualismus), dann Konsistenz geprüft, ggf. Retry, danach Bewertung.

### Nodes (eine Datei `graphs/concept_perspectives.py`)
1. `fetch_context_neutral` (Retriever-Chain) – filtert auf {concept} ohne Weltanschauung.
2. `fetch_context_mathematismus` (Retriever-Chain) – Qdrant-Filter `worldviews` contains "Mathematismus".
3. `fetch_context_individualismus` (Retriever-Chain) – Filter `worldviews` contains "Individualismus".
4. `llm_prompt1|2|3` (LLM-Chain pro Prompt) – nutzt jeweilige Prompt-Template; parallel ausführbar (async Graph, langgraph.async) für schnellere Multi-Perspektiven.
5. `compare_results` (Hybrid: LLM-Judge + Embedding-Similarity) – Cosine-Similarity (z.B. LangChain SemanticSimilarity) + LLM-Score; Schwelle z.B. sim < 0.7 oder LLM-Score < 7 → Divergenz loggen (\"Prompt 2 weicht in Polarität ab\") und Retry markieren.
6. `maybe_retry` (Decision) – falls Schwelle unterschritten: erneutes Retrieval+LLM mit erweitertem k oder anderem Filter; max 2 Retries.
7. `evaluate` (LLM-as-Judge) – bewertet Qualität/Kohärenz, erstellt Summary.
8. `persist` – schreibt Events/Chunks in DB mit IDs.

### Chains (eine Datei pro Chain unter `chains/`)
- `retrieval_chain.py`: kapselt Qdrant-Retriever (mit Filter) → optional Reranker; dynamisches k (Start k=5, bei Retry k=10 + ContextualCompressionRetriever).
- `llm_chain_perspective.py`: Prompt-Building + LLM-Call, annotiert mit `run_id` + Prompt-Checksum.
- `consistency_chain.py`: vergleicht drei Texte, gibt Score + Hinweise zurück.
- `evaluation_chain.py`: bewertet finale Antworten.

### IDs im Flow
- API erzeugt `request_id` und `graph_id`; legt sie in Graph-Context.
- Jeder Node erzeugt `run_id` und meldet Start/Ende an Logging-Hook (inkl. Prompt/Context Hash, Modellname, Dauer, Tokens).
- `persist` schreibt: `request_id`, `graph_id`, `run_id`, `node`, `branch`, `concept`, `retrieved_k`, `filters`, `model`, `score`, `answer_text`, `context_refs` (chunk_ids).

## Async, Batching, Security
- Asynchronität & Parallelität: LangGraph-Nodes sollten `async` sein; parallele LLM-Calls für unabhängige Perspektiven via `asyncio.gather` oder LangGraph-Concurrency. Concurrency-Limits (Semaphores) für LLM/Qdrant, Timeouts und Circuit-Breaker pro Node.
- Batching: Embeddings/Retrieval bereits batchbar; bei Multi-Concept-Jobs Batch pro Retriever-Node; für LLM nur vorsichtig (Prompt-Kontamination vermeiden). Sampling-/Top-k-Reduktion vor LLM-Aufruf, Context-Window-Checks.
- Security / Prompt Injection: Input-Validierung (Länge, verbotene Tokens), System-Prompt-Härtung (keine Tool-Aufrufe zulassen), Kontext-Whitelisting (nur geprüfte Chunks), Escape/neutralize user input in prompts, Defense-in-Depth: logging + anomaly detection auf LLM-Antworten, optional policy model oder guardrails (regex/LLM) pro Node.

## Antwort auf die Fragen
- **Sinnvoll?** Ja: Mehrperspektivisch + Konsistenzschritt + begrenzte Retries ist robust und transparent.
- **Context filtern?** Für weltanschauliche Prompts strikt nach Tag/Weltanschauung filtern (z.B. Qdrant-Payload `worldviews`). Neutraler Prompt kann breiter suchen.
- **Verbesserungen:**
  - Konsistenz-Score automatisieren (Embeddings oder LLM-Judge) und Retry nur bei Niedrig-Score triggern.
  - Prompt-Härtung: "Falls Kontext unzureichend, antworte 'Unzureichender Kontext'."; strukturierte Ausgabe (Abschnitte) für vergleichbare Ergebnisse.
  - Dynamisches k und Reranking (Contextual Compression Retriever) für kompakteren Kontext.
  - Mehr Telemetrie: Tokens, Kosten, Timing in DB + LangFuse; Dashboards pro Graph/Agent.
  - Tests: Unit-Tests pro Chain, Integrationstest pro Graph mit Mock-Qdrant/LLM.

## Testbarkeit (konkret)
- Unit-Tests pro Chain: Mock Qdrant/Embeddings/LLM, prüfen Filter, Prompt-Shape, ID-Anreicherung, Error-Handling.
- Graph-Tests: Simulierte Branches/Retries mit Fake LLM-Antworten und Scores; Snapshot-Tests für Persist-Datensätze (IDs, node, branch).
- Property-Tests: Konsistenz-Checker sollte bei permutierten Antworten stabil bleiben; Retry-Abbruch nach N Versuchen garantiert.
- Contract-Tests fürs Logging: Schema-validierung für `persist`-Payload (request_id/graph_id/run_id/node/branch/context_refs).
