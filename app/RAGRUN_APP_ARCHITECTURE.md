RAGRUN – Architekturübersicht des `app`-Verzeichnisses
======================================================

Ziel dieses Dokuments ist eine belastbare Bestandsaufnahme der aktuellen RAG-Implementierung sowie ein Zielbild für ein Refactoring, das Ingestion und Retrieval klar trennt und die Rollen von LangChain (Orchestrierung), LangGraph (Agents/State Machines) und LangFuse (Telemetry/Tracing) hervorhebt.


Aktueller Stand (Dez 2025)
--------------------------
- Entry Point: `app/main.py` startet FastAPI, registriert zwei Router (`/api/v1/rag` für Ingestion/Deletion, `/api/v1/agent/philo-von-freisinn` für Retrieval/Concept-Explain) und bündelt Health-Checks (Qdrant, Embedding-Service, LangFuse).
- Konfiguration: `app/config.py` (Pydantic Settings, Prefx `RAGRUN_`) liefert URLs/Keys für Qdrant, Postgres, Embedding-Service, LangFuse, DeepSeek.
- Infrastruktur:
  - Qdrant: `infra/qdrant_client.py` minimaler HTTP-Wrapper für Ensure/Upsert/Delete/Search/Scroll/Retrieve.
  - Embeddings: `infra/embedding_client.py` ruft den personal embedding service batchweise auf.
  - LLM: `infra/deepseek_client.py` (reiner Chat-Call).
  - Persistenz: `db/tables.py` + `db/session.py` definieren ein relationales Mirror-Table (`rag_chunks`) und einen SQLAlchemy-Engine-Fabrikator.
  - Mirror: `ingestion/repositories/mirror_repository.py` schreibt/liest Chunk-Metadaten in das Mirror-Table (Upsert/Delete/List per Source).
  - Telemetrie: `core/telemetry.py` sendet best-effort Ingestion-Metriken an LangFuse (falls konfiguriert).
- Ingestion-Pfad (API `POST /api/v1/rag/upload-chunks`):
  - `api/rag.py` parst JSONL zu `ChunkRecord` (aus `ragrun.models`), validiert und ruft `IngestionService`.
  - `IngestionService` orchestriert: Dedupe → Klassifikation (unchanged/changed/new) via bestehender Payloads (Qdrant-Retrieve) → Embedding der geänderten/neuen Chunks → `ensure_collection` + Upsert in Qdrant → Payload-Update für unveränderte → Mirror-Upsert in Postgres → optionaler Cleanup alter Chunk-IDs pro Source → optional LangFuse-Metrik.
  - Sparse/BM25: Qdrant benötigt einen Text-Index auf dem Payload-Feld `text`. Die Ingestion stellt den Index nun automatisch sicher. Für bereits vorhandene Collections muss der Index einmalig erzeugt werden (z. B. `POST /collections/{collection}/index` mit `{"field_name":"text","field_schema":{"type":"text"}}`) oder per Re-Ingestion.
  - Löschpfad `POST /api/v1/rag/delete-chunks` holt Chunk-IDs aus dem Mirror und löscht sie in Qdrant + Mirror.
  - Listing `GET /api/v1/rag/books/titles` liest ausschließlich aus dem Mirror (SQL).
- Retrieval/Concept-Explain (API `POST /api/v1/agent/philo-von-freisinn/retrieval/concept-explain`):
  - `api/concept_explain.py` baut Service-Layer via Lazy-Singletons.
  - `ConceptExplainService`: embed des Begriffs → Vektor-Suche in Qdrant (`k` Treffer) → optionale Summary-Expansion (Scroll nach `source_id`, nur für summary-chunk-types) → Prompt-Bau → Aufruf DeepSeek → Rückgabe mit Primär- und Expanded-Treffern. Kein Standard-RAG-Fallback implementiert; Chat-Endpunkt liefert 501 für generische Prompts.
- Beobachtungen/Gaps:
  - Ingestion- und Retrieval-Logik liegen nun unter `ingestion/` bzw. `retrieval/`, teilen Infrastruktur über `core/providers`, aber die Domänendienste sind noch nicht vollständig entkoppelt.
  - LangChain/LangGraph werden derzeit gar nicht genutzt; Orchestrierung liegt in manuell geschriebenen Services.
  - LangFuse ist für Ingestion und Retrieval über best-effort Hooks angebunden; end-to-end Tracing/Spans fehlen.
  - Dependency Injection / Lifecycle-Management der Clients erfolgt jetzt zentral via `core/providers`, aber es gibt noch keine konfigurierbaren Scopes (Request/Graph).
  - Gemeinsame Domänenmodelle (Chunk, Source, Query) sind über `shared/models.py` re-exportiert; Payload-Dicts tauchen dennoch an mehreren Stellen auf.


Zielbild: klare Trennung & explizite Orchestrierung
---------------------------------------------------
Prinzipien:
- Trenne Ingestion (Daten rein) strikt von Retrieval/Serving (Antworten raus). Verzeichnisnamen: ingestion und retrieval
- Hebe Infrastruktur-Schichten (Clients, DB, Settings) aus den Domänen heraus; teile sie nur über zentrale Provider. Verzecihnisname: provider
- Nutze LangChain für Chains/Tools und LangGraph für zustandsbehaftete Abläufe (z. B. Ingestion-Pipeline mit Validation → Embedding → Upsert → Cleanup; Retrieval-Pipeline mit Suche → Re-Rank → Prompt → LLM).
- Integriere LangFuse als Querschnitt (Tracing/Telemetry) für beide Pfade.

Vorgeschlagene Struktur (Top-Level `app/`)
- `app/main.py` – FastAPI-Setup, Router-Mounting, Health.
- `app/config.py` – Settings/Secrets.
- `app/core/`
  - `logging.py`, `telemetry.py` (LangFuse), `providers.py` (Factory/DI für Clients), gemeinsame Fehlerklassen.
- `app/infra/`
  - `qdrant_client.py`, `embedding_client.py`, `llm/` (DeepSeek/HF/OpenAI), `db/` (Engine, Tables), evtl. `cache/`, `storage/`.
- `app/shared/`
  - Domänenmodelle (Chunk, Source, Query, RetrievalResult), Schemas, DTOs, Utilities (hashing, id-mapping), Prompt-Bausteine, Settings-Konstanten.
- `app/ingestion/`
  - `api/` – Endpoints für Upload/Delete/Inventory.
  - `services/` – Use-Cases (UploadChunksService, DeleteChunksService) mit klaren Ports auf Infra.
  - `pipelines/` (LangChain) – modulare Steps (validate → embed → upsert → mirror → cleanup).
  - `graphs/` (LangGraph) – optionale zustandsbehaftete Abläufe/Retry/Backpressure.
  - `repositories/` – Mirror/Metadata-Zugriff.
  - `telemetry/` – Ingestion-spezifische Metriken/Events.
- `app/retrieval/`
  - `api/` – Retrieval/Chat/Tools/Agents Endpoints.
  - `services/` – Query-Service, Reranker, Answer Synthesizer (z. B. ConceptExplainService).
  - `chains/` (LangChain) – Retrieval-Augmented Chains (search → filter → rerank → prompt).
  - `graphs/` (LangGraph) – Agent- oder Multi-Step-Flows (z. B. Decision: concept vs. generic; multi-hop Retrieval).
  - `prompts/` – System/User Prompt-Templates, kontextualisiert nach Agent.
  - `telemetry/` – Trace/Span-Hooks für LangFuse.
- `app/api/` (Root)
  - Montagepunkte, die auf die domänenspezifischen APIs verweisen; nur dünne Router, keine Logik.

Abhängigkeitsrichtung (gewünscht)
```
api -> services/use-cases -> chains/graphs -> infra clients
shared + core werden nur nach unten importiert, nie zyklisch.
```

Integration der Kern-Bausteine
- LangChain: Bietet die wiederverwendbaren Chains (Embeddings-Chain, Qdrant-Search-Chain, Prompt-Chain, Rerank-Chain). Diese Chains leben in `ingestion/pipelines` bzw. `retrieval/chains` und kapseln Tools/Prompts.
- LangGraph: Steuert komplexere Abläufe als Graph/State Machine (z. B. Ingestion mit Retry/Partial Failure; Retrieval mit Branching zwischen Concept-Explain und Standard-RAG). Graphs orchestrieren Chains und Services.
- LangFuse: Ein zentrales Telemetry-Layer (`core/telemetry`) stellt Span/Trace-Hooks, die in Chains/Graphs eingehängt werden (Ingestion + Retrieval). Healthz bleibt best-effort.
- Infrastruktur-Clients: Qdrant/Embeddings/LLM werden über `core/providers` oder Factories injiziert; so sind Mocks für Tests und spätere Provider-Switches (z. B. andere Vektor-DB) möglich.

Migrationsleitplanken (kurz)
- Schrittweise Extraktion: zuerst Services in `ingestion/` und `retrieval/` verschieben, dann Chains/Graphs ergänzen.
- API-Stabilität: Endpunkt-Pfade und Request/Response-Schemas beibehalten, interne Implementierung austauschen.
- Telemetrie ausbauen: LangFuse-Hooks in beiden Pfaden; Healthz ergänzt um Graph/Chain-Status falls sinnvoll.
- Tests: Ports/Adapters erlauben Unit-Tests der Chains/Graphs ohne Netzwerke; End-to-End-Tests gegen lokale Qdrant/Embeddings.
