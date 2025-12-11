# LangGraph Utilization Strategy for `ragkeep` Assistants

## Zielbild
- Assistenten wie `philo-von-freisinn` werden als LangGraph-Agents ausgeführt, die Qdrant-Kollektionen (Bücher + Augmentierungen) nutzen und ihre charakter-spezifischen Prompts aus `ragkeep/assistants/*` erhalten.
- Eine einheitliche Graph-Basis stellt Retrieval-, Augmentierungs- und Admin-Tools bereit; pro Assistent werden nur Manifest-gebundene Parameter (Sammlung, Filter, Prompting) injiziert.
- LangGraph-Assistants-API (Threads/Runs) + Postgres-Checkpointer liefern zustandsvolle Dialoge, Streaming und Auditierbarkeit; LangFuse zeichnet Pfade und Tool-Aufrufe auf.

## Ausgangslage
- Manifeste definieren Sammlung (`rag-collection`), Bücherlisten, Augmentation-Typen (summaries, concepts, quotes), Schreibstil und Beschreibung.
- Prompt-Dateien unter `prompts/` geben den Charakter des jeweiligen Assistenten vor.
- Chunks (Bücher + Augmentierungen) liegen in Qdrant; LangChain-Pfade für Upload/Retrieval sind bereits angelegt (s. `LANGCHAIN_QDRANT_ARCHITECTURE.md` §E–G).
- LangGraph ist vorgesehen für Agents/Branching, aber noch nicht mit den `ragkeep`-Manifesten verdrahtet.

## LangGraph-Assistant-Funktionalität nutzen
- **Assistant-Definition**: Aus jedem Manifest wird ein LangGraph-Assistant (Name, Beschreibung, System-Prompt, Tools, Defaults). Mapping:
  - `rag-collection` → Default-Filter der Retriever-Tools (`collection`, `chunk_type`, ggf. `worldview`).
  - `primary-books` / `books` → Metadaten-Filter + Tooling für „narrow retrieval“; optional Priorisierung via `importance`-Boost.
  - `augmentation-types` → Aktivierte Zweigintegration (Summaries/Concepts/Quotes Retrieval).
  - `writing-style` / `description` → System-Prompt + Response Rewriter Node.
- **Threads & Runs**: LangGraph Assistants API nutzt Postgres-Checkpointer für `messages_state`; Threads entsprechen User-Sessions, Runs repräsentieren Graph-Exekutionen. Aktiviert mensch-in-der-Schleife (Pause/Resume) und Streaming.
- **Tools**:
  - `qdrant_retriever_books`: Filter `chunk_type=book`, optional `source_id`-Whitelist aus Manifest.
  - `qdrant_retriever_aug`: Filter `chunk_type in augmentation-types`.
  - `augment_context`: fasst book + summary + concept Treffer zusammen.
  - `rerank` (später) und `self_critique`/`fallback_search` als separate Nodes.
  - Admin-Tools (reindex, sync manifest) nur für Operator-Assistenten.
- **Branching** (Supervisor → Subgraphs):
  - `book_query` → embed → retrieve books → optional rerank → LLM antwortet.
  - `concept/typology` → retrieve concepts/quotes → augment_context → LLM vergleicht/erklärt.
  - `maintenance` → Admin-Tooling (keine LLM-Antwort nötig).
- **Prompting**: Charakter-Prompts aus `prompts/` als System-Node; optional „style rewriter“ Node nach der Antwort, um Schreibstil sicherzustellen.
- **Observability**: LangFuse Callback in jedem Node; Assistant/Thread IDs als Trace-Attribute. Speichert Branch-Entscheidungen und Tool-Inputs.

## Umsetzungsplan (konkret)
1) **Schemas angleichen**: Manifest-Schema für Agents in `ragrun` definieren (YAML) und 1:1 auf `assistants/*/assistant-manifest.yaml` mappen (inkl. `augmentation-types`).
2) **Loader bauen**: Service, der Manifest liest, Prompts lädt, Tools parametrisiert und daraus einen LangGraph-StateGraph erzeugt (`MessagesState`, `MemorySaver`/Postgres).
3) **Base Graph implementieren**: Supervisor + Branches (book, concept/typology, maintenance) mit klaren Tool-Nodes; Defaults aus Manifest injizieren (Sammlung, Filter, Stil).
4) **Assistants-API verdrahten**: FastAPI-Endpunkt `/agent/{assistant}` nutzt LangGraph Assistants API (Threads/Runs) für Streaming und Checkpointing.
5) **Tests**: Szenario-Tests für einen Assistenten (z. B. `philo-von-freisinn`) mit Skript-Dialogen; Asserts auf Branch-Trace, verwendete Tools, Zitationskontext.
6) **Observability**: LangFuse Events pro Node/Branch; Metrics-Dashboards für Retrieval-Score, Branch-Häufigkeiten, Tool-Latenzen.
7) **Rollout**: Basisgraph live schalten, anschließend weitere Assistenten durch Manifest-Only Rollout onboarden.

## Risiken / offene Punkte
- Reranking fehlt noch; Retrieval-Qualität hängt aktuell nur von dichten Vektoren ab.
- Manifest-Felder für Weltanschauungen/Chapters (z. B. `sophia-von-einklang`) benötigen Mapping auf Filterlogik – muss im Loader geklärt werden.
- Prompt-Versionierung: Änderungen in `prompts/` sollten mit Assistant-Version (z. B. `last-updated`) geloggt werden, sonst schwierige Reproduzierbarkeit.
- Speicherstrategie: Postgres-Checkpointer benötigt Migrationspfad/Retention (Thread TTL).
