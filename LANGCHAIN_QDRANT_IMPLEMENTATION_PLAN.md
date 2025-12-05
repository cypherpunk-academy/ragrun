# LangChain + Qdrant Implementation Plan for `ragrun`

This document translates `LANGCHAIN_QDRANT_ARCHITECTURE.md` into an executable, multi-step delivery plan. Each phase lists the concrete objectives, key tasks, dependencies, validation criteria, and relevant architecture references.

## Phase 0 – Foundations & Environments
- **Objectives**: ensure developers can run LangChain + Qdrant locally and in staging; lock in baseline dependencies.
- **Key tasks**
  1. Provision Docker Compose stack (FastAPI, Qdrant ≥1.10, Postgres, LangFuse, personal-embedding-service) – see Architecture §§B, H.
  2. Define `.env` templates covering `QDRANT_URL`, `LANGFUSE_*`, DeepSeek + fallback LLM API keys (§B, §H).
  3. Document make/CLI commands for bringing the stack up/down and running smoke tests.
- **Dependencies**: none – unblock all other phases.
- **Validation**: `docker compose up` succeeds; health endpoints for Qdrant, LangFuse, embedding service respond.

## Phase 1 – Metadata & Data Model Enforcement
- **Objectives**: mirror `CHUNK_METADATA_MODEL.md` in code and tests so ingestion rejects invalid chunks.
- **Key tasks**
  1. Implement shared Pydantic (or dataclass) model covering `author` through `language` + `tags` (§C, metadata table).
  2. Add JSON Schema export for client validation (feeds `@ragChunk` / `@ragUpload`).
  3. Create fixture-driven tests covering required fields, enums, defaults, parent-child checks (§C).
- **Dependencies**: Phase 0 (env + Postgres) for running tests with DB integration.
- **Validation**: running the test suite fails on malformed payloads and passes on canonical fixtures.

## Phase 2 – Personal Embedding Service Integration
- **Objectives**: harden the existing FastAPI embedding service and expose configuration hooks for future models (§B, Future Thoughts table).
- **Key tasks**
  1. Containerize `personal-embedding-service` with the local `T-Systems-…` model, GPU batching, rate limiting.
  2. Expose `embedding_model` config knob (per-request override in prep for multi-collection models).
  3. Add observability hooks (timings → LangFuse ingestion dataset §H).
- **Dependencies**: Phase 0 (env). Optional dependency on Phase 8 to wire future rerankers.
- **Validation**: `/embeddings` returns deterministic vectors; metrics arrive in LangFuse.

## Phase 3 – Ingestion Pipeline (`/rag/upload`, `/rag/delete`) ✅ COMPLETE
- **Objectives**: implement the hot path from JSONL chunks (produced by `@ragChunk`) to Qdrant + Postgres mirrored data (§D, §C).
- **Key tasks**
  1. ✅ Build FastAPI routers for upload/delete with schema validation + dedupe logic.
  2. ✅ Batch requests to the embedding service (size 64), handle retries/backoff.
  3. ✅ Implement Qdrant client wrapper (collection auto-create, HNSW params §B) + Postgres mirror writes.
  4. ✅ Emit LangFuse ingestion traces after each batch (§H).
- **Dependencies**: Phases 0–2.
- **Validation**: ✅ end-to-end test: run `ragrun ingest` against a sample manifest → verify points in Qdrant (`collections/…/points/count`) and Postgres rows.

**Implementation status**
- ✅ `docker-compose.yml` provisions Postgres, Qdrant, the personal-embeddings-service, and the ragrun API. Copy `env.example` to `.env`, then start the stack with `docker compose up -d`.
- ✅ `python -m ragrun ingest --collection books --file examples/sample_chunks.jsonl --api http://localhost:8000` uploads JSONL chunks via `/rag/upload`, batching requests automatically.
- ✅ Verification steps:
  1. `curl http://localhost:6333/collections/books | jq '.result.points_count'` → returns the number of Qdrant points.
  2. `docker compose exec postgres psql -U ragrun -d ragrun -c "select chunk_id, collection from rag_chunks;"` → mirrors show up in Postgres.
  3. `curl -X DELETE http://localhost:8000/rag/delete -H "Content-Type: application/json" -d '{"collection": "books", "chunk_ids": ["chunk-001"]}'` → deletes from both Qdrant and Postgres.
- ✅ LangFuse telemetry is emitted when `RAGRUN_LANGFUSE_*` env vars are set; otherwise the hooks no-op so ingestion still succeeds locally.
- ✅ All 17 tests pass (`python -m pytest tests/`), including unit tests for `IngestionService`, `ChunkMirrorRepository`, `/rag/upload`, `/rag/delete`, and the CLI.
- **Known requirements**: Postgres `rag_chunks` table must be initialized before first use. Run: `docker compose exec postgres psql -U ragrun -d ragrun < app/db/schema.sql` (or create manually).

## Phase 4 – Retrieval APIs (`/rag/query`, `/rag/query-advanced`)
- **Objectives**: expose LangChain-powered retrievers backed by Qdrant and optional BM25 (§D, §G).
- **Key tasks**
  1. Implement query embedding flow with caching + telemetry.
  2. Wire dense search (top-k) and optional rerank node placeholder.
  3. Build hybrid dense+BM25 pipeline (dense α + sparse 1−α) with α request parameter (§Future Thoughts, row 3).
  4. Format responses with citations (chunk metadata) + token usage stats.
- **Dependencies**: Phase 3 (data), Phase 2 (embeddings).
- **Validation**: integration tests that seed collections, call `/rag/query`, `/rag/query-advanced`, and assert deterministic ordering.

## Phase 5 – CLI Tooling
- **Objectives**: give ops/dev teams automation for ingestion, analytics, monitoring (§E).
- **Key tasks**
  1. Build Typer/Click CLI with subcommands: `ingest`, `analytics`, `monitor`, `agent define`, `health`.
  2. Integrate CLI auth flow with API keys.
  3. Provide human-friendly output (tables, charts) and JSON export options.
- **Dependencies**: Phases 3–4 for API endpoints.
- **Validation**: run CLI against local stack; add smoke tests in CI (invoke CLI with mocked HTTP).

## Phase 6 – LangGraph Agents & Branching
- **Objectives**: realize the `philo-von-freisinn` agent (and friends) with LangGraph StateGraphs (§F, §G).
- **Key tasks**
  1. Define YAML manifest schema + Postgres storage for agents.
  2. Implement loader that builds LangGraph graphs with `messages_state`, `MemorySaver`, and the defined tools.
  3. Build branch nodes: supervisor classifier, book branch, typology branch, reflection branch; plug in retriever tools (books vs. concepts collections).
  4. Implement admin agent toolkit (reindex, sync manifest).
- **Dependencies**: Phase 4 (retrieval tools), Phase 5 (CLI for `agent define`).
- **Validation**: scenario tests driving `/agent/philo-von-freisinn` with scripted prompts and asserting LangFuse branch traces.

## Phase 7 – Observability & LangFuse Instrumentation
- **Objectives**: ensure every ingestion/retrieval/agent request is traceable (§H).
- **Key tasks**
  1. Add LangFuse callback handlers to LangChain chains, retrievers, embedding client.
  2. Emit structured datasets: `ingestion_runs`, `query_evals`, `agent_paths`.
  3. Build dashboards / SQL for SLA metrics (latency, error rate, embedding throughput).
  4. Create export scripts for evaluation sets (100 reference queries, LLM-as-judge scoring).
- **Dependencies**: Phases 3–6 for touchpoints.
- **Validation**: Observability smoke test verifying traces appear end-to-end for each major workflow.

## Phase 8 – Deployment, Security & Ops
- **Objectives**: harden for production and provide rollout guidance (§B, §E, §H).
- **Key tasks**
  1. Author IaC or Railway configs for FastAPI + Qdrant + LangFuse; set up snapshots/backups for Qdrant volumes.
  2. Implement API-key auth + per-device rate limiting (shared middleware for REST + agent endpoints).
  3. Add CI/CD pipeline (lint, tests, deploy) with environment promotion steps.
  4. Produce runbook (incident response, scaling guidelines, API quota policies).
- **Dependencies**: completion of functional phases (0–7).
- **Validation**: staging deployment passes load test (50k chunk ingest, <50 ms median query per §D) and security checklist.

## Phase 9 – Future Enhancements (Tracked Improvements)
- **Objectives**: capture the “Future thoughts” table as backlog items for iterative upgrades (§J).
- **Key tasks**
  1. Per-collection embedding model config + manifest field (`embedding_model`).
  2. Concrete reranker integration (Cohere, bge, or Jina) reachable via the embedding service or dedicated microservice.
  3. Default hybrid search (dense + BM25) with tunable α surfaced in `/rag/query-advanced`.
  4. Parent-child recursive retrieval tool leveraging `parent_id` metadata.
  5. Multi-LLM routing (DeepSeek primary, Groq/Claude fallbacks) with latency-aware policy.
- **Dependencies**: baseline system live.
- **Validation**: each enhancement ships behind feature flags and has regression tests.
