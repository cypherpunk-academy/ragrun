## Plan: Simplified RAG Upload Endpoint and Chunk Deletion by Metadata (aligned with current code)

Context
- Architecture reference: `plans/ARCHITECTURE.md` (RAG API groups, LocalVectorStoreManager, Personal Embeddings Service)
- Chunk metadata reference for filters: `ragprep/src/types/ragSchemas.ts` → `ChunkOutput.metadata`
  - Supported keys to filter by (initial pass): `chunk_id`, `chunk_index`, `book_id`, `author`, `book_index`, `book_title`, `book_subtitle`, `chapter_level_1`, `chapter_level_2`, `chapter_level_3`, `paragraph_numbers`, `paragraph_page`, `content_length`, `created_at`

Goals
1) Add an endpoint to delete chunks by metadata filter, with a force `--all` option.
2) Add a simplified upload endpoint that ingests pre-chunked JSONL (chunks.jsonl) into the local vector DB (Chroma) using the personal embeddings service.
3) Provide minimal CLI wrappers to call these endpoints.

---

Step 0 — Endpoint structure refactor (modular routers)

Rationale
- Improve maintainability and discoverability by splitting monolithic `rag.py` into modular, per-endpoint files grouped by route prefix.

Directory structure
- Already implemented under `app/api/endpoints/rag/` with one file per endpoint. Current modules include:
  - `app/api/endpoints/rag/__init__.py` — aggregates sub-routers
  - `app/api/endpoints/rag/retrieve.py` → `POST /rag/retrieve`
  - `app/api/endpoints/rag/search.py` → `POST /rag/search`
  - `app/api/endpoints/rag/upload.py` → `POST /rag/upload`
  - `app/api/endpoints/rag/collections.py` → `GET /rag/collections`
  - `app/api/endpoints/rag/health.py` → `GET /rag/health`
  - `app/api/endpoints/rag/delete_chunks.py` → `POST /rag/delete-chunks`
  - `app/api/endpoints/rag/upload_chunks.py` → `POST /rag/upload-chunks`

Module pattern
- Each file defines and exports `router = APIRouter()` with its route(s) and Pydantic models local to that endpoint.
- `app/api/endpoints/rag/__init__.py` aggregates:
  - `from fastapi import APIRouter`
  - `router = APIRouter(prefix="/rag")`
  - `from .query import router as query_router` (and so on)
  - `router.include_router(query_router)` for each submodule

Registration
- API v1 aggregator already imports the group router via `from app.api.endpoints.rag import router as rag_router`.
- OpenAPI entries are present for all routes listed above.

Migration methodology
- Move existing endpoint functions and Pydantic schemas from `rag.py` into corresponding modules.
- Extract any shared request/response models into `app/api/endpoints/rag/schemas.py` if needed.
- Keep route paths unchanged to preserve compatibility.

Verification
- Run server and confirm all endpoints respond as before.
- Check OpenAPI docs render with the same operations and tags.

---

Step 0.5 — Decommission Pinecone and remove "local_" naming (personal-only stack)

Rationale
- We standardize on the personal embeddings service + ChromaDB. Remove Pinecone support and the `local_` route/file naming. This reduces cognitive load and configuration surface.

Configuration changes
- Remove `VECTOR_DB_TYPE` branches and Pinecone fallback:
  - In `app/db/local_vector_db.py`, delete the conditional that imports/instantiates Pinecone. Always instantiate `LocalVectorStoreManager` as the singleton `vector_db`.
  - Remove `app/db/vector_db.py` and any Pinecone-specific setup code, if present.
  - Prune related settings/environment variables from `app/core/config.py` and `.env.example` (index names, API keys, environment flags that only affect Pinecone).
- Embeddings: enforce exclusive use of the personal embeddings microservice across code paths. Remove/disable any in-process embedding fallbacks where present.

API route and filename renames
- Unified `/rag/*` endpoints are already in place; no `/rag/local/*` aliases are exposed.
- Files under `app/api/endpoints/rag/` match the unified route names (`upload.py`, `search.py`, `collections.py`, `health.py`, `delete_chunks.py`, `upload_chunks.py`, `retrieve.py`).

Docs and references
- Update `plans/ARCHITECTURE.md` and any docs to remove Pinecone mentions as active path; keep a short migration note.
- Update examples and cURL snippets to the new `/rag/*` paths.

Tests
- Delete/adjust Pinecone path tests.
- Update endpoint path assertions and integration tests to new routes.

Acceptance
- Primary path uses Chroma via `LocalVectorStoreManager`; server runs without Pinecone env vars.
- All `local_` prefixes removed from URLs and filenames.
- OpenAPI shows unified `/rag/*` endpoints.

---

Step 1 — Delete chunks by metadata (with --all)

API design
- Route: `POST /api/v1/rag/delete-chunks` (implemented in `app/api/endpoints/rag/delete_chunks.py`).
- Auth: same as other RAG endpoints (JWT bearer or `X-API-Key`).
- Request body:
  - `filter?: Dict[str, Any]`
  - `all?: boolean` (default `false`)
  - `collection_name?: string` (default `"philosophical_768"`)
  - `dry_run?: boolean` (default `false`)
  - `limit?: number`
- Response body:
  - `{ deleted_count: number, collection: string, preview_only: boolean, note?: string }`

Implementation details
- `LocalVectorStoreManager` (`app/db/local_vector_db.py`) implements:
  - `count_by_filter(filter) -> int` (approximate), `delete_by_filter(filter) -> { deleted_count }`, `delete_all() -> { deleted_count }`, and `_convert_filter_to_where`.
- Endpoint validates `all`/`filter`, supports `dry_run` preview via `count_by_filter`/collection stats, and enforces `limit` when provided.

Filter semantics
- Accept Pinecone-style operators per current converter: `$in`, `$eq`, and plain equality.
- Example requests:
  - Delete by `book_id`:
    ```json
    { "filter": { "book_id": "Rudolf_Steiner#Die_Philosophie_der_Freiheit#4" } }
    ```
  - Delete by multiple chapters:
    ```json
    { "filter": { "chapter_level_1": { "$in": ["Vorrede", "I. Erkenntnistheorie"] } } }
    ```
  - Delete all:
    ```json
    { "all": true }
    ```

Testing
- Unit: 
  - `_convert_filter_to_where` coverage for equality and `$in`.
  - `delete_by_filter` removes only matching entries; `delete_all` clears collection.
- Integration:
  - Seed a temp collection, delete by `book_id`, assert remaining counts.
  - Dry-run returns preview and does not change counts.

Docs
- Add OpenAPI descriptions and cURL examples.
- Update `docs/LOCAL_VECTOR_DB_SETUP.md` with deletion examples and safety notes.

---

Step 2 — Simplified upload endpoint for pre-chunked JSONL

Ingests lines shaped like `ChunkOutput` from `ragSchemas.ts`:
```json
{ "text": "...", "metadata": { "chunk_id": "...", "book_id": "...", "author": "...", "chapter_level_1": "...", "paragraph_numbers": [1,2], "content_length": 1534, "created_at": "2024-08-15T12:34:56Z" } }
```

API design
- Route: `POST /api/v1/rag/upload-chunks` (implemented in `app/api/endpoints/rag/upload_chunks.py`).
- Request body:
  - One of:
    - `chunks_jsonl_path: string` (server-readable path), or
    - `chunks_jsonl_content: string` (full JSONL as text)
  - `collection_name?: string`
  - `batch_size?: number` (default 64)
- Response:
  - `{ success: boolean, total_lines: number, processed: number, upserted: number, skipped: number, errors: string[], collection: string, processing_time_ms: number }`

Implementation details
- Robust JSONL parsing with blank/comment-line skipping; line-level error collection.
- For each chunk: use `metadata.chunk_id` as vector `id` when present; embed `text` in batches via personal embeddings service; upsert vectors through `LocalVectorStoreManager.upsert_vectors` with `{ ...metadata, text }`.
- Logging includes counts and previews; continues on batch errors; summarizes failures.

Testing
- Integration: ingest a small `chunks.jsonl`, assert `count` increases and vector search finds known snippets.
- Error-path: malformed line is reported, remaining lines still processed.

Docs
- Add cURL examples (file path and inline content variants).
- Note recommended `batch_size` based on embedding service throughput.

---

Step 3 — Minimal CLI wrappers (present)

The legacy Python `click` CLI (`scripts/rag-cli/rag_cli.py`) was removed. Use the shell wrapper or direct API:
  - Delete chunks: `POST /rag/delete-chunks` with `filter` or `all`
  - Upload chunks: `POST /rag/upload-chunks` with `chunks_jsonl_path` or `chunks_jsonl_content`

---

Security, safety, and observability
- Auth: reuse JWT/API-Key with optional RBAC (`rag:write` for deletion/upload).
- Rate limiting: apply same slowapi policies as other write endpoints.
- Safety:
  - `--all` requires `all=true`; consider optional `confirm: "DELETE"` in future.
  - `limit` guard for large, accidental deletes (non-`all`).
- Observability:
  - Structured logs include request id, user id (if available), `collection_name`, and filter summary.
  - Health endpoint `/rag/health` exposes collection stats via store APIs.

---

Acceptance checklist
- LocalVectorStoreManager supports `count_by_filter`, `delete_by_filter`, and `delete_all`.
- Endpoint `/rag/delete-chunks` implemented with `all`, `filter`, `dry_run`, `limit`.
- Upload endpoint `/rag/upload-chunks` accepts JSONL (path or inline) and upserts.
- OpenAPI docs list both endpoints; README shows them under API routes.


