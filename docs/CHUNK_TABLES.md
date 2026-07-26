# Chunk tables (Postgres)

- **`rag_chunks`** – Primary store (DB-first). Rows are keyed by `(rag_partition, chunk_id)`:
  - **`rag_partition = __shared__`** – Shared corpus body text: `chunk_type` `book` or `secondary_book` only (raw books / lectures as produced by chunking).
  - **`rag_partition = <assistant rag-collection>`** – Assistant-owned rows (summaries, quotes, talks, concepts, typologies, etc.).
  Payloads arrive via `POST /api/v1/rag/store-chunks`. `embedded_at` is set after a successful `POST /api/v1/rag/embed-chunks` for the **Qdrant** collection (assistant name); that embed run unions the assistant partition with a **whitelist** of `source_id` values for `__shared__` (from `assistant-manifest.yaml`, sent as `shared_source_ids`).
- **`vector_chunks`** – Mirror of Qdrant payloads for SQL/analytics. Still keyed by `(collection, chunk_id)` where `collection` is the **Qdrant** collection name (not `rag_partition`). Unchanged by the shared-corpus split.

Alembic: `0015` renames the legacy mirror to `vector_chunks`, `0016` creates `rag_chunks`, `0018` renames `collection` → `rag_partition` and moves book rows to `__shared__`.
