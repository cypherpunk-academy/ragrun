# Phase 3 Implementation Notes

## Overview
Phase 3 implements the core ingestion pipeline for ragrun, providing REST endpoints for uploading and deleting chunks, with data stored in both Qdrant (vectors) and Postgres (metadata mirror).

## Key Implementation Details

### UUID Conversion for Qdrant Point IDs
Qdrant requires point IDs to be either UUIDs or unsigned integers. Since our chunk metadata uses string IDs (e.g., `demo-chunk-001`), we convert them to deterministic UUIDs using UUID v5:

```python
from uuid import uuid5, NAMESPACE_DNS

point_uuid = uuid5(NAMESPACE_DNS, chunk.metadata.chunk_id)
```

This ensures:
- The same chunk_id always maps to the same UUID (idempotency)
- Qdrant accepts the IDs
- The original chunk_id is preserved in the payload and Postgres mirror

### Database Initialization
Before first use, initialize the Postgres schema:

```bash
docker compose exec postgres psql -U ragrun -d ragrun < app/db/schema.sql
```

Or manually via `psql` / `init_db.py`.

### Testing
All tests pass (`python -m pytest tests/`):
- Unit tests for `IngestionService` (upload, delete, deduplication)
- Integration tests for `ChunkMirrorRepository` (SQLite-based)
- API tests for `/rag/upload` and `/rag/delete` endpoints
- CLI tests for `python -m ragrun ingest`

### End-to-End Verification
1. Start the stack: `docker compose up -d`
2. Ingest sample data: `python -m ragrun ingest --collection demo-books --file examples/sample_chunks.jsonl --api http://localhost:8000`
3. Verify Qdrant: `curl http://localhost:6333/collections/demo-books | jq '.result.points_count'`
4. Verify Postgres: `docker compose exec postgres psql -U ragrun -d ragrun -c "SELECT collection, chunk_id FROM rag_chunks;"`
5. Delete a chunk: `curl -X DELETE http://localhost:8000/rag/delete -H "Content-Type: application/json" -d '{"collection": "demo-books", "chunk_ids": ["demo-chunk-001"]}'`
6. Re-verify both stores to confirm deletion

### LangFuse Telemetry
When `RAGRUN_LANGFUSE_*` environment variables are set, ingestion traces are sent to LangFuse with:
- `ingestion_id` (trace ID)
- Collection name
- Count of chunks ingested
- Duplicate count
- Duration (milliseconds)
- Embedding model and vector size

If LangFuse is not configured, telemetry is silently skipped and ingestion proceeds normally.

## Architecture Decisions

### Why Postgres Mirror?
While Qdrant stores vectors and payload metadata, the Postgres mirror enables:
- SQL-based analytics and reporting
- Fast filtering on metadata fields (worldview, source_id, etc.)
- Backup/restore workflows
- Cross-collection queries

### Deduplication Strategy
Chunks are deduplicated based on `(chunk_id, content_hash)` tuple before embedding. This prevents:
- Unnecessary embedding API calls for duplicate content
- Redundant Qdrant upserts
- Wasted vector storage

### Batch Size
Default batch size is 64 chunks per embedding request (configurable via API). This balances:
- Throughput (fewer HTTP round-trips)
- Memory usage (embedding service processes batches in-memory)
- Timeout constraints (30s default per batch)

## Known Limitations
- No automatic retry logic for transient Qdrant/Postgres failures (caller must retry)
- Telemetry is best-effort (failures don't block ingestion)
- No support for partial batch success (all-or-nothing upsert)

## Next Steps (Phase 4)
- Implement retrieval APIs (`/rag/query`, `/rag/query-advanced`)
- Add LangChain retrievers backed by Qdrant
- Optional: BM25 hybrid search
- Query result reranking and citation formatting
