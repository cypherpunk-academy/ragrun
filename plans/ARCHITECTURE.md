## Architecture Overview

This project is a production-ready Personal RAG Server built on FastAPI. It exposes a secure REST API for conversation threads, messages, assistant operations, and RAG workflows. It integrates a configurable vector database (local ChromaDB or Pinecone for rollback), a MongoDB for application data, external/local LLM providers, and a personal embeddings microservice.

- **API layer (FastAPI)**: Routers under `settings.API_V1_STR` (default `/api/v1`) provide endpoints for `auth`, `assistants`, `threads`, `messages`, `rag`, and `health`. Security headers, CORS, trusted hosts, request IDs, structured logging, and rate limiting are applied centrally.
- **Services layer**:
  - `RAGService`: chunk, embed, upsert, retrieve, and generate RAG responses.
  - `BookUploadService`: full book pipeline (metadata → chunking → embeddings → vector store) with progress and quality reporting.
  - `EmbeddingService` and `PersonalEmbeddingsService`: local in-process SentenceTransformers (for some paths) or asynchronous HTTP client for the personal embeddings microservice (`:8001`).
  - `LLM service factory`: abstracts OpenAI/DeepSeek selection via `settings.LLM_PROVIDER`.
  - `MetadataPipeline`: extraction/validation for rich, consistent chunk metadata.
- **Data layer**:
  - `MongoDB` (Motor): assistants, threads, messages, users, categories, tags.
  - `Vector DB`: Pinecone (rollback) or local ChromaDB (`LocalVectorStoreManager`) with a Pinecone-compatible API surface.
- **External components**: personal embeddings microservice, LLM providers (OpenAI or DeepSeek), optional Pinecone.
- **Configuration**: `app/core/config.py` controls security, CORS, vector DB type, LLM provider, embedding dimensions, environment flags, etc.

## Detailed Chapters

### Chunking

There are three chunking implementations used in different contexts:

- **`RAGService.add_document` (simple character window):**
  - Parameters: `chunk_size` (default 1000 chars), `chunk_overlap` (default 200 chars).
  - Algorithm: linear character slicing with overlap; minimal metadata. Suitable for quick ingestion via `/api/v1/rag/documents`.

- **`BookUploadService.TextChunker` (semantics-aware with metadata):**
  - Parameters from `UploadConfig`: `chunk_size`, `chunk_overlap`, `min_chunk_size`, `max_chunk_size`.
  - Breaks at word boundaries within the overlap window to reduce mid-word splits.
  - Two-pass: compute boundaries → attach rich `DocumentMetadata` per chunk, including `document_id`, `chunk_id`, indices, lengths, text preview, and per-chunk `page_number` via `MetadataExtractor`.
  - Used in `/api/v1/rag/local/upload` with full pipeline: metadata → chunk → embed → upsert.

- **`FileProcessor.chunk_text` (file-centric heuristics):**
  - Attempts paragraph (`\n\n`) or sentence (`. `, `! `, `? `) alignment when possible; otherwise character fallback with overlap.
  - Used by knowledge base scanning/import tools to pre-process `.txt` and related files.

Identity & metadata:
- Document IDs in book uploads are hashed from path and content (`doc_{timestamp}_{pathHash}_{contentHash}`) to ensure idempotence.
- Chunk IDs are deterministic (`{document_id}_chunk_{index}`).
- Metadata includes author/title/worldview when available, total/indices, lengths, text, optional `page_number`, timestamps, and any domain-specific fields produced by `MetadataPipeline`.

Embedding:
- Simple RAG ingestion uses in-process `EmbeddingService` (SentenceTransformers; Apple MPS/CPU auto-detect; FP16 on GPU; cached/warm-up).
- Book uploads call the external `PersonalEmbeddingsService` over HTTP (`/api/v1/embeddings`) with batching and retries, then upsert into ChromaDB.

Upsert & query:
- Local mode uses `LocalVectorStoreManager` (ChromaDB) with a Pinecone-compatible shape (`id`, `values`, `metadata`) for migration ease.
- Query returns Pinecone-like matches with `id`, `score`, and merged `metadata` including `text`.

### Hosting

- Server: FastAPI app in `app/main.py`, typically run with Uvicorn. Lifespan connects/disconnects MongoDB and initializes vector DB lazily (Pinecone only if configured).
- Middlewares:
  - Security headers (CSP, HSTS, X-Frame-Options, etc.) configurable via `ENABLE_SECURITY_HEADERS`.
  - CORS with `BACKEND_CORS_ORIGINS`, `CORS_ALLOW_METHODS/HEADERS`.
  - TrustedHost in production.
  - Rate limiting (slowapi) and unified exception handlers.
  - Request ID injection and structured request timing logs.
- Environments: `settings.ENVIRONMENT` drives `docs_url`/`redoc_url`, host trust, and reload. Configuration via `.env` and env vars. Important keys:
  - Vector DB: `VECTOR_DB_TYPE=local|chroma|pinecone`, paths, index names, dimensions.
  - LLM: `LLM_PROVIDER=openai|deepseek`, model choices per provider.
  - Embeddings: `EMBEDDINGS_MODEL`, `EMBEDDINGS_DIMENSION`, local service URL (`LOCAL_EMBEDDING_SERVICE_URL`).
  - Security: `SECRET_KEY`, token expiries, password policy, API key prefix/length.
  - Mongo: `MONGODB_URI`, `MONGODB_DB_NAME`.
- Local vector setup: see `docs/LOCAL_VECTOR_DB_SETUP.md` for environment examples, persistence path (`./data/vector_db`), and quick verification.

### REST API Interface

Prefix: `settings.API_V1_STR` (default `/api/v1`). Key groups:

- **Auth (`/auth`)**
  - `POST /login` (OAuth2 form) → JWT access/refresh via `user_service`.
  - `POST /register` → create user.
  - `GET /me`, `PUT /me` → current user info/update (JWT/API-Key auth).
  - API Keys: `POST /api-keys`, `GET /api-keys`, `DELETE /api-keys/{id}`, `PATCH /api-keys/{id}/revoke`.
  - Admin users: `GET/POST /users`, `GET/PUT/DELETE /users/{user_id}` (permission-guarded).

- **Assistants (`/assistants`)**
  - `GET /` → list assistants (optionally include capabilities; worldview filter).
  - `GET /{assistant_id}` → assistant detail.
  - `DELETE /{assistant_id}` → delete assistant.
  - `GET /{assistant_id}/models` → available LLM models.
  - Templates:
    - `POST /{assistant_id}/resolve` → JSON with `gedanke`, summaries, timings.
    - `POST /{assistant_id}/reformulate` → variations based on worldview template.
    - `POST /{assistant_id}/glossary` → structured glossary terms.
  - `GET /weltanschauungen/list` → assistants grouped by worldview.

- **Threads (`/threads`)**
  - `POST /` → create thread (returns `id`, metadata, timestamps).
  - `GET /` → list with pagination cursors.
  - `GET /{thread_id}` → fetch.
  - `POST /{thread_id}` → update metadata.
  - `DELETE /{thread_id}` → delete (+ cascade delete messages).

- **Messages (`/threads/{thread_id}/messages`)**
  - `POST /` → create message with content blocks; if role is `user`, the server auto-generates an assistant reply via `RAGService` using optional thread assistant system prompt.
  - `GET /` → list with pagination.
  - `GET /{message_id}` → fetch.
  - `POST /{message_id}` → update metadata.
  - `DELETE /{message_id}` → delete.

- **RAG (`/rag`)**
  - `POST /query` → input conversation (`messages`), optional `filter`, `system_prompt`, `top_k`; returns `{content, model, retrieved_documents}`.
  - `POST /documents` → quick ingestion (content+metadata, simple chunking params).
  - `POST /search` → semantic search over vector DB.
  - Local Vector DB ops (ChromaDB):
    - `POST /local/upload` → full book pipeline (metadata → chunk → embed via personal service → upsert to ChromaDB); returns rich `UploadResult`.
    - `POST /local/search` → query a specific collection using personal embeddings.
    - `GET /local/collections` → list collections.
    - `GET /local/health` → health of embeddings service + Chroma.

- **Health**
  - `GET /api/v1/health` → service + dependency status.
  - `GET /health` and `GET /info` at root for general status and configuration summary.

Security:
- JWT bearer or `X-API-Key` supported (`get_current_user` tries both). RBAC and permission dependencies are provided (`require_permission`, `require_any/all_permissions`, `require_role`). Rate limiting per user/IP.

### Assistant Architecture

- API endpoints in `assistants.py` implement OpenAI-compatible shapes, but route to a hybrid assistant manager (`assistants.deepseek_assistant_manager.DeepSeekAssistantManager`) aliased as `PineconeAssistantManager` for drop-in replacement semantics.
- Worldviews are mapped from assistant identifiers and used by `TemplateProcessor` to construct prompts for three template families: resolve, reformulate, and glossary.
- Responses are parsed from assistant chat output into structured JSON payloads tailored for the domain (e.g., `gedanke`, `gedanke_zusammenfassung`, `gedanke_kind`, `glossar`).
- LLM provider is selected centrally by the factory (`settings.LLM_PROVIDER`), enabling DeepSeek or OpenAI models.
- The `docs/README_assistant_architecture.md` outlines a migration to a code-only assistant definition approach (dataclasses; version control; batch testing, A/B, and fine-tuning examples). The current system already separates assistant routing from RAG: assistant endpoints call LLM with optional knowledge base usage; RAG endpoints use vector retrieval + `llm_service.generate_with_rag` to compose answers.

#### Data flows (typical)

- RAG Query:
  1) Client → `POST /api/v1/rag/query` with messages
  2) `RAGService.query`: embed last user message → vector query (Chroma/Pinecone)
  3) `RAGService.generate_rag_response`: send messages + retrieved context to LLM → return content + retrieved document metadata

- Book Upload (local):
  1) Client → `POST /api/v1/rag/local/upload` with file path/content
  2) `MetadataPipeline` → `TextChunker` → `PersonalEmbeddingsService`
  3) `LocalVectorStoreManager.upsert_vectors` (Chroma)
  4) Return `UploadResult` with counts, quality, and any warnings/errors

---

## Operational Notes

- Pinecone vs Local: `LocalVectorStoreManager` mimics Pinecone’s request/response shape, easing rollback and mixed deployments.
- Performance: embedding batching, MPS/CUDA/CPU auto-detect, warm-up inference, and request-level logging are in place. Batch size is configurable.
- Observability: health endpoints, collection listing/stats, and detailed book upload logs (truncated titles, page hints, text previews) facilitate debugging.


