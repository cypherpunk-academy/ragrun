### Purpose
Design a new query endpoint and retrieval architecture with strong filtering and early-exit logic. Capture current endpoint usage (by ragprep), deprecation candidates, conceptual schema updates, and an implementation-agnostic plan leveraging LangChain/LangGraph building blocks.

---

### Current API usage by ragprep (confirmed)
- Used endpoints:
  - `GET /rag/collections`
  - `GET /rag/books/titles`
  - `POST /rag/upload-chunks`
  - `POST /rag/delete-chunks`

- Not used by ragprep (candidates for deprecation from ragprep POV):
  - `/assistants/**`
  - `/threads/**`
  - `/threads/{thread_id}/messages/**`
  - Legacy `/rag` query endpoints (ragprep currently doesn’t call them)

Note: These may be used by other clients. Deprecation here means “not needed for ragprep flows”. Keep health and auth as-is.

---

### Potential use of deprecated endpoints for `rp rag:debug`
- Goal: Reuse existing, less-used endpoints (e.g., `/assistants/**`, `/threads/**`, `/threads/{thread_id}/messages/**`) as an admin/diagnostic surface for `rp rag:debug` without impacting end-user flows.

- Capabilities (read-only):
  - Vector DB overview: total chunks, by `chunk_type`, by `collection`, by `book_id`, by `author`, by `chapter`, by `essay_title`.
  - Data health: orphaned chunks (missing parent), duplicate chunk hashes, inconsistent page/chapter ordering, broken references, missing metadata fields, embedding out-of-date vs content hash.
  - Index status: last maintenance time, pending reindex counts, shard sizes, top-N largest parents.
  - Ingestion logs: recent uploads/deletions with counts and durations, failures with reasons.

- Capabilities (write/patch — gated, audit-only):
  - Patch metadata on specific chunks or parents: set `chunk_type`, `essay_title`, fix `author`, adjust `chapter` numbering, reassign `collection`.
  - Repair: re-embed stale chunks, reindex collections, delete/undelete flagged chunks, merge/split adjacent chunks within a parent.
  - Bulk ops by filter: apply patches to all items matching metadata predicates (with dry-run report). 

- Safety & access:
  - Require admin role, explicit environment flag, and CSRF/nonce for write ops.
  - All write ops are dry-run by default; require `confirm=true` to apply.
  - Emit audit trail (who, when, filters, diff summary); keep a rollback plan (export pre-patch snapshot ids).

- CLI mapping (`rp rag:debug`):
  - `rp rag:debug state --facets chunk_type,author,collection` → GET debug state & facet counts
  - `rp rag:debug check --checks orphaned,duplicates,stale-embeddings` → GET health report
  - `rp rag:debug reembed --filter stale=true --confirm` → trigger re-embedding jobs

- Endpoint sketch (repurpose or alias under `/rag/debug` while keeping old mounts):
  - `GET /rag/debug/state` → overview + facets + top warnings
  - `GET /rag/debug/checks?checks=orphaned,duplicates,stale-embeddings` → detailed diagnostics
  - `POST /rag/debug/reindex` → body: { collections?, parents? }

- Why reuse deprecated endpoints:
  - Minimizes new surface area by leveraging existing auth/middleware and router wiring.
  - Allows keeping consumer-facing API clean while enabling operator-grade maintenance.
  - Clear separation: `ragprep` never calls these; only the `rp` CLI uses them under admin credentials.

---

### Conceptual schema updates (ChunkOutput & metadata)
- Add fields:
  - `chunk_type`: enum: `book | secondary_book | concept | essay | order | question | summary`
  - `essay_title`: string (optional)
  - `chunk_type` descriptions:
    - `book`: Chunk is part of a book with a unique id and a title
    - `secondary_book`: Chunk is part of a book that is mentioned or broadens the view of the central books of the collection
    - `concept`: A rich, lifestrong concept (Begriff) with one word as title and a text to describe it, limited to one chunk
    - `essay`: A max 5‑chunk long text following a certain structure (7 steps, described in the prompt) that addresses a topic; can be a Begriff or a question or any topic interesting around the books
    - `order`: A zusammengehörige list of Begriffe, like 12 Weltanschauungen or 12 senses, limited to one chunk
    - `question`: A question that is common around the books of the collection, limited to one chunk
    - `summary`: A summary of an essay, a chapter (level 1, 2 or 3), book etc.; first line states the relation, like "## Chapter about something" or "What is Freiheit?"

- Notes:
  - `book` and `essay` content spans multiple chunks. Retrieval and response formatting should support grouping and collapsing by parent document (`book_id`, `essay_title`).
  - Ensure these attributes are present anywhere chunk parameters are referenced: stored metadata in the vector DB, API models (server), and client-side types (ragprep `ragSchemas.ts`).
  - Maintain backward compatibility: treat missing `chunk_type` as `book` (or `unknown`) during transition.

---

### New query endpoint (concept only)
- Path: `POST /rag/query` (v2 semantics). Consider `X-Query-Version: 2` header or a `?v=2` param if keeping older behavior in parallel during migration.

- Request (conceptual):
  - `prompt`: string
  - `filters` (all optional):
    - `book_ids`: string[]
    - `chunk_types`: (`chunk_type`)[]  // enum values above
    - `authors`: string[]
    - `chapters`: (number | string)[]  // by number or normalized chapter ids
    - `essay_titles`: string[]
    - `collections`: string[]
  - `retrieve`: object
    - `k`: number (default 8–12)
    - `max_parent_docs`: number (for grouping)
    - `ranking`: string (e.g., "ensemble", "vector", "bm25", "rrf")
    - `threshold`: number (semantic score min; used for direct-hits too)
  - `generation`: object
    - `mode`: `none | synthesize | answer`
    - `max_tokens`: number
    - `language`: `de | en | auto`
  - `debug`: boolean

- Response (conceptual):
  - `direct_hits`: array of high-confidence hits from `summary | order | concept | essay`
  - `retrieved`: array of grouped results (with children chunks)
  - `facets`: counts by `chunk_type`, `author`, `book_id`, `chapter`, `collection`
  - `used_filters`: echo back normalized filters
  - `strategy_trace` (if debug): which stages ran, timings, thresholds, branches that early-exited
  - `answer` (if `generation.mode != none`): synthesized text and citations

---

### Retrieval strategy: direct-hit + double-tap

1) Direct-hit detection (fast path, early exit)
   - Purpose: If the prompt is satisfied by a `summary | order | concept | essay`, return immediately (or prioritize these in the response) without deep retrieval.
   - Signals (use a hybrid of lightweight heuristics and LLM classification):
     - Metadata filters strongly match (e.g., exact `essay_title`, matching `order` name)
     - High semantic similarity against a dedicated index for `summary/order/concept/essay`
     - Intent classifier labels prompt as "form request" (e.g., "Give me the summary/order/definition of X") vs "content request"
   - Implementation building blocks:
     - LangChain SelfQueryRetriever for translating filters to metadata constraints
     - Separate retriever tuned for special types (`summary/order/concept/essay`) with higher match threshold
     - Optional BM25 inverted index for exact/near-exact term hits
   - Outcome:
     - If a confident direct hit is found (score ≥ threshold), return it under `direct_hits`. If `generation.mode=answer`, produce a short synthesis grounded only on those direct hits.

2) Double-tap pipeline (parallel-first, then iterative refinement)
   - Stage A (in parallel):
     - A1: Direct-hit search (above)
     - A2: Book-level retrieval using filters (vector + BM25 ensemble)
   - Stage B (refine on A2 results):
     - Expand within selected books: search both `book` and `secondary_book`
     - ParentDocumentRetriever or hierarchical retrieval to pull relevant chunks and minimal surrounding context
   - Stage C (generation, optional):
     - Context compression and reranking (ContextualCompressionRetriever, Maximal Marginal Relevance)
     - Prompt construction and synthesis with strict citation controls
   - Orchestration:
     - Use LangGraph to run A1/A2 concurrently, fan-in results, evaluate early-exit conditions, and proceed to B/C as needed
     - Configure timeouts and budget per stage; degrade gracefully under load

---

### Pre-LLM steps commonly used and helpful
- Query understanding & routing
  - Intent classification: content vs form (definition, summary, outline) vs task (write, transform)
  - Query rewriting: LangChain MultiQueryRetriever to generate diversified paraphrases for better recall
  - Tool/routing policy: choose direct-hit-only, ensemble retrieval, or broader exploration based on intent

- Retrieval hardening
  - EnsembleRetriever: vector + BM25 + keyword filters; combine via Reciprocal Rank Fusion (RRF)
  - Metadata-first pruning: apply `book_ids`, `chunk_types`, `authors`, `chapters`, `essay_titles`, `collections` before dense retrieval
  - Parent/child retrieval: fetch minimal children under selected parent doc (book/essay) to reduce context bloat
  - Context compression: LLMChainFilter or embeddings-based compression to remove low-signal sentences
  - Deduplication & clustering: merge near-duplicates and collapse adjacent chunks from same parent

- Safety & quality gates
  - Source coverage threshold: don’t generate if context is too thin; return curated hits instead
  - Citation enforcement: require n≥2 distinct sources for open-ended claims
  - Language handling: auto-detect prompt language, prefer matching-language sources when available

---

### LangChain/LangGraph building blocks to leverage (non-prescriptive)
- LangChain
  - SelfQueryRetriever for structured metadata filters
  - MultiQueryRetriever for diversified query generation
  - EnsembleRetriever (or manual RRF) for combining vector + BM25
  - ParentDocumentRetriever for hierarchical retrieval (book/essay grouping)
  - ContextualCompressionRetriever + LLMChainFilter for context shrink
  - MMR (Maximal Marginal Relevance) for diversity

- LangGraph
  - Graph nodes: DirectHitNode, BookSearchNode, ExpansionNode, CompressionNode, GenerationNode
  - Parallel branches: run DirectHitNode and BookSearchNode concurrently
  - Conditional edges: early-exit when high-confidence direct hit is present
  - Tracing: capture timings and branch decisions for `strategy_trace`

---

### Telemetry, evaluation, and learning loops
- Logging (foundation): persist queries, top‑K retrievals, selected contexts, answers, and user feedback (👍/👎, clicks, dwell time) in a structured, privacy‑aware store keyed by request/session ids
- Good/Bad pairs: treat cited chunks as positives and ignored/misleading chunks as negatives; feed supervised reranker training and context pruning
- Hard‑negative mining: curate confusable pairs (e.g., Stoicism vs Epicureanism) to improve discriminative retrievers and rerankers
- Lückenfinder: detect no‑answer cases and faulty citations; open issues for ingestion gaps, source expansion, or chunking adjustments
- Curated gold set: weekly 30–50 real questions with verified gold documents; track nDCG/Recall/MRR and regressions across model/index changes
- Pipeline note: nightly jobs aggregate logs → generate training/eval datasets → run offline evals; promote new rerankers only if metrics improve

---

### Response shaping & UX
- Grouping: Return grouped results by `book_id` and `essay_title`, each with ordered chunks
- Facets: Provide counts to enable client-side filter refinement
- Pagination: `offset/limit` on groups and on chunks-within-group
- Scoring: expose both raw similarity and normalized rank scores; include which retrievers contributed

---

### Migration & compatibility plan
1) Add `chunk_type`, `essay_title` to stored metadata and API models; backfill existing data with best-effort inference
2) Introduce `/rag/query` v2 behavior gated by param/header; keep legacy behavior available for a deprecation window
3) Update ragprep types (`ragSchemas.ts`) to accept new fields; ensure filters map to server-side metadata filters
4) Roll out LangGraph orchestration gradually: start with direct-hit and book search in parallel; add expansion + compression once validated
5) Instrumentation: add tracing and per-stage timing to monitor hit quality and latency

---

### Open questions
- Exact chapter identifier scheme to standardize across books (number vs slug vs TOC path)
- Confidence thresholds for direct-hit early exit per `chunk_type`
- Minimum context size and citation policy before allowing generation
- Whether `/assistants`, `/threads`, `/messages` serve other clients we must preserve

---

### Next steps (no code yet)
- Confirm deprecation scope based on non-ragprep consumers
- Finalize filter field names and chapter ID format
- Decide on query versioning approach (param vs header vs new path)
- Choose initial LangChain/LangGraph components and evaluation metrics for each stage


