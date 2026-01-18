# LangChain/LangGraph Implementation Plan

## Scope
- Implement LangGraph-based endpoint `concept_explain_worldviews.py` (replaces `retrieval/chains/concept_explain.py`).
- Inputs: `{ concept: string, worldviews: string[] }`.
- Outputs (JSON): `{ concept_explanation, worldviews: [{ worldview, main_points, how_details, context_refs }] }`.
- Models: DeepSeek Reasoner v3 for step 1; DeepSeek Chat for step 2 (both prompts hardened).

## Topology (LangGraph)
```mermaid
flowchart TD
  startInput[Input concept, worldviews] --> prepInputs[prepare_inputs]
  prepInputs --> philoExplain[philo_explain (DeepSeek Reasoner, k10 primary)]
  philoExplain --> mapWorldviews{per_worldview map}
  mapWorldviews --> whatStep[worldview_what (DeepSeek Chat, k5 context1)]
  whatStep --> howStep[worldview_how (DeepSeek Chat, k10 context2 + main_points)]
  howStep --> assemble[assemble_results]
  assemble --> endOutput[Return JSON]
```

### State shape
```ts
{
  request_id: string;
  graph_id: string;
  concept: string;
  worldviews: string[];
  concept_explanation?: string;
  per_worldview?: Record<string, {
    main_points?: string;
    how_details?: string;
    context1_refs?: string[];
    context2_refs?: string[];
    errors?: string[];
  }>;
  errors?: string[];
}
```

### Nodes
- `prepare_inputs`: validate concept/worldviews, normalize casing, init IDs.
- `philo_explain`: Runnable with prompt `assistants/philo-von-freisinn/assistant/prompts/concept-explain-user.prompt`; retrieval k=10, filter: primary books of Philo only; model: DeepSeek Reasoner; logs prompt/context checksums.
- `worldview_prepare` (inline in map): load worldview description (`.../instructions.md`), build retrievers:
  - `context1`: k=5, filter worldview primary books.
  - `context2`: k=10, filter worldview all books (book + secondary_book).
- `worldview_what`: prompt `.../worldviews/<wv>/prompts/concept-explain-what.prompt`; vars `{concept_explanation, context1, worldview_description}`; model: DeepSeek Chat.
- `worldview_how`: prompt `.../worldviews/<wv>/prompts/concept-explain-how.prompt`; vars `{concept_explanation, context2, worldview_description, main_points}`; model: DeepSeek Chat.
- `assemble_results`: merge per-worldview outputs, attach context refs, collect errors.

## Files to add/modify
- `app/retrieval/graphs/concept_explain_worldviews.py` – LangGraph builder (async), concurrency caps for per-worldview map.
- `app/retrieval/chains/concept_explain_worldviews.py` – thin entrypoint callable by API (keeps compatibility naming).
- `app/retrieval/services/providers.py` (if needed) – expose DeepSeek Reasoner/Chat providers with model names; keep provider/ID utils under `core/infra/shared` per Architektur-Doku.
- `app/retrieval/prompts/` – helper loader for prompts (reuse existing utilities).
- `app/api/` (router/service) – route to new graph; deprecate old chain alias.
- Tests: `tests/retrieval/test_concept_explain_worldviews.py` with mocked LLM/Qdrant.
- `app/retrieval/telemetry/` – optional: shared logging hooks; add dedicated `persist`/telemetry node for this graph.

## Tracing, IDs, telemetry
- Generate `graph_id` (UUIDv7) per graph; `run_id` per node/tool.
- Persist to DB via dedicated `persist`/telemetry node (after `assemble_results`) or logging hooks: `request_id`, `graph_id`, `run_id`, `node`, `branch/worldview`, `llm_model`, `prompt_checksum`, `context_checksum`, `chunk_ids`, `k`, `filters`, durations, token counts.
- Optional LangFuse spans with same IDs; noop if unavailable.

## Error handling and retries
- Validate inputs (non-empty concept, worldviews whitelist).
- If retrieval empty: emit warning, continue with placeholder note; still log context_checksum.
- LLM failure per node: record error in `per_worldview[wv].errors`; continue other worldviews; top-level errors aggregated.
- Retries/backoff/timeouts: per-node timeouts; optional single retry on transient (HTTP 5xx/timeout) with backoff; concurrency limits via semaphores (separate caps for LLM and Qdrant).
- Concurrency inside per-worldview map: set explicit semaphore/max_concurrency (e.g., 4–6 worldview branches) to stay under ~8–10 concurrent LLM calls even when widening; document the chosen value in the graph builder.

## Security and prompt hardening
- Escape user input in prompts; enforce max length on concept/worldview strings (concept ≤ 256 chars; worldviews must be in whitelist); deny tool usage in system prompts; optional anomaly detection/policy model before returning answers.
- Strict filters per worldview; deny tool usage in system prompts.
- If context insufficient, instruct models to answer with `"Unzureichender Kontext"`.

## Retrieval k and reranking
- Default: dynamic k with widening as the normal path (not fixed k).
  - Initial k_base: concept/primary = 10, context1 (worldview primary) = 5, context2 (primary+secondary) = 10.
  - Rerank/compress to k_final (e.g., 6 for concept/context2, 4–6 for context1):
    - Option A: embedding similarity reranker (cheap).
    - Option B: LLM-as-compressor (contextual compression retriever, higher quality).
  - Widen rule (default on): trigger when reranker coverage/similarity is low or LLM signals “insufficient context”; widen k_base (e.g., concept 10→18–20; context1 5→10; context2 10→18) and rerank again; keep k_final same or slightly larger (e.g., 6→8).
  - Log chosen k_base, widened k_base (if any), k_final, and pre/post rerank chunk_ids/scores/checksums.
  - Apply separately to `context1` (primary) and `context2` (primary+secondary); shared rerank logic, distinct filters.
  - Safeguards: cap k_base to avoid overlong prompts; per-node timeouts and semaphores so widened k does not overload Qdrant/LLM.

## Retrieval strategy default vs optional hybrid
- Default: dense + reranker/compressor (no sparse leg) for lower latency/complexity. Ship this first.
- Optional hybrid (BM25/TF-IDF + dense) with RRF (feature-flagged):
  - Add sparse leg over same corpus and filters.
  - Per retriever call (concept primary; worldview context1 primary; worldview context2 primary+secondary):
    - Dense: vector search k_dense (e.g., 12–15).
    - Sparse: BM25/TF-IDF k_sparse (e.g., 30).
    - Fuse via RRF: fused_score = Σ 1/(k_rrf + rank_i), k_rrf ≈ 60–100; deduplicate by chunk_id.
    - Take top fused (e.g., 30) into reranker/compressor to k_final (6–8).
    - Log dense_score, sparse_score, fused_score, rerank_score, chunk_ids.
  - Safeguards: cap k_sparse/k_dense for latency; semaphores/timeouts; ensure sparse index stays in sync with payload filters.

## Implementation steps for hybrid + reranker
1) Add sparse retriever (BM25/TF-IDF) with worldview/book_type filters; expose via providers/infra.
2) Extend graph retriever utility to issue dense + sparse in parallel per call; collect scores.
3) Implement RRF fusion with chunk_id dedup; configurable k_rrf; return fused list.
4) Feed fused list to existing reranker/compressor (embedding or LLM) to k_final; keep logging of all score layers.
5) Wire into nodes:
   - `philo_explain`: dense+sparse on concept over primary books; fuse → rerank → prompt.
   - `worldview_what` (`context1`): dense+sparse over worldview primary; fuse → rerank → prompt.
   - `worldview_how` (`context2`): dense+sparse over worldview primary+secondary; fuse → rerank → prompt.
6) Update telemetry schema/tests to capture dense/sparse/fused/rerank scores and chunk_ids; ensure dedup is validated.
7) State reducer for fan-out: for parallel worldview branches writing to `per_worldview` dict, set an explicit reducer/merge function (e.g., `operator.or_` or custom merge) to avoid lost updates in LangGraph fan-out.
8) Sufficiency flag: have `worldview_how` (or a tiny follow-up node) emit `sufficiency: high|medium|low|insufficient` based on grounding quality; surface in UI and reuse for auto-widen in future.

## DeepSeek model selection and health checks
- Separate providers for reasoning vs chat models; defaults via settings (e.g., `DEEPSEEK_REASONER_MODEL`, `DEEPSEEK_CHAT_MODEL`).
- Use reasoning model for `philo_explain`, chat model for worldview steps.
- Fallback: if reasoning model unavailable, downgrade to strongest available variant with warning.
- Health/diagnostic: startup check or endpoint that calls `/models` (if exposed) or a minimal probe call to confirm the configured model exists.
- Model naming reality (Jan 2026): prefer current frontier models (e.g., `deepseek-v3.2` / `deepseek-v3.2-speciale` reasoning) via env vars; legacy `deepseek-chat`/`deepseek-reasoner` kept as fallbacks.

## Testing matrix
- Unit: state transitions per node; prompt variable rendering; retriever filters (context1 vs context2).
- Integration (mocked LLM/Qdrant): full graph run with two worldviews; ensure map executes what→how sequencing.
- Telemetry contract: schema for persist payload (request_id, graph_id, run_id, node, branch, chunk_ids).

## Migration steps
- Mark old `retrieval/chains/concept_explain.py` as deprecated or delegate to new graph.
- Wire API endpoint to new entrypoint; update docs/examples.
- Keep model names configurable via env/settings with sane defaults (Reasoner v3, Chat v3).
