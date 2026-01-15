# Concept Explain Worldviews Event Persistence

## Purpose
Capture per-step retrieval and LLM events for the concept→worldview graph with a shared `graph_event_id` (UUIDv4). The schema is general so other graphs can use it via `graph_name`.

## Schema (Postgres)
Table: `retrieval_graph_events`

- `id` bigserial PRIMARY KEY  
- `graph_event_id` uuid NOT NULL — run-level correlation  
- `graph_name` text NOT NULL — e.g., `concept_explain_worldviews`  
- `step` text NOT NULL — e.g., `concept_retrieval`, `concept_reasoning`, `wv_context1_retrieval`, `wv_what`, `wv_context2_retrieval`, `wv_how`  
- `concept` text NOT NULL  
- `worldview` text NULL — NULL for concept-level steps  
- `query_text` text NULL — retrieval query  
- `prompt_messages` jsonb NULL — LLM messages/prompt  
- `context_refs` jsonb NULL — IDs/metadata of retrieved snippets  
- `context_text` text NULL — flattened context window  
- `response_text` text NULL — LLM output  
- `retrieval_mode` text NULL — dense/hybrid/dense->hybrid  
- `sufficiency` text NULL — insufficient/low/medium/high  
- `errors` jsonb NULL — array of error strings  
- `metadata` jsonb NULL — k-values, widen flags, timings, etc.  
- `created_at` timestamptz NOT NULL DEFAULT now()

Indexes:
- `idx_rge_graph_event_id` on `(graph_event_id)`
- `idx_rge_graph_event_step` on `(graph_event_id, step)`
- Optional `(graph_event_id, worldview)` for per-worldview reads.

Example DDL:
```sql
CREATE TABLE retrieval_graph_events (
  id bigserial PRIMARY KEY,
  graph_event_id uuid NOT NULL,
  graph_name text NOT NULL,
  step text NOT NULL,
  concept text NOT NULL,
  worldview text NULL,
  query_text text NULL,
  prompt_messages jsonb NULL,
  context_refs jsonb NULL,
  context_text text NULL,
  response_text text NULL,
  retrieval_mode text NULL,
  sufficiency text NULL,
  errors jsonb NULL,
  metadata jsonb NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX idx_rge_graph_event_id ON retrieval_graph_events(graph_event_id);
CREATE INDEX idx_rge_graph_event_step ON retrieval_graph_events(graph_event_id, step);
```

## Write points in `run_concept_explain_worldviews_graph`
Generate `graph_event_id = uuid4()` and set `graph_name = "concept_explain_worldviews"` near graph start.

1) Concept retrieval  
   - step: `concept_retrieval`  
   - fields: `query_text=concept`, `context_refs/context_text` from `concept_outcome`, `retrieval_mode=concept_outcome.mode`, `metadata` (k-values, widen flags).

2) Concept reasoning (DeepSeek reasoning)  
   - step: `concept_reasoning`  
   - fields: `prompt_messages=philo_messages`, `response_text=concept_explanation`, optionally reuse `context_*`.

Per-worldview (loop):

3) Context1 retrieval (primary)  
   - step: `wv_context1_retrieval`  
   - fields: `worldview=wv`, `query_text=concept_explanation`, `context_refs/context_text` from `ctx1_outcome`, `retrieval_mode=ctx1_outcome.mode`.

4) “What” LLM call (chat)  
   - step: `wv_what`  
   - fields: `prompt_messages=what_prompt`, `response_text=main_points`, optionally `context_refs/context_text` (ctx1).

5) Context2 retrieval (primary+secondary)  
   - step: `wv_context2_retrieval`  
   - fields: `worldview=wv`, `query_text=concept_explanation`, `context_refs/context_text` from `ctx2_outcome`, `retrieval_mode=ctx2_outcome.mode`, `sufficiency`.

6) “How” LLM call (chat)  
   - step: `wv_how`  
   - fields: `prompt_messages=how_prompt`, `response_text=how_details`, `context_refs/context_text` (ctx2), `sufficiency`, `errors` if any.

Optional lifecycle events:
- `graph_start` at entry with basic metadata.
- `graph_end` with summary counts and sufficiency histogram.

## API sketch (pseudocode)
```python
async def record_graph_event(
    db,
    graph_event_id: UUID,
    *,
    graph_name: str,
    step: str,
    concept: str,
    worldview: str | None = None,
    query_text: str | None = None,
    prompt_messages: dict | list | None = None,
    context_refs: list | dict | None = None,
    context_text: str | None = None,
    response_text: str | None = None,
    retrieval_mode: str | None = None,
    sufficiency: str | None = None,
    errors: list[str] | None = None,
    metadata: dict | None = None,
) -> None:
    ...
```
