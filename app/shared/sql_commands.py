"""Named SQL snippets shared for graph event inspection."""

# Latest graph_event_id by creation time.
GET_LATEST_GRAPH_EVENT_ID = """
SELECT graph_event_id
FROM retrieval_graph_events
ORDER BY created_at DESC
LIMIT 1
"""

# All events for a given graph_event_id ordered by creation time.
GET_GRAPH_EVENTS_BY_ID = """
SELECT
  id,
  graph_event_id,
  graph_name,
  step,
  concept,
  worldview,
  query_text,
  prompt_messages,
  context_refs,
  context_source,
  context_text,
  response_text,
  retrieval_mode,
  sufficiency,
  errors,
  metadata,
  created_at
FROM retrieval_graph_events
WHERE graph_event_id = %(graph_event_id)s
ORDER BY created_at ASC
"""