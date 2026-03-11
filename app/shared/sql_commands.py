"""Named SQL snippets shared for graph event inspection."""

# Latest graph_event_id by creation time.
GET_LATEST_GRAPH_EVENT_ID = """
SELECT graph_event_id
FROM event_metadata
WHERE graph_event_id IS NOT NULL
ORDER BY created_at DESC
LIMIT 1
"""

# All events for a given graph_event_id ordered by creation time.
GET_GRAPH_EVENTS_BY_ID = """
SELECT
  em.id,
  em.graph_event_id,
  em.graph_name,
  em.step,
  em.worldview,
  em.retrieval_mode,
  em.sufficiency,
  em.error_count,
  em.chunk_ids,
  em.created_at,
  ec.concept,
  ec.query_text,
  ec.prompt_messages,
  ec.context_refs,
  ec.context_source,
  ec.context_text,
  ec.response_text,
  ec.errors
FROM event_metadata em
LEFT JOIN event_content ec ON ec.event_metadata_id = em.id
WHERE em.graph_event_id = %(graph_event_id)s
ORDER BY em.created_at ASC
"""
