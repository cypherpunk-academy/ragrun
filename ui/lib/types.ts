export type CollectionStat = { name: string; count: number };
export type StatusCount = { status: string; count: number };
export type UsageByModel = {
  model: string;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  call_count: number;
};
export type UsageTimeseriesPoint = {
  day: string;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
};

export type AdminStats = {
  rag_chunks_total: number;
  rag_chunks_embedded: number;
  rag_chunks_deprecated: number;
  rag_chunks_by_partition: CollectionStat[];
  vector_chunks_total: number;
  vector_chunks_by_collection: CollectionStat[];
  rag_talks_total: number;
  rag_talks_by_status: StatusCount[];
  rag_turns_total: number;
  avg_turns_per_talk: number;
  rag_references_total: number;
  avg_refs_per_turn: number;
  rag_usage_total_calls: number;
  rag_usage_total_tokens: number;
  rag_usage_by_model: UsageByModel[];
  rag_usage_timeseries: UsageTimeseriesPoint[];
};

export type TalkSummary = {
  talk_id: string;
  collection: string;
  title: string;
  slug: string;
  mensch_name: string;
  publishing_status: string;
  created_at?: string;
  updated_at?: string;
  turn_count: number;
};

export type AdminTalksResponse = {
  total: number;
  items: TalkSummary[];
};

export type TalkReference = {
  ref_id: string;
  ref_index: number;
  chunk_id?: string;
  relevance?: number;
  source_title?: string;
  segment_title?: string;
};

export type TalkUsage = {
  id: number;
  model?: string;
  provider: string;
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
  created_at?: string;
};

export type TalkTurn = {
  turn_id: string;
  turn_index: number;
  user_message: string;
  assistant_message: string;
  updated_at?: string;
  references: TalkReference[];
  usage: TalkUsage[];
};

export type TalkDetails = {
  talk_id: string;
  collection: string;
  title: string;
  slug: string;
  mensch_name: string;
  publishing_status: string;
  summary?: string;
  bug_description?: string;
  turns: TalkTurn[];
};
