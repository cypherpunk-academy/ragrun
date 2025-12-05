-- Initialize the rag_chunks table for ragrun
-- This table mirrors chunk metadata from Qdrant for SQL queries and analytics

CREATE TABLE IF NOT EXISTS rag_chunks (
    collection VARCHAR(128) NOT NULL,
    chunk_id VARCHAR(256) NOT NULL,
    source_id VARCHAR(256) NOT NULL,
    chunk_type VARCHAR(64) NOT NULL,
    language VARCHAR(8) NOT NULL,
    worldview VARCHAR(128),
    importance INTEGER,
    content_hash VARCHAR(128) NOT NULL,
    text TEXT,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    metadata JSONB NOT NULL,
    PRIMARY KEY (collection, chunk_id)
);

-- Optional: Add indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_chunks_source_id ON rag_chunks(source_id);
CREATE INDEX IF NOT EXISTS idx_chunks_worldview ON rag_chunks(worldview) WHERE worldview IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_chunks_created_at ON rag_chunks(created_at DESC);
