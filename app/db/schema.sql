-- vector_chunks: mirror of Qdrant payloads for SQL queries and analytics
CREATE TABLE IF NOT EXISTS vector_chunks (
    collection VARCHAR(128) NOT NULL,
    chunk_id VARCHAR(256) NOT NULL,
    source_id VARCHAR(256) NOT NULL,
    chunk_type VARCHAR(64) NOT NULL,
    language VARCHAR(8) NOT NULL,
    worldviews TEXT[],
    importance INTEGER,
    content_hash VARCHAR(128) NOT NULL,
    text TEXT,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    metadata JSONB NOT NULL,
    references JSONB,
    PRIMARY KEY (collection, chunk_id)
);

CREATE INDEX IF NOT EXISTS idx_vector_chunks_source_id ON vector_chunks(source_id);
CREATE INDEX IF NOT EXISTS idx_vector_chunks_created_at ON vector_chunks(created_at DESC);

-- rag_chunks: primary chunk store (DB-first); embedded_at set after Qdrant sync
CREATE TABLE IF NOT EXISTS rag_chunks (
    rag_partition VARCHAR(128) NOT NULL,
    chunk_id VARCHAR(256) NOT NULL,
    source_id VARCHAR(256) NOT NULL,
    chunk_type VARCHAR(64) NOT NULL,
    language VARCHAR(8) NOT NULL,
    worldviews TEXT[],
    importance INTEGER,
    content_hash VARCHAR(128) NOT NULL,
    text TEXT,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    metadata JSONB NOT NULL,
    references JSONB,
    scope VARCHAR(64),
    embedded_at TIMESTAMPTZ,
    PRIMARY KEY (rag_partition, chunk_id)
);

CREATE INDEX IF NOT EXISTS idx_rag_chunks_source_id ON rag_chunks(source_id);
CREATE INDEX IF NOT EXISTS idx_rag_chunks_created_at ON rag_chunks(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_rag_chunks_rag_partition_embedded_at ON rag_chunks(rag_partition, embedded_at);
