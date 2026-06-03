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

-- rag_paragraphs: individual paragraph texts linked to their source lecture/book
CREATE TABLE IF NOT EXISTS rag_paragraphs (
    id VARCHAR(512) NOT NULL PRIMARY KEY,
    source_id VARCHAR(256) NOT NULL,
    language VARCHAR(8) NOT NULL,
    segment_index INTEGER NOT NULL,
    segment_title TEXT NOT NULL,
    paragraph_number INTEGER NOT NULL,
    text_raw TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_rag_paragraphs_source_id ON rag_paragraphs(source_id);

-- app_paragraph_chunk: many-to-many mapping between paragraphs and chunks
CREATE TABLE IF NOT EXISTS app_paragraph_chunk (
    paragraph_id VARCHAR(512) NOT NULL,
    chunk_id VARCHAR(256) NOT NULL,
    rag_partition VARCHAR(128) NOT NULL,
    PRIMARY KEY (paragraph_id, chunk_id, rag_partition)
);

CREATE INDEX IF NOT EXISTS idx_app_paragraph_chunk_chunk_id ON app_paragraph_chunk(chunk_id);
