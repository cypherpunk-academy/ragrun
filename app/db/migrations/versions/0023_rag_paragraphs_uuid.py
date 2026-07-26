"""rag_paragraphs UUID primary key, segment_slug natural key, lemma_fingerprint."""

import sqlalchemy as sa
from alembic import op

revision = "0023_rag_paragraphs_uuid"
down_revision = "0022_rag_chunks_partition_source"
branch_labels = None
depends_on = None


def upgrade() -> None:
  # Fresh cutover — nothing live on paragraphs/bookmarks/notes/talks mappings.
  #
  # rag_chunks: Paragraph-Navigation läuft über app_paragraph_chunk. Nach UUID-Cutover
  # sind alte Mappings weg; verknüpfte Chunk-Zeilen würden sonst in Suche/Qdrant
  # ohne paragraph_id landen. Zusätzlich Legacy-Zitate (source_id …:quotes).
  op.execute(
      """
      DELETE FROM rag_chunks
      WHERE chunk_id IN (SELECT DISTINCT chunk_id FROM app_paragraph_chunk)
      """
  )
  op.execute(
      """
      DELETE FROM rag_chunks
      WHERE source_id LIKE '%' || chr(58) || 'quotes'
         OR chunk_type IN ('book', 'secondary_book', 'talk', 'quote', 'quote_explanation')
      """
  )

  op.execute("TRUNCATE app_paragraph_chunk")
  op.execute("TRUNCATE rag_paragraphs CASCADE")
  op.execute("TRUNCATE app_bookmarks CASCADE")
  op.execute("TRUNCATE app_notes CASCADE")
  op.execute("TRUNCATE rag_talks CASCADE")

  op.execute(
      """
      ALTER TABLE rag_paragraphs
        ADD COLUMN IF NOT EXISTS segment_slug TEXT,
        ADD COLUMN IF NOT EXISTS lemma_fingerprint JSONB,
        ADD COLUMN IF NOT EXISTS deprecated_at TIMESTAMPTZ
      """
  )

  # Drop old PK and recreate as UUID (existing rows were truncated).
  op.execute("ALTER TABLE rag_paragraphs DROP CONSTRAINT IF EXISTS rag_paragraphs_pkey")
  op.execute(
      """
      ALTER TABLE rag_paragraphs
        ALTER COLUMN id DROP DEFAULT,
        ALTER COLUMN id TYPE UUID USING gen_random_uuid()
      """
  )
  op.execute("ALTER TABLE rag_paragraphs ALTER COLUMN id SET DEFAULT gen_random_uuid()")
  op.execute("ALTER TABLE rag_paragraphs ADD PRIMARY KEY (id)")

  op.execute(
      """
      ALTER TABLE rag_paragraphs
        DROP CONSTRAINT IF EXISTS rag_paragraphs_natural_key
      """
  )
  op.execute(
      """
      CREATE UNIQUE INDEX IF NOT EXISTS rag_paragraphs_natural_key
        ON rag_paragraphs (source_id, segment_slug, paragraph_number)
        WHERE deprecated_at IS NULL
      """
  )

  op.execute(
      """
      ALTER TABLE app_paragraph_chunk
        ALTER COLUMN paragraph_id TYPE uuid USING paragraph_id::uuid
      """
  )


def downgrade() -> None:
  op.execute("DROP INDEX IF EXISTS rag_paragraphs_natural_key")
  op.execute("ALTER TABLE rag_paragraphs DROP COLUMN IF EXISTS lemma_fingerprint")
  op.execute("ALTER TABLE rag_paragraphs DROP COLUMN IF EXISTS segment_slug")
  op.execute("ALTER TABLE rag_paragraphs DROP COLUMN IF EXISTS deprecated_at")
  op.execute("ALTER TABLE rag_paragraphs ALTER COLUMN id TYPE VARCHAR(512) USING id::text")
