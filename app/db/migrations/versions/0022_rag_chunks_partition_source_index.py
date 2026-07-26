"""Index rag_chunks (rag_partition, source_id) for orphan deprecation."""

import sqlalchemy as sa
from alembic import op

revision = "0022_rag_chunks_partition_source"
down_revision = "0021"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(
        "idx_rag_chunks_rag_partition_source_id",
        "rag_chunks",
        ["rag_partition", "source_id"],
        unique=False,
        postgresql_where=sa.text("deprecated_at IS NULL"),
    )


def downgrade() -> None:
    op.drop_index("idx_rag_chunks_rag_partition_source_id", table_name="rag_chunks")
