"""rename worldview to worldviews array"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Convert worldview column to worldviews array.
    
    Steps:
    1. Add new worldviews column as text[]
    2. Migrate data: convert single worldview string to array
    3. Drop old worldview column
    """
    # Add the new worldviews column as ARRAY(String)
    op.add_column(
        "rag_chunks",
        sa.Column("worldviews", postgresql.ARRAY(sa.String()), nullable=True),
    )
    
    # Migrate existing data: wrap single worldview value in array
    op.execute("""
        UPDATE rag_chunks
        SET worldviews = ARRAY[worldview]
        WHERE worldview IS NOT NULL
    """)
    
    # Drop the old worldview column
    op.drop_column("rag_chunks", "worldview")


def downgrade() -> None:
    """Revert worldviews array back to single worldview string.
    
    Warning: This will lose data if a chunk has multiple worldviews.
    Only the first worldview will be preserved.
    """
    # Add back the old worldview column
    op.add_column(
        "rag_chunks",
        sa.Column("worldview", sa.String(length=128), nullable=True),
    )
    
    # Migrate data: take first element from worldviews array
    op.execute("""
        UPDATE rag_chunks
        SET worldview = worldviews[1]
        WHERE worldviews IS NOT NULL AND array_length(worldviews, 1) > 0
    """)
    
    # Drop the worldviews column
    op.drop_column("rag_chunks", "worldviews")
