"""Initialize the database schema for ragrun."""
from __future__ import annotations

from sqlalchemy import create_engine

from app.config import settings
from app.db.tables import metadata


def init_db() -> None:
    """Create all tables in the database."""
    engine = create_engine(str(settings.postgres_dsn))
    metadata.create_all(engine)
    print("Database schema initialized successfully.")


if __name__ == "__main__":
    init_db()
