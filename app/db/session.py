"""Engine factory and helpers."""
from __future__ import annotations

from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from app.config import settings

from .tables import metadata


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Create (or reuse) the SQLAlchemy engine and ensure tables exist."""

    engine = create_engine(
        settings.postgres_dsn,
        future=True,
        pool_pre_ping=True,
    )
    metadata.create_all(engine)
    return engine

