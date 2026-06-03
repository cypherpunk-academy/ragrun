"""Engine factory and helpers."""
from __future__ import annotations

from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.pool import NullPool

from app.config import settings

from .tables import metadata


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Create (or reuse) the SQLAlchemy engine and ensure tables exist.

    NullPool: each query opens a fresh connection to Supabase and closes it
    immediately after. Avoids pool state issues and Supabase rate-limiting
    caused by rapid sequential queries on a shared connection.
    """

    engine = create_engine(
        settings.postgres_dsn,
        future=True,
        poolclass=NullPool,
        connect_args={
            "connect_timeout": 15,
            "options": "-c statement_timeout=55000",
            "keepalives": 1,
            "keepalives_idle": 5,
            "keepalives_interval": 3,
            "keepalives_count": 3,
        },
    )
    metadata.create_all(engine)
    return engine

