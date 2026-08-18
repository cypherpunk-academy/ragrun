"""Engine factory and helpers."""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Sequence

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine, Row
from sqlalchemy.pool import NullPool
from sqlalchemy.sql import Selectable

from app.config import settings

from .tables import metadata

# psycopg: disable server-side prepared statements (required for Supavisor txn mode :6543).
_SUPABASE_CONNECT_ARGS = {
    "connect_timeout": 15,
    "prepare_threshold": None,
    "options": "-c statement_timeout=55000 -c idle_in_transaction_session_timeout=30000",
    "keepalives": 1,
    "keepalives_idle": 5,
    "keepalives_interval": 3,
    "keepalives_count": 3,
}


def fetch_mappings_autocommit(engine: Engine, stmt: Selectable) -> Sequence[Any]:
    """Run a read-only SELECT without leaving an idle-in-transaction session."""
    with engine.connect() as connection:
        autocommit = connection.execution_options(isolation_level="AUTOCOMMIT")
        return autocommit.execute(stmt).mappings().all()


def fetch_one_autocommit(engine: Engine, stmt: Selectable) -> Row[Any]:
    """Run a read-only aggregate SELECT in autocommit mode."""
    with engine.connect() as connection:
        autocommit = connection.execution_options(isolation_level="AUTOCOMMIT")
        return autocommit.execute(stmt).one()


def _validate_postgres_dsn(dsn: str) -> None:
    """Fail fast when the DSN is empty or contains un-encoded special characters."""
    if not dsn or not dsn.strip():
        raise RuntimeError(
            "RAGRUN_POSTGRES_DSN is empty. "
            "Check your .env — special characters like '!' must be URL-encoded (! → %21)."
        )
    # password sits between the first ':' after '://' and the '@'
    try:
        after_scheme = dsn.split("://", 1)[1]
        userinfo = after_scheme.split("@", 1)[0]
        if ":" in userinfo:
            password = userinfo.split(":", 1)[1]
            bad = [ch for ch in password if ch in "!#$ &'()+,;="]
            if bad:
                raise RuntimeError(
                    f"RAGRUN_POSTGRES_DSN password contains un-encoded characters: {bad}. "
                    f"URL-encode them in .env (e.g. ! → %21, # → %23)."
                )
    except (IndexError, ValueError):
        pass  # malformed DSN — let SQLAlchemy report the real error


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Create (or reuse) the SQLAlchemy engine and ensure tables exist.

    NullPool: each query opens a fresh connection to Supabase and closes it
    immediately after. Avoids pool state issues and Supabase rate-limiting
    caused by rapid sequential queries on a shared connection.
    """
    _validate_postgres_dsn(settings.postgres_dsn)

    engine = create_engine(
        settings.postgres_dsn,
        future=True,
        poolclass=NullPool,
        connect_args=_SUPABASE_CONNECT_ARGS,
    )
    metadata.create_all(engine)
    return engine

