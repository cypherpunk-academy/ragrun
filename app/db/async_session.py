"""Async SQLAlchemy engine/session helpers."""
from __future__ import annotations

from functools import lru_cache

from sqlalchemy.engine import URL, make_url
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.config import settings


def _as_async_url(dsn: str) -> str | URL:
    """Convert a sync psycopg DSN to its async variant.

    Important: avoid `str(URL)` because SQLAlchemy masks passwords (`***`) in string
    form. We must preserve the real password for engine connections.
    """

    url = make_url(dsn)
    driver = url.drivername
    if driver == "postgresql+psycopg":
        url = url.set(drivername="postgresql+psycopg_async")
    # Preserve password for actual engine connections.
    try:
        return url.render_as_string(hide_password=False)
    except TypeError:
        # Fallback for older SQLAlchemy versions.
        return url


@lru_cache(maxsize=1)
def get_async_engine() -> AsyncEngine:
    """Return a cached async engine configured for Postgres."""

    return create_async_engine(
        _as_async_url(str(settings.postgres_dsn)),
        future=True,
        pool_pre_ping=True,
    )


@lru_cache(maxsize=1)
def get_async_sessionmaker() -> sessionmaker[AsyncSession]:
    """Session factory for async DB work."""

    return sessionmaker(
        get_async_engine(),
        expire_on_commit=False,
        class_=AsyncSession,
    )


