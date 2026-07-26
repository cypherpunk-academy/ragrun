"""Shared slowapi rate limiter instance."""
from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, TypeVar, get_type_hints

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

F = TypeVar("F", bound=Callable[..., Any])


def _repair_pep563_signature(fn: F) -> F:
    """Re-bind evaluated annotations so FastAPI sees real types after slowapi wrap.

    ``@limiter.limit`` + ``from __future__ import annotations`` leaves string
    annotations on ``inspect.signature``; FastAPI 0.104 then treats Pydantic
    models / ``Depends`` as query params (HTTP 422 missing body).
    """
    hints = get_type_hints(fn, include_extras=True)
    sig = inspect.signature(fn)
    params = [
        param.replace(annotation=hints.get(name, param.annotation))
        for name, param in sig.parameters.items()
    ]
    fn.__signature__ = sig.replace(  # type: ignore[attr-defined]
        parameters=params,
        return_annotation=hints.get("return", sig.return_annotation),
    )
    fn.__annotations__ = hints
    return fn


def limit(rate: str) -> Callable[[F], F]:
    """Rate-limit decorator safe to use in PEP 563 modules."""

    def decorator(func: F) -> F:
        return _repair_pep563_signature(limiter.limit(rate)(func))

    return decorator
