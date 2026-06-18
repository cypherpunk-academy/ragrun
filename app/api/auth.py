"""Supabase JWT authentication for ragapp /app/* routes."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import PyJWKClient

from app.config import settings

_bearer = HTTPBearer(auto_error=False)

_JWT_ALGORITHMS = ["RS256", "ES256", "EdDSA", "HS256"]


@dataclass(frozen=True, slots=True)
class AuthUser:
    """Authenticated Supabase user from JWT claims."""

    user_id: str
    email: str | None
    raw_token: str


@lru_cache(maxsize=1)
def _jwks_client() -> PyJWKClient | None:
    base = (settings.supabase_url or "").strip().rstrip("/")
    if not base:
        return None
    return PyJWKClient(
        f"{base}/auth/v1/.well-known/jwks.json",
        cache_keys=True,
        lifespan=300,
    )


def _decode_supabase_jwt(token: str) -> dict:
    secret = (settings.supabase_jwt_secret or "").strip()
    decode_options = {"require": ["sub", "exp"]}

    try:
        if secret:
            return jwt.decode(
                token,
                secret,
                algorithms=["HS256"],
                audience="authenticated",
                options=decode_options,
            )

        jwks = _jwks_client()
        if jwks is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "Supabase auth not configured: set RAGRUN_SUPABASE_URL "
                    "(or EXPO_PUBLIC_SUPABASE_URL) or RAGRUN_SUPABASE_JWT_SECRET"
                ),
            )

        signing_key = jwks.get_signing_key_from_jwt(token)
        return jwt.decode(
            token,
            signing_key.key,
            algorithms=_JWT_ALGORITHMS,
            audience="authenticated",
            options=decode_options,
        )
    except HTTPException:
        raise
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
        ) from exc
    except jwt.InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        ) from exc


def parse_bearer_token(token: str) -> AuthUser:
    claims = _decode_supabase_jwt(token)
    sub = claims.get("sub")
    if not isinstance(sub, str) or not sub.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing sub claim",
        )
    email = claims.get("email")
    return AuthUser(
        user_id=sub.strip(),
        email=email if isinstance(email, str) else None,
        raw_token=token,
    )


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> AuthUser:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return parse_bearer_token(credentials.credentials)
