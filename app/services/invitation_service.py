"""Invitation-based registration: code generation, DB ops, email dispatch."""
from __future__ import annotations

import logging
import secrets
from datetime import datetime, timedelta, timezone

import httpx
from sqlalchemy import select, func, update
from sqlalchemy.engine import Engine

from app.config import settings
from app.db.tables import invitations_table

logger = logging.getLogger(__name__)

MAX_INVITATIONS_PER_DAY = 20
INVITATION_EXPIRY_HOURS = 48


def generate_code() -> str:
    """Generate a random 4-digit numeric code."""
    return str(secrets.randbelow(9000) + 1000)


def count_recent_invitations(engine: Engine, inviter_user_id: str) -> int:
    """Count invitations sent by this user in the last 24 hours."""
    since = datetime.now(timezone.utc) - timedelta(hours=24)
    stmt = (
        select(func.count())
        .select_from(invitations_table)
        .where(
            invitations_table.c.inviter_user_id == inviter_user_id,
            invitations_table.c.created_at >= since,
        )
    )
    with engine.connect() as conn:
        result = conn.execute(stmt)
        return result.scalar_one()


def create_invitation(
    engine: Engine,
    *,
    inviter_user_id: str,
    inviter_email: str | None,
    invitee_email: str,
) -> str:
    """Create an invitation record and return the 4-digit code.

    Re-inviting the same email replaces any previous pending invitation
    (new code + fresh expiry; old codes become invalid).

    Raises ValueError if rate limit exceeded.
    """
    email_lower = invitee_email.lower().strip()
    code = generate_code()
    now = datetime.now(timezone.utc)
    expires = now + timedelta(hours=INVITATION_EXPIRY_HOURS)

    with engine.begin() as conn:
        existing_pending = conn.execute(
            select(invitations_table.c.id)
            .where(
                invitations_table.c.invitee_email == email_lower,
                invitations_table.c.status == "pending",
            )
            .limit(1)
        ).first()

        # Re-invites (same email) replace the previous pending row and do not
        # consume an extra daily slot.
        if existing_pending is None:
            recent = count_recent_invitations(engine, inviter_user_id)
            if recent >= MAX_INVITATIONS_PER_DAY:
                raise ValueError(f"Maximal {MAX_INVITATIONS_PER_DAY} Einladungen pro Tag erlaubt.")

        # One active invite per email: drop earlier pending attempts (incl. expired).
        conn.execute(
            invitations_table.delete().where(
                invitations_table.c.invitee_email == email_lower,
                invitations_table.c.status == "pending",
            )
        )
        conn.execute(
            invitations_table.insert().values(
                inviter_user_id=inviter_user_id,
                inviter_email=inviter_email,
                invitee_email=email_lower,
                code=code,
                status="pending",
                created_at=now,
                expires_at=expires,
            )
        )

    logger.info("Invitation created for %s by %s", email_lower, inviter_user_id)
    return code


def invitation_status_for_email(engine: Engine, email: str) -> str:
    """Return latest invitation state for an email: none | pending | expired | redeemed."""
    now = datetime.now(timezone.utc)
    email_lower = email.lower().strip()
    with engine.connect() as conn:
        row = conn.execute(
            select(
                invitations_table.c.status,
                invitations_table.c.expires_at,
            )
            .where(invitations_table.c.invitee_email == email_lower)
            .order_by(invitations_table.c.created_at.desc())
            .limit(1)
        ).first()

    if row is None:
        return "none"

    status, expires_at = row.status, row.expires_at
    if status == "redeemed":
        return "redeemed"

    if expires_at is not None:
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if expires_at <= now:
            return "expired"

    if status == "pending":
        return "pending"
    return "none"


def redeem_invitation(engine: Engine, *, email: str, code: str) -> bool:
    """Validate and redeem an invitation code.

    Returns True if redeemed successfully.
    Raises ValueError with a user-facing message on failure.
    """
    now = datetime.now(timezone.utc)
    email_lower = email.lower().strip()

    with engine.begin() as conn:
        # Atomic: only one concurrent redeem can succeed
        result = conn.execute(
            update(invitations_table)
            .where(
                invitations_table.c.invitee_email == email_lower,
                invitations_table.c.code == code,
                invitations_table.c.status == "pending",
                invitations_table.c.expires_at > now,
            )
            .values(status="redeemed", redeemed_at=now)
            .returning(invitations_table.c.id)
        )
        row = result.first()

    if row is None:
        # Check if code exists but expired
        with engine.connect() as conn:
            expired = conn.execute(
                select(invitations_table.c.id).where(
                    invitations_table.c.invitee_email == email_lower,
                    invitations_table.c.code == code,
                    invitations_table.c.expires_at <= now,
                )
            )
            if expired.first():
                raise ValueError("Der Einladungscode ist abgelaufen.")
        raise ValueError("Ungültiger Einladungscode.")

    logger.info("Invitation redeemed for %s", email_lower)
    return True


async def send_invitation_email(invitee_email: str, code: str) -> None:
    """Send the invitation email via Supabase Edge Function."""
    base = (settings.supabase_url or "").rstrip("/")
    key = settings.supabase_service_role_key
    if not base or not key:
        raise RuntimeError("supabase_url and supabase_service_role_key must be configured")

    url = f"{base}/functions/v1/send-invitation"
    payload = {
        "invitee_email": invitee_email,
        "code": code,
        "google_play_url": settings.google_play_url or "",
    }

    async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
        resp = await client.post(
            url,
            json=payload,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
        )

    if resp.status_code >= 400:
        logger.error("Edge Function send-invitation failed: %s %s", resp.status_code, resp.text)
        raise RuntimeError(f"Einladungs-E-Mail konnte nicht gesendet werden (Status {resp.status_code})")

    logger.info("Invitation email sent to %s via Edge Function", invitee_email)
