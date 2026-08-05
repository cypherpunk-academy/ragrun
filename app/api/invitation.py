"""Invitation API: send and redeem invitation codes."""
from __future__ import annotations

import logging
import re
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

from app.api.auth import AuthUser, get_current_user
from app.api.limiter import limit
from app.db.session import get_engine
from app.services import invitation_service, supabase_admin

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/app/invitations", tags=["invitations"])

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class SendRequest(BaseModel):
    invitee_email: str


class SendResponse(BaseModel):
    sent: bool


class RedeemRequest(BaseModel):
    email: str
    code: str


class RedeemResponse(BaseModel):
    redeemed: bool
    email_otp: str | None = None


class CheckEmailRequest(BaseModel):
    email: str


class CheckEmailResponse(BaseModel):
    exists: bool


@router.post("/send", response_model=SendResponse)
@limit("6/hour")
async def send_invitation(
    body: SendRequest,
    request: Request,
    user: Annotated[AuthUser, Depends(get_current_user)],
) -> SendResponse:
    """Send an invitation to a new user (JWT-protected, rate-limited)."""
    email = body.invitee_email.lower().strip()
    if not _EMAIL_RE.match(email):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Ungültige E-Mail-Adresse.")

    engine = get_engine()
    try:
        code = invitation_service.create_invitation(
            engine,
            inviter_user_id=user.user_id,
            inviter_email=user.email,
            invitee_email=email,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc)) from exc

    try:
        await invitation_service.send_invitation_email(email, code)
    except Exception:
        logger.exception("Failed to send invitation email to %s", email)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Einladungs-E-Mail konnte nicht gesendet werden.",
        )

    return SendResponse(sent=True)


@router.post("/redeem", response_model=RedeemResponse)
@limit("10/hour")
async def redeem_invitation(
    body: RedeemRequest,
    request: Request,
) -> RedeemResponse:
    """Redeem an invitation code and create the Supabase user (no JWT required)."""
    email = body.email.lower().strip()
    code = body.code.strip()

    if not _EMAIL_RE.match(email):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Ungültige E-Mail-Adresse.")
    if not re.match(r"^\d{4}$", code):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Code muss 4 Ziffern haben.")

    engine = get_engine()
    try:
        invitation_service.redeem_invitation(engine, email=email, code=code)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    # Create the Supabase auth user and get an OTP for immediate login
    # (single admin/generate_link call — no rate-limit issue).
    email_otp: str | None = None
    try:
        email_otp = await supabase_admin.create_user_and_get_otp(email)
    except Exception:
        logger.exception("Failed to create Supabase user for %s", email)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Benutzer konnte nicht angelegt werden.",
        )

    return RedeemResponse(redeemed=True, email_otp=email_otp)


@router.post("/check-email", response_model=CheckEmailResponse)
@limit("20/hour")
async def check_email(
    body: CheckEmailRequest,
    request: Request,
) -> CheckEmailResponse:
    """Check if an email is already registered (fallback for shouldCreateUser)."""
    email = body.email.lower().strip()
    if not _EMAIL_RE.match(email):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Ungültige E-Mail-Adresse.")

    try:
        exists = await supabase_admin.check_user_exists(email)
    except Exception:
        logger.exception("Failed to check email %s", email)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="E-Mail-Prüfung fehlgeschlagen.",
        )

    return CheckEmailResponse(exists=exists)
