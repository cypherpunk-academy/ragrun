"""OAuth consent page for Supabase Auth OAuth 2.1 server.

Supabase redirects browsers to Site URL + Authorization Path after
`/auth/v1/oauth/authorize`. For local vortests that is typically:

  Site URL = http://localhost:8000
  Authorization Path = /oauth/consent
  → http://localhost:8000/oauth/consent?authorization_id=…

The page is static HTML with supabase-js (Email-OTP + Apple/Google).
Approve/deny uses supabase.auth.oauth.* — Auth issues the code; ragrun
only hosts the UI.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["oauth"])

_TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "web" / "oauth_consent.html"
_PLACEHOLDER = "__RAGRUN_OAUTH_CONFIG__"


def _render_consent_html() -> str:
    try:
        template = _TEMPLATE_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        logger.exception("OAuth consent template missing: %s", _TEMPLATE_PATH)
        raise HTTPException(status_code=500, detail="Consent template missing") from exc

    if _PLACEHOLDER not in template:
        raise HTTPException(status_code=500, detail="Consent template misconfigured")

    config = {
        "supabaseUrl": (settings.supabase_url or "").strip().rstrip("/"),
        "supabaseAnonKey": (settings.supabase_anon_key or "").strip(),
    }
    # Safe embed into <script>: escape U+2028/U+2029 and prevent </script> breakouts.
    config_json = (
        json.dumps(config, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    return template.replace(_PLACEHOLDER, config_json, 1)


@router.get("/oauth/consent", response_class=HTMLResponse)
def oauth_consent() -> HTMLResponse:
    """Serve the Philo OAuth consent / login page."""
    return HTMLResponse(content=_render_consent_html(), status_code=200)
