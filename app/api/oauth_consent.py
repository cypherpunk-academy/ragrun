"""OAuth consent page for Supabase Auth OAuth 2.1 server.

Supabase redirects browsers to Site URL + Authorization Path after
`/auth/v1/oauth/authorize`. For local vortests that is typically:

  Site URL = http://localhost:8000
  Authorization Path = /oauth/consent
  → http://localhost:8000/oauth/consent?authorization_id=…

The page is static HTML with supabase-js (Email-OTP + Apple/Google).
Email login is OTP-only on this page (no magic-link redirect). In each
Supabase project, set **Magic Link** and **Confirm signup** templates to
use ``{{ .Token }}`` only — do not include ``{{ .ConfirmationURL }}`` in
either template (Supabase sends a link when that variable is present).
New users with **Confirm email** enabled receive **Confirm signup**, not
Magic Link; copy the same OTP HTML there. Approve/deny uses
supabase.auth.oauth.* — Auth issues the code; ragrun only hosts the UI.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["oauth"])

_TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "web" / "oauth_consent.html"
_ASSETS_DIR = Path(__file__).resolve().parent.parent / "web" / "philo"
_CONSENT_ASSETS: dict[str, str] = {
    "oauth-background.jpg": "image/jpeg",
    "avatar.png": "image/png",
}
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


@router.get("/oauth/consent/assets/{asset_name}")
def oauth_consent_asset(asset_name: str) -> FileResponse:
    """Static images for the consent page (bundled under app/web/philo/)."""
    media_type = _CONSENT_ASSETS.get(asset_name)
    if media_type is None:
        raise HTTPException(status_code=404, detail="Not found")
    path = _ASSETS_DIR / asset_name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Asset missing")
    return FileResponse(path, media_type=media_type)
