"""MCP server with whoami tool, using Supabase as OAuth 2.1 Authorization Server.

The MCP SDK's streamable HTTP app is mounted into the main FastAPI app at /mcp/.
Supabase handles all OAuth (authorize, token, DCR); ragrun only verifies the
resulting JWT and serves MCP tools.
"""
from __future__ import annotations

import logging
from urllib.parse import urlparse

from mcp.server.auth.provider import AccessToken, TokenVerifier
from mcp.server.auth.settings import AuthSettings
from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.routing import Route

from app.config import settings

logger = logging.getLogger(__name__)

# Module-level reference so the parent lifespan can access it
_mcp_server: MCPServer | None = None


class SupabaseTokenVerifier(TokenVerifier):
    """Verify Supabase-issued OAuth access tokens (JWTs)."""

    async def verify_token(self, token: str) -> AccessToken | None:
        from app.api.auth import parse_bearer_token

        try:
            user = parse_bearer_token(token)
        except Exception:
            logger.debug("MCP token verification failed")
            return None

        return AccessToken(
            token=token,
            client_id="supabase",
            scopes=[],
            expires_at=None,
            subject=user.user_id,
            claims={"email": user.email},
        )


def _build_mcp_server() -> MCPServer:
    base = (settings.supabase_url or "").strip().rstrip("/")
    mcp_url = (settings.mcp_base_url or "").strip().rstrip("/")

    if not base:
        logger.warning("MCP: no SUPABASE_URL, auth disabled")

    auth = None
    token_verifier = None
    if base:
        auth = AuthSettings(
            issuer_url=f"{base}/auth/v1",
            resource_server_url=f"{mcp_url}/mcp" if mcp_url else None,
            validate_token_resource=False,
        )
        token_verifier = SupabaseTokenVerifier()

    mcp = MCPServer(
        name="philo",
        title="Philo MCP Server",
        description="MCP server for the Philo philosophical assistant",
        version="0.1.0",
        auth=auth,
        token_verifier=token_verifier,
    )

    @mcp.tool()
    async def whoami() -> dict:
        """Returns the authenticated user's ID and email."""
        from mcp.server.auth.middleware.auth_context import get_access_token

        access_token = get_access_token()
        if access_token is None:
            return {"error": "not authenticated"}

        return {
            "user_id": access_token.subject,
            "email": access_token.claims.get("email") if access_token.claims else None,
            "authenticated": True,
        }

    return mcp


def get_mcp_server() -> MCPServer:
    global _mcp_server
    if _mcp_server is None:
        _mcp_server = _build_mcp_server()
    return _mcp_server


def create_resource_metadata_route() -> Route | None:
    """Create the RFC 9728 Protected Resource Metadata route for the root app."""
    mcp_url = (settings.mcp_base_url or "").strip().rstrip("/")
    base = (settings.supabase_url or "").strip().rstrip("/")
    if not mcp_url or not base:
        return None

    from mcp.server.auth.handlers.metadata import ProtectedResourceMetadataHandler
    from mcp.server.auth.routes import build_resource_metadata_url
    from mcp.shared.auth import ProtectedResourceMetadata

    resource_url = f"{mcp_url}/mcp"
    metadata = ProtectedResourceMetadata(
        resource=resource_url,
        authorization_servers=[f"{base}/auth/v1"],
    )
    handler = ProtectedResourceMetadataHandler(metadata)
    metadata_url = build_resource_metadata_url(resource_url)
    path = urlparse(str(metadata_url)).path

    async def endpoint(request):
        return await handler.handle(request)

    return Route(path, endpoint=endpoint, methods=["GET", "OPTIONS"])


def create_mcp_app() -> Starlette:
    """Create the MCP ASGI app to be mounted at /mcp in the main FastAPI app."""
    mcp = get_mcp_server()
    mcp_url = (settings.mcp_base_url or "").strip().rstrip("/")

    # Allow the public hostname (ngrok/staging/prod) through DNS rebinding protection
    transport_security = None
    if mcp_url:
        from urllib.parse import urlparse
        host = urlparse(mcp_url).netloc
        transport_security = TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=[host],
        )

    return mcp.streamable_http_app(
        streamable_http_path="/",
        stateless_http=True,
        transport_security=transport_security,
    )
