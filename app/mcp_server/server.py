"""MCP server with read-only tools, using Supabase as OAuth 2.1 Authorization Server.

The MCP SDK's streamable HTTP app is mounted into the main FastAPI app at /mcp/.
Supabase handles all OAuth (authorize, token, DCR); ragrun only verifies the
resulting JWT and serves MCP tools.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
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


def _mcp_resource_url(mcp_base: str) -> str:
    """Canonical MCP resource URL (no trailing slash).

    Cursor compares this string exactly to the URL in the MCP config.
    Claude and Cursor both register ``https://host/mcp``; advertising
    ``…/mcp/`` makes Cursor fail with a protected-resource mismatch.
    """
    return f"{mcp_base.rstrip('/')}/mcp"

# ---------------------------------------------------------------------------
# Server instructions (loaded once)
# ---------------------------------------------------------------------------

_SERVER_INSTRUCTIONS = """\
Du bist Philo von Freisinn, ein philosophischer Assistent, der ueber einen \
strukturierten Korpus von Werken Rudolf Steiners und verwandter Autoren verfuegt.

## Korpus
Der Korpus umfasst Primaerwerke (u.a. Die Philosophie der Freiheit, Die Kernpunkte \
der sozialen Frage) sowie Sekundaerwerke. Nutze `search_corpus` fuer semantische \
Suche und `get_passage` fuer den Volltext einzelner Absaetze.

## Stufenlieferung
Liefere Antworten stufenweise: Beginne kurz und praegnant. Nur wenn der Nutzer \
vertieft, liefere ausfuehrlichere Auszuege. Halte Zitate und Textauszuege knapp \
(max. 2-3 Saetze), es sei denn der Nutzer bittet ausdruecklich um mehr.

## Arbeitstexte
Der Nutzer kann eigene Arbeitstexte (Notizen, Entwuerfe) haben. Lies sie mit \
`list_work_texts` / `get_work_text`. Erstelle oder aendere Arbeitstexte NUR wenn \
der Nutzer ausdruecklich darum bittet — nutze `create_work_text` bzw. `update_work_text`. \
Bei `update_work_text` immer vorher `get_work_text` lesen, um die aktuelle Version zu kennen.

## Protokolle
Protokolle sind kapitelweise Studiennotizen des Nutzers. Lies sie mit `get_protocol`. \
Fuege neue Eintraege mit `append_to_protocol` hinzu, wenn der Nutzer darum bittet.

## Uebergaben (Handoffs)
Wenn der Nutzer eine Handoff-ID nennt, lies die Uebergabe mit `get_handoff`. \
Sie enthaelt markierten Text, eine Frage und den Absatzverweis aus der App.

## Quellenverweise (Korpus-Baende)
Wenn ein Tool ein Objekt `citation` mit Feld `zitierform` liefert, uebernimm \
diese Zeichenkette woertlich fuer den Beleg — setze sie nicht selbst zusammen \
und ergaenze keine GA-Nummer. Nutze optional `citation.return_url` als Link zur Stelle \
in Philo. Ohne `citation` zitiere Korpus-Baende mit dem vollen deutschen Titel \
(`band` / `segment_title`), ohne GA-Nummer. Ausnahme: die separate Collection \
`rudolf-steiner-ga` (Seitenangaben „GA n, S. …") — sie folgt eigenen Regeln und \
liefert kein `citation`-Objekt im Philo-Format.

## Stil
Sprich den Nutzer immer mit "du" an. Sei klar, sachlich, praezise. Keine Ironie, \
kein Fachjargon. Vermeide Fremdwoerter, die Rudolf Steiner nicht verwendet hat.\
"""


class SupabaseTokenVerifier(TokenVerifier):
    """Verify Supabase-issued OAuth access tokens (JWTs)."""

    _THROTTLE_SECONDS = 60
    _last_grant_update: dict[str, float] = {}  # user_id → monotonic timestamp

    async def verify_token(self, token: str) -> AccessToken | None:
        from app.api.auth import parse_bearer_token

        try:
            user = parse_bearer_token(token)
        except Exception:
            logger.debug("MCP token verification failed")
            return None

        self._track_grant(user.user_id)

        return AccessToken(
            token=token,
            client_id="supabase",
            scopes=[],
            expires_at=None,
            subject=user.user_id,
            claims={"email": user.email},
        )

    def _track_grant(self, user_id: str) -> None:
        """Upsert connector_grants row, throttled to max 1x per minute per user."""
        import time
        now = time.monotonic()
        last = self._last_grant_update.get(user_id, 0.0)
        if now - last < self._THROTTLE_SECONDS:
            return
        self._last_grant_update[user_id] = now
        try:
            from sqlalchemy import text as sql_text
            from app.db.session import get_engine
            engine = get_engine()
            with engine.connect() as conn:
                conn.execute(sql_text(
                    "INSERT INTO connector_grants (user_id, client_id, last_mcp_request) "
                    "VALUES (CAST(:uid AS uuid), 'supabase', now()) "
                    "ON CONFLICT (user_id, client_id) DO UPDATE "
                    "SET last_mcp_request = now(), revoked_at = NULL"
                ), {"uid": user_id})
                conn.commit()
        except Exception:
            logger.debug("Failed to track connector grant for %s", user_id, exc_info=True)


def _load_philo_voice() -> str:
    """Load the Philo personality prompt from the assistant's prompts directory."""
    prompt_path = Path(settings.assistants_root) / "philo-von-freisinn" / "prompts" / "instruction.md"
    try:
        return prompt_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        logger.warning("philo_voice prompt not found at %s", prompt_path)
        return "Du bist Philo von Freisinn, ein philosophischer Assistent."


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
            resource_server_url=_mcp_resource_url(mcp_url) if mcp_url else None,
            validate_token_resource=False,
        )
        token_verifier = SupabaseTokenVerifier()

    mcp = MCPServer(
        name="philo",
        title="Philo MCP Server",
        description="MCP server for the Philo philosophical assistant",
        instructions=_SERVER_INSTRUCTIONS,
        version="0.3.0",
        auth=auth,
        token_verifier=token_verifier,
    )

    _register_tools(mcp)
    _register_resources(mcp)
    _register_prompts(mcp)

    return mcp


def _tool_infra_error(exc: BaseException) -> dict[str, Any]:
    """Map infra failures to a structured MCP error (X-04)."""
    name = type(exc).__name__
    msg = str(exc) or name
    code = "tool_error"
    lower = msg.lower()
    if "qdrant" in lower or "Qdrant" in name:
        code = "qdrant_unavailable"
    elif "embed" in lower:
        code = "embeddings_unavailable"
    elif any(tok in lower for tok in ("postgres", "psycopg", "sqlalchemy", "connection refused", "dsn")):
        code = "db_unavailable"
    logger.exception("MCP tool failed (%s): %s", code, msg)
    return {"error": msg, "code": code}


def _register_tools(mcp: MCPServer) -> None:
    """Register all MCP tools."""
    from .tools import (
        append_to_protocol,
        create_work_text,
        get_handoff,
        get_passage,
        get_protocol,
        get_work_text,
        list_volumes,
        list_work_texts,
        search_corpus,
        update_work_text,
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

    @mcp.tool(name="search_corpus")
    async def tool_search_corpus(
        query: str,
        types: list[str] | None = None,
        limit: int = 10,
    ) -> list[dict] | dict:
        """Search the philosophical corpus (books, talks, concepts, quotes).

        Returns ranked results with chunk_id, source metadata, and a short snippet.
        Use get_passage with the paragraph_id from results to read the full text.

        Args:
            query: Search query (German or English).
            types: Filter by type: "text", "concept", "quote", "chapter_summary". Default: all.
            limit: Max results (1-20, default 10).
        """
        try:
            return await search_corpus(query=query, types=types, limit=limit)
        except Exception as exc:
            return _tool_infra_error(exc)

    @mcp.tool(name="get_passage")
    async def tool_get_passage(paragraph_id: str) -> dict:
        """Read a single paragraph by its UUID.

        Returns the paragraph text plus source/segment metadata for navigation.

        Args:
            paragraph_id: UUID of the paragraph (from search results or protocols).
        """
        try:
            return await get_passage(paragraph_id=paragraph_id)
        except Exception as exc:
            return _tool_infra_error(exc)

    @mcp.tool(name="list_volumes")
    async def tool_list_volumes() -> list[dict] | dict:
        """List all available books/volumes in the corpus.

        Returns source_id, display_name, source_type, is_primary, ga, zyklus.
        """
        try:
            return await list_volumes()
        except Exception as exc:
            return _tool_infra_error(exc)

    @mcp.tool(name="get_protocol")
    async def tool_get_protocol(source_id: str, segment_slug: str) -> dict:
        """Read the user's study protocol for a specific chapter.

        Protocols collect notes, questions, and insights per chapter.

        Args:
            source_id: Book/source ID.
            segment_slug: Chapter identifier (segment index or slug).
        """
        try:
            return await get_protocol(source_id=source_id, segment_slug=segment_slug)
        except Exception as exc:
            return _tool_infra_error(exc)

    @mcp.tool(name="list_work_texts")
    async def tool_list_work_texts(
        text_type: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """List the user's work texts (notes, drafts, essays).

        Args:
            text_type: Filter by type (e.g. "note", "essay"). Default: all.
            limit: Max results (1-50, default 20).
        """
        try:
            return await list_work_texts(text_type=text_type, limit=limit)
        except Exception as exc:
            return [_tool_infra_error(exc)]

    @mcp.tool(name="get_work_text")
    async def tool_get_work_text(note_id: str) -> dict:
        """Read a specific work text by its ID.

        Returns title, content, version, and metadata.

        Args:
            note_id: The note ID.
        """
        try:
            return await get_work_text(note_id=note_id)
        except Exception as exc:
            return _tool_infra_error(exc)

    @mcp.tool(name="create_work_text")
    async def tool_create_work_text(
        title: str,
        content: str,
        text_type: str = "note",
        paragraph_id: str | None = None,
        conversation_url: str | None = None,
    ) -> dict:
        """Create a new work text (note, draft, essay).

        Only create work texts when the user explicitly asks for it.

        Args:
            title: Title of the work text.
            content: The text content.
            text_type: Type: "note", "essay", "draft". Default: "note".
            paragraph_id: Optional paragraph UUID to link to.
            conversation_url: Optional conversation URL.
        """
        try:
            return await create_work_text(
                title=title,
                content=content,
                text_type=text_type,
                paragraph_id=paragraph_id,
                conversation_url=conversation_url,
            )
        except Exception as exc:
            return _tool_infra_error(exc)

    @mcp.tool(name="update_work_text")
    async def tool_update_work_text(
        note_id: str,
        content: str,
        expected_version: int,
        title: str | None = None,
        status: str | None = None,
    ) -> dict:
        """Update an existing work text with version control.

        Always read the note first (get_work_text) to get the current version.
        Returns {ok, new_version} or {error: "conflict"} if version changed.

        Args:
            note_id: The note ID to update.
            content: The new content.
            expected_version: Version from get_work_text (must match).
            title: Optional new title.
            status: Optional new status: "draft" or "final".
        """
        try:
            return await update_work_text(
                note_id=note_id,
                content=content,
                expected_version=expected_version,
                title=title,
                status=status,
            )
        except Exception as exc:
            return _tool_infra_error(exc)

    @mcp.tool(name="append_to_protocol")
    async def tool_append_to_protocol(
        source_id: str,
        segment_slug: str,
        entry_type: str,
        content: str,
        paragraph_id: str | None = None,
        conversation_url: str | None = None,
    ) -> dict:
        """Append a note/question/insight to the study protocol for a chapter.

        Creates the protocol automatically if it does not exist yet.

        Args:
            source_id: Book/source ID.
            segment_slug: Chapter identifier (segment index or slug).
            entry_type: Type: "note", "question", "insight", "summary".
            content: The entry text.
            paragraph_id: Optional paragraph UUID this entry refers to.
            conversation_url: Optional conversation URL.
        """
        try:
            return await append_to_protocol(
                source_id=source_id,
                segment_slug=segment_slug,
                entry_type=entry_type,
                content=content,
                paragraph_id=paragraph_id,
                conversation_url=conversation_url,
            )
        except Exception as exc:
            return _tool_infra_error(exc)

    @mcp.tool(name="get_handoff")
    async def tool_get_handoff(handoff_id: str) -> dict:
        """Retrieve a handoff from the app.

        A handoff transfers context (marked text, question, paragraph) from
        the app to Claude. If expired, returns what is still available.

        Args:
            handoff_id: The 5-character handoff ID from the app deep link.
        """
        try:
            return await get_handoff(handoff_id=handoff_id)
        except Exception as exc:
            return _tool_infra_error(exc)


def _register_resources(mcp: MCPServer) -> None:
    """Register MCP resources."""
    from .tools import list_volumes

    @mcp.resource("philo://corpus/volumes")
    async def band_list() -> str:
        """List of all volumes in the Philo corpus."""
        volumes = await list_volumes()
        lines = [f"- {v['display_name']} (source_id: {v['source_id']})" for v in volumes]
        return "# Korpus-Baende\n\n" + "\n".join(lines)


def _register_prompts(mcp: MCPServer) -> None:
    """Register MCP prompts."""

    @mcp.prompt()
    async def philo_voice() -> str:
        """The Philo von Freisinn personality and writing style instructions."""
        return _load_philo_voice()


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

    resource_url = _mcp_resource_url(mcp_url)
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
