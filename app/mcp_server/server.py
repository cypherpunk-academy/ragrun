"""MCP server with read-only tools, using Supabase as OAuth 2.1 Authorization Server.

The MCP SDK's streamable HTTP app is mounted into the main FastAPI app at /mcp/.
Supabase handles all OAuth (authorize, token, DCR); ragrun only verifies the
resulting JWT and serves MCP tools.
"""
from __future__ import annotations

import logging
from pathlib import Path
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

## Quellenverweise
Verwende bei Verweisen auf Werke immer den vollstaendigen deutschen Titel, wie er \
im Korpus steht (z.B. "Die Philosophie der Freiheit", nicht "PdF" oder "GA 4"). \
Bei Werken Rudolf Steiners fuege die GA-Nummer in Klammern hinzu, wenn es den Lesefluss \
nicht stoert — z.B. "Die Kernpunkte der sozialen Frage (GA 23)". \
Die GA-Nummer entspricht dem Index im Quellen-ID (z.B. source_id "...#4" = GA 4). \
Bei Nicht-Steiner-Autoren genuegt der Titel ohne GA-Nummer.

## Stil
Sprich den Nutzer immer mit "du" an. Sei klar, sachlich, praezise. Keine Ironie, \
kein Fachjargon. Vermeide Fremdwoerter, die Rudolf Steiner nicht verwendet hat.\
"""


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
            resource_server_url=f"{mcp_url}/mcp/" if mcp_url else None,
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
    ) -> list[dict]:
        """Search the philosophical corpus (books, talks, concepts, quotes).

        Returns ranked results with chunk_id, source metadata, and a short snippet.
        Use get_passage with the paragraph_id from results to read the full text.

        Args:
            query: Search query (German or English).
            types: Filter by type: "text", "concept", "quote", "chapter_summary". Default: all.
            limit: Max results (1-20, default 10).
        """
        return await search_corpus(query=query, types=types, limit=limit)

    @mcp.tool(name="get_passage")
    async def tool_get_passage(paragraph_id: str) -> dict:
        """Read a single paragraph by its UUID.

        Returns the paragraph text plus source/segment metadata for navigation.

        Args:
            paragraph_id: UUID of the paragraph (from search results or protocols).
        """
        return await get_passage(paragraph_id=paragraph_id)

    @mcp.tool(name="list_volumes")
    async def tool_list_volumes() -> list[dict]:
        """List all available books/volumes in the corpus.

        Returns source_id and display_name for each volume.
        """
        return await list_volumes()

    @mcp.tool(name="get_protocol")
    async def tool_get_protocol(source_id: str, segment_slug: str) -> dict:
        """Read the user's study protocol for a specific chapter.

        Protocols collect notes, questions, and insights per chapter.

        Args:
            source_id: Book/source ID.
            segment_slug: Chapter identifier (segment index or slug).
        """
        return await get_protocol(source_id=source_id, segment_slug=segment_slug)

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
        return await list_work_texts(text_type=text_type, limit=limit)

    @mcp.tool(name="get_work_text")
    async def tool_get_work_text(note_id: str) -> dict:
        """Read a specific work text by its ID.

        Returns title, content, version, and metadata.

        Args:
            note_id: The note ID.
        """
        return await get_work_text(note_id=note_id)

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
        return await create_work_text(
            title=title,
            content=content,
            text_type=text_type,
            paragraph_id=paragraph_id,
            conversation_url=conversation_url,
        )

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
        return await update_work_text(
            note_id=note_id,
            content=content,
            expected_version=expected_version,
            title=title,
            status=status,
        )

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
        return await append_to_protocol(
            source_id=source_id,
            segment_slug=segment_slug,
            entry_type=entry_type,
            content=content,
            paragraph_id=paragraph_id,
            conversation_url=conversation_url,
        )

    @mcp.tool(name="get_handoff")
    async def tool_get_handoff(handoff_id: str) -> dict:
        """Retrieve a handoff from the app.

        A handoff transfers context (marked text, question, paragraph) from
        the app to Claude. If expired, returns what is still available.

        Args:
            handoff_id: The 5-character handoff ID from the app deep link.
        """
        return await get_handoff(handoff_id=handoff_id)


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

    resource_url = f"{mcp_url}/mcp/"
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
