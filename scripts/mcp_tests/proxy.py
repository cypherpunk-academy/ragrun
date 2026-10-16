#!/usr/bin/env python3
"""Stdio MCP proxy: mint a test-user JWT and forward tools to Philo Dev/Staging.

Cursor talks to this process over stdio. The proxy opens Streamable HTTP against
``{MCP_BASE_URL}/mcp`` with a freshly minted Bearer token. No token is written
to disk or to the MCP config.

Usage (from ragrun root):
  .venv/bin/python scripts/mcp_tests/proxy.py --env dev --user a
  .venv/bin/python scripts/mcp_tests/proxy.py --env staging --user antonia
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from mcp.server.mcpserver import MCPServer
from mcp.shared._httpx_utils import create_mcp_http_client

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mcp_tests.mint import (  # noqa: E402
    load_env_files,
    mint_user,
    resolve_email,
)

STAGING_DEFAULT_MCP = "https://staging.ragxxx.com"


def _mcp_url() -> str:
    import os

    raw = (os.environ.get("MCP_BASE_URL") or os.environ.get("RAGRUN_MCP_BASE_URL") or "").strip()
    if not raw:
        raise RuntimeError("MCP_BASE_URL oder RAGRUN_MCP_BASE_URL fehlt")
    raw = raw.rstrip("/")
    if raw.endswith("/mcp"):
        return raw
    return raw + "/mcp"


def _load_environment(env_name: str) -> None:
    import os

    name = env_name.strip().lower()
    if name == "dev":
        load_env_files(ROOT / ".env.dev", ROOT / ".env", ROOT / "tests" / ".env.mcp")
    elif name == "staging":
        load_env_files(ROOT / ".env.staging", ROOT / ".env", ROOT / "tests" / ".env.mcp")
        if not (os.environ.get("MCP_BASE_URL") or os.environ.get("RAGRUN_MCP_BASE_URL")):
            os.environ["RAGRUN_MCP_BASE_URL"] = STAGING_DEFAULT_MCP
    else:
        raise SystemExit(f"unbekannte Umgebung {env_name!r}; erwartet dev|staging")


def _parse_tool_result(result: Any) -> Any:
    structured = getattr(result, "structured_content", None)
    if structured is not None:
        if isinstance(structured, dict) and set(structured) == {"result"}:
            return structured["result"]
        return structured
    parts: list[str] = []
    for block in getattr(result, "content", ()) or ():
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    raw = "".join(parts).strip()
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if isinstance(payload, dict) and set(payload) == {"result"}:
        return payload["result"]
    return payload


def _make_forwarder(remote: Client, tool_name: str, schema: dict[str, Any], description: str):
    props = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    param_decls: list[str] = []
    for pname in props:
        if not pname.isidentifier():
            continue
        if pname in required:
            param_decls.append(f"{pname}: Any")
        else:
            param_decls.append(f"{pname}: Any = None")
    safe_name = tool_name if tool_name.isidentifier() else "forward_tool"
    doc = (description or tool_name).replace('"""', "'")
    keys = [pname for pname in props if pname.isidentifier()]
    dict_lit = ", ".join(f"{pname!r}: {pname}" for pname in keys)
    src = (
        f"async def {safe_name}({', '.join(param_decls)}):\n"
        f'    """{doc}"""\n'
        f"    kwargs = {{{dict_lit}}}\n"
        f"    cleaned = {{key: value for key, value in kwargs.items() if value is not None}}\n"
        f"    result = await remote.call_tool(tool_name, cleaned)\n"
        f"    return parse_result(result)\n"
    )
    ns: dict[str, Any] = {
        "Any": Any,
        "remote": remote,
        "tool_name": tool_name,
        "parse_result": _parse_tool_result,
    }
    exec(src, ns, ns)  # noqa: S102 — local tool wrapper from remote schema
    return ns[safe_name]


async def _run(env_name: str, user: str) -> None:
    _load_environment(env_name)
    email = resolve_email(user)
    token = mint_user(user)
    url = _mcp_url()
    label = f"philo-{env_name}-{user}"

    http = create_mcp_http_client(
        headers={
            "Authorization": f"Bearer {token}",
            "ngrok-skip-browser-warning": "true",
        }
    )
    await http.__aenter__()
    try:
        transport = streamable_http_client(url, http_client=http, terminate_on_close=False)
        remote = Client(transport, mode="legacy", read_timeout_seconds=150)
        await remote.__aenter__()
        try:
            listed = await remote.list_tools()
            tools = list(getattr(listed, "tools", ()) or ())
            server = MCPServer(
                name=label,
                title=f"Philo MCP ({env_name} / {email})",
                description=f"Proxy to {url} as {email}",
                instructions=(
                    f"Du sprichst den Philo-MCP-Server auf {env_name} als {email}. "
                    "Werkzeuge gehen direkt zum Server. Kein Sprachmodell dazwischen."
                ),
            )
            for tool in tools:
                schema = getattr(tool, "inputSchema", None) or getattr(tool, "input_schema", None) or {}
                if not isinstance(schema, dict):
                    schema = {}
                description = getattr(tool, "description", None) or tool.name
                forward = _make_forwarder(remote, tool.name, schema, description)
                server.add_tool(forward, name=tool.name, description=description, structured_output=False)
                registered = server._tool_manager.get_tool(tool.name)
                if registered is not None and schema:
                    registered.parameters = schema
            # stderr only — stdout is the MCP wire
            print(f"{label}: {len(tools)} tools → {url} as {email}", file=sys.stderr, flush=True)
            await server.run_stdio_async()
        finally:
            await remote.__aexit__(None, None, None)
    finally:
        await http.__aexit__(None, None, None)


def main() -> int:
    parser = argparse.ArgumentParser(description="Philo MCP stdio proxy for Cursor")
    parser.add_argument("--env", required=True, choices=("dev", "staging"))
    parser.add_argument("--user", required=True, help="a|b|anton|antonia")
    args = parser.parse_args()
    try:
        asyncio.run(_run(args.env, args.user))
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
