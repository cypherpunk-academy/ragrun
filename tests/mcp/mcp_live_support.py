"""Shared helpers for the Philo MCP live suite (Ebene A).

Tokens and the server URL come from the environment or a gitignored env file.
Missing tokens are minted via ``scripts/mcp_tests/mint.py``. This module never
prints secrets.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import yaml
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client
from mcp.shared.exceptions import MCPError

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "mcp_tests"
TESTDATA = ROOT / "tests" / "mcp" / "mcp_testdata.yaml"
ENV_FILES = (
    ROOT / "tests" / ".env.mcp",
    ROOT / ".env.dev",
    ROOT / ".env",
)

CITATION_FIELDS = (
    "paragraph_id",
    "band",
    "segment_kind",
    "segment_title",
    "paragraph_number",
    "ort",
    "datum",
    "zitierform",
    "return_url",
)

TEXT_CHUNK_TYPES = frozenset({"book", "secondary_book", "talk"})
CONCEPT_CHUNK_TYPES = frozenset({"begriff", "typology"})
QUOTE_CHUNK_TYPES = frozenset({"quote", "quote_explanation"})


def load_env() -> None:
    """Fill missing variables from gitignored env files. Existing values win."""
    for path in ENV_FILES:
        if not path.is_file():
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def load_testdata() -> dict[str, Any]:
    data = yaml.safe_load(TESTDATA.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"unexpected testdata in {TESTDATA}")
    return data


def mcp_url() -> str:
    raw = (os.environ.get("MCP_BASE_URL") or os.environ.get("RAGRUN_MCP_BASE_URL") or "").strip()
    if not raw:
        raise RuntimeError("MCP_BASE_URL oder RAGRUN_MCP_BASE_URL fehlt")
    raw = raw.rstrip("/")
    if raw.endswith("/mcp"):
        return raw
    return raw + "/mcp"


def token_a() -> str:
    return os.environ.get("MCP_TOKEN_USER_A", "").strip()


def token_b() -> str:
    return os.environ.get("MCP_TOKEN_USER_B", "").strip()


def ensure_tokens() -> None:
    """Mint Anton/Antonia tokens when ``MCP_TOKEN_USER_*`` are unset."""
    load_env()
    if token_a() and token_b():
        return
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.mcp_tests.mint import mint_user

    if not token_a():
        os.environ["MCP_TOKEN_USER_A"] = mint_user("a")
    if not token_b():
        os.environ["MCP_TOKEN_USER_B"] = mint_user("b")


def postgres_dsn() -> str:
    raw = os.environ.get("RAGRUN_POSTGRES_DSN") or os.environ.get("DATABASE_URL") or ""
    if not raw:
        raise RuntimeError("RAGRUN_POSTGRES_DSN fehlt")
    if raw.startswith("postgresql://"):
        return raw.replace("postgresql://", "postgresql+psycopg://", 1)
    return raw


def _run_script(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        check=False,
    )


def run_cleanup() -> str:
    proc = _run_script("test_cleanup.py")
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "test_cleanup.py failed")
    return proc.stdout.strip()


def run_seed() -> str:
    proc = _run_script("seed_fixtures.py")
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "seed_fixtures.py failed")
    return proc.stdout.strip()


def run_handoffs() -> dict[str, str]:
    proc = _run_script("test_handoffs.py")
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "test_handoffs.py failed")
    found: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key in {"handoff_id", "handoff_old", "handoff_b"} and value:
            found[key] = value
    missing = {"handoff_id", "handoff_old", "handoff_b"} - found.keys()
    if missing:
        raise RuntimeError(f"test_handoffs.py ohne {sorted(missing)}: {proc.stdout.strip()}")
    return found


def _unwrap_result(payload: Any) -> Any:
    """List-valued tools arrive as ``{"result": [...]}`` on this protocol version."""
    if isinstance(payload, dict) and set(payload) == {"result"}:
        return payload["result"]
    return payload


def parse_tool_result(result: Any) -> Any:
    structured = getattr(result, "structured_content", None)
    if structured is not None:
        return _unwrap_result(structured)
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
        return {"_raw": raw}
    return _unwrap_result(payload)


def _disconnect(exc: BaseException) -> bool:
    text = str(exc).lower()
    if any(token in text for token in ("connection closed", "sse stream ended", "connecterror", "server disconnected")):
        return True
    nested = getattr(exc, "exceptions", None)
    return bool(nested) and any(_disconnect(sub) for sub in nested)


def find_mcp_error(exc: BaseException) -> MCPError | None:
    if isinstance(exc, MCPError):
        return exc
    nested = getattr(exc, "exceptions", None)
    if nested:
        for sub in nested:
            found = find_mcp_error(sub)
            if found is not None:
                return found
    return None


def http_initialize_status(url: str, token: str | None) -> int:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "ngrok-skip-browser-warning": "true",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "philo-mcp-tests", "version": "0"},
        },
    }
    response = httpx.post(url, headers=headers, json=body, timeout=30, follow_redirects=True)
    return response.status_code


def citation_problem(citation: Any) -> str | None:
    if not isinstance(citation, dict):
        return "citation fehlt"
    missing = [name for name in CITATION_FIELDS if name not in citation]
    if missing:
        return "citation ohne " + ", ".join(missing)
    kind = citation.get("segment_kind")
    if kind not in {"kapitel", "vortrag"}:
        return f"segment_kind {kind!r}"
    zitier = str(citation.get("zitierform") or "")
    if "GA " in zitier or zitier.upper().startswith("GA"):
        return f"GA in zitierform: {zitier}"
    return None


def has_stacktrace(payload: Any) -> bool:
    blob = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return "Traceback (most recent call last)" in blob or "Traceback" in blob


class Hub:
    """One event loop and one owner task for both Streamable-HTTP clients.

    Enter and exit stay on that task. A dropped ngrok session is reopened once.
    """

    def __init__(self, url: str, token_a_value: str, token_b_value: str) -> None:
        self.url = url
        self.tokens = {"a": token_a_value, "b": token_b_value}
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, name="mcp-live", daemon=True)
        self.thread.start()
        self.clients: dict[str, Client] = {}
        self._https: dict[str, Any] = {}
        self._queue: asyncio.Queue | None = None
        self._started = threading.Event()
        self._finished = threading.Event()
        self._start_error: BaseException | None = None
        asyncio.run_coroutine_threadsafe(self._serve(), self.loop)
        if not self._started.wait(90):
            self._stop_loop()
            raise TimeoutError("MCP-Client startet nicht")
        if self._start_error is not None:
            self._stop_loop()
            raise self._start_error

    async def _serve(self) -> None:
        self._queue = asyncio.Queue()
        try:
            await self._open_both()
        except BaseException as exc:
            self._start_error = exc
            await self._aclose()
            self._started.set()
            self._finished.set()
            return
        self._started.set()
        try:
            while True:
                item = await self._queue.get()
                if item is None:
                    break
                coro, fut = item
                try:
                    result = await coro
                except BaseException as exc:
                    if not fut.done():
                        fut.set_exception(exc)
                else:
                    if not fut.done():
                        fut.set_result(result)
        finally:
            await self._aclose()
            self._finished.set()

    def _submit(self, coro: Any, timeout: float) -> Any:
        async def _put() -> Any:
            fut = self.loop.create_future()
            assert self._queue is not None
            await self._queue.put((coro, fut))
            return await fut

        return asyncio.run_coroutine_threadsafe(_put(), self.loop).result(timeout)

    async def _open_client(self, token: str) -> tuple[Client, Any]:
        http = create_mcp_http_client(
            headers={
                "Authorization": f"Bearer {token}",
                "ngrok-skip-browser-warning": "true",
            }
        )
        await http.__aenter__()
        transport = streamable_http_client(self.url, http_client=http, terminate_on_close=False)
        client = Client(transport, mode="legacy", read_timeout_seconds=150)
        try:
            await client.__aenter__()
        except BaseException:
            await http.__aexit__(None, None, None)
            raise
        return client, http

    async def _open_both(self) -> None:
        for key in ("a", "b"):
            client, http = await self._open_client(self.tokens[key])
            self.clients[key] = client
            self._https[key] = http

    async def _reopen(self, user: str) -> None:
        client = self.clients.pop(user, None)
        http = self._https.pop(user, None)
        for closer in (client, http):
            if closer is None:
                continue
            try:
                await closer.__aexit__(None, None, None)
            except BaseException:
                pass
        fresh, http_fresh = await self._open_client(self.tokens[user])
        self.clients[user] = fresh
        self._https[user] = http_fresh

    def call(self, user: str, tool: str, arguments: dict[str, Any] | None = None, timeout: float = 180) -> Any:
        async def _call() -> Any:
            try:
                return await self.clients[user].call_tool(tool, arguments or {})
            except BaseException as exc:
                if not _disconnect(exc):
                    raise
                await self._reopen(user)
                return await self.clients[user].call_tool(tool, arguments or {})

        return parse_tool_result(self._submit(_call(), timeout))

    def timed_call(
        self,
        user: str,
        tool: str,
        arguments: dict[str, Any] | None = None,
        timeout: float = 180,
    ) -> tuple[Any, float]:
        started = time.perf_counter()
        body = self.call(user, tool, arguments, timeout=timeout)
        return body, time.perf_counter() - started

    def race_updates(self, note_id: str, version: int, content_a: str, content_b: str) -> list[Any]:
        token = self.tokens["a"]

        async def _race() -> list[Any]:
            first, http_first = await self._open_client(token)
            second, http_second = await self._open_client(token)
            try:
                left, right = await asyncio.gather(
                    first.call_tool(
                        "update_work_text",
                        {"note_id": note_id, "content": content_a, "expected_version": version},
                    ),
                    second.call_tool(
                        "update_work_text",
                        {"note_id": note_id, "content": content_b, "expected_version": version},
                    ),
                )
                return [parse_tool_result(left), parse_tool_result(right)]
            finally:
                await first.__aexit__(None, None, None)
                await second.__aexit__(None, None, None)
                await http_first.__aexit__(None, None, None)
                await http_second.__aexit__(None, None, None)

        return self._submit(_race(), 180)

    async def _aclose(self) -> None:
        for key, client in list(self.clients.items()):
            try:
                await client.__aexit__(None, None, None)
            except BaseException:
                pass
            http = self._https.get(key)
            if http is not None:
                try:
                    await http.__aexit__(None, None, None)
                except BaseException:
                    pass
        self.clients.clear()
        self._https.clear()

    def _stop_loop(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=5)

    def close(self) -> None:
        if self._queue is not None and not self._finished.is_set():
            async def _stop() -> None:
                assert self._queue is not None
                await self._queue.put(None)

            try:
                asyncio.run_coroutine_threadsafe(_stop(), self.loop).result(30)
            except BaseException:
                pass
            self._finished.wait(30)
        self._stop_loop()


def db_fetchone(sql: str, params: dict[str, Any]) -> Any:
    from sqlalchemy import create_engine, text

    engine = create_engine(postgres_dsn())
    with engine.connect() as conn:
        return conn.execute(text(sql), params).first()


def db_execute(sql: str, params: dict[str, Any]) -> None:
    from sqlalchemy import create_engine, text

    engine = create_engine(postgres_dsn())
    with engine.begin() as conn:
        conn.execute(text(sql), params)
