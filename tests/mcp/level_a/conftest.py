"""Session setup for the Philo MCP Ebene A suite.

The suite talks to the live Dev server. It stays out of a plain ``pytest`` run
unless this directory is on the command line (or ``MCP_LEVEL_A=1``).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from tests.mcp.mcp_live_support import (
    Hub,
    ensure_tokens,
    load_env,
    load_testdata,
    mcp_url,
    run_cleanup,
    run_handoffs,
    run_seed,
    token_a,
    token_b,
)

load_env()

ROOT = Path(__file__).resolve().parents[3]
REPORT_DIR = ROOT / "tests" / "reports"
RESULTS: list[dict[str, str]] = []

ID_ORDER = [
    "W-01", "W-02", "W-03",
    "S-01", "S-02", "S-03", "S-04", "S-05", "S-06", "S-07", "S-08", "S-09", "S-10", "S-11", "S-12",
    "P-01", "P-02", "P-03", "P-04",
    "V-01", "V-02", "V-03", "V-04", "V-05",
    "LCT-01", "LCT-02", "LCT-03", "LCT-04",
    "G-01", "G-02", "G-03", "G-04", "G-05",
    "A-01", "A-02", "A-03", "A-04", "A-05", "A-06", "A-07", "A-08",
    "L-01", "L-02", "L-03", "L-04", "L-05",
    "T-01", "T-02", "T-03",
    "C-01", "C-02", "C-03", "C-05", "C-06", "C-07", "C-08",
    "U-01", "U-02", "U-03", "U-04", "U-05", "U-06", "U-07", "U-08",
    "H-01", "H-02", "H-03", "H-04", "H-05", "H-06", "H-07",
    "CITE-01", "CITE-02", "CITE-03",
    "X-01", "X-02", "X-04",
]


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--mcp-id", action="store", default=None, help="Nur diese Test-ID ausführen, z. B. W-01")


def _explicitly_invoked(config: pytest.Config) -> bool:
    if os.environ.get("MCP_LEVEL_A") == "1":
        return True
    args = list(config.invocation_params.args)
    return any("level_a" in str(arg) for arg in args)


def _test_id(name: str) -> str:
    match = re.match(r"test_([A-Z]+)_(\d+)", name)
    if not match:
        return name
    return f"{match.group(1)}-{match.group(2)}"


def _is_level_a(item: pytest.Item) -> bool:
    location = str(getattr(item, "path", item.fspath))
    return "level_a" in location


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    own = [item for item in items if _is_level_a(item)]
    if not own:
        return
    if not _explicitly_invoked(config):
        skip = pytest.mark.skip(reason="Ebene A läuft nur über pytest tests/mcp/level_a")
        for item in own:
            item.add_marker(skip)

    wanted = config.getoption("--mcp-id")
    deselected: list[pytest.Item] = []
    if wanted and _explicitly_invoked(config):
        key = "test_" + wanted.strip().upper().replace("-", "_")
        kept: list[pytest.Item] = []
        for item in items:
            if _is_level_a(item) and not (
                item.name == key or item.name.startswith(key + "_") or item.name.startswith(key + "[")
            ):
                deselected.append(item)
            else:
                kept.append(item)
        if deselected:
            config.hook.pytest_deselected(items=deselected)
        items[:] = kept

    def rank(item: pytest.Item) -> int:
        if item.name.startswith("test_G_02"):
            return 0
        if item.name.startswith("test_A_"):
            return 2
        return 1

    others = [item for item in items if not _is_level_a(item)]
    level = [item for item in items if _is_level_a(item)]
    level.sort(key=rank)
    items[:] = others + level


@dataclass
class Live:
    data: dict[str, Any]
    hub: Hub
    handoff_id: str
    handoff_old: str
    handoff_b: str

    @property
    def user_a(self) -> str:
        return self.data["users"]["a"]["user_id"]

    @property
    def user_b(self) -> str:
        return self.data["users"]["b"]["user_id"]

    @property
    def email_a(self) -> str:
        return self.data["users"]["a"]["email"]

    @property
    def email_b(self) -> str:
        return self.data["users"]["b"]["email"]

    @property
    def source_id(self) -> str:
        return self.data["volumes"]["source_id"]["id"]

    @property
    def source_id_bg(self) -> str:
        return self.data["volumes"]["source_id_bg"]["id"]

    @property
    def search_bg(self) -> str:
        return self.data["volumes"]["source_id_bg"]["search_term"]

    @property
    def segment(self) -> str:
        return self.data["segments"]["segment"]

    @property
    def segment_idx(self) -> str:
        return str(self.data["segments"]["segment_idx"])

    @property
    def segment_empty(self) -> str:
        return self.data["segments"]["segment_empty"]

    @property
    def para_id(self) -> str:
        return self.data["paragraph"]["para_id"]

    @property
    def preview(self) -> str:
        return self.data["paragraph"]["preview"]

    @property
    def note_a(self) -> str:
        return self.data["work_texts"]["note_a"]["note_id"]

    @property
    def essay_a(self) -> str:
        return self.data["work_texts"]["essay_a"]["note_id"]

    @property
    def note_b(self) -> str:
        return self.data["work_texts"]["note_b"]["note_id"]

    @property
    def lecture_ga(self) -> str:
        return str(self.data["lectures"]["ga"])

    @property
    def lecture_count(self) -> int:
        return int(self.data["lectures"]["lecture_count"])

    @property
    def lecture_id(self) -> str:
        return self.data["lectures"]["lecture_id"]

    @property
    def lecture_source_id(self) -> str:
        return self.data["lectures"]["source_id"]

    @property
    def volume_count(self) -> int:
        return int(self.data["volumes"]["count"])

    def call(self, user: str, tool: str, arguments: dict[str, Any] | None = None, timeout: float = 180) -> Any:
        return self.hub.call(user, tool, arguments, timeout=timeout)

    def timed_call(
        self, user: str, tool: str, arguments: dict[str, Any] | None = None, timeout: float = 180
    ) -> tuple[Any, float]:
        return self.hub.timed_call(user, tool, arguments, timeout=timeout)


@pytest.fixture(scope="session")
def server_url() -> str:
    return mcp_url()


@pytest.fixture(scope="session")
def live() -> Any:
    try:
        ensure_tokens()
    except Exception as exc:
        pytest.skip(f"Tokens nicht verfügbar: {exc}")
    if not (token_a() and token_b()):
        pytest.skip("MCP_TOKEN_USER_A und MCP_TOKEN_USER_B fehlen")
    hub: Hub | None = None
    try:
        run_cleanup()
        run_seed()
        fresh = run_handoffs()
        data = load_testdata()
        hub = Hub(mcp_url(), token_a(), token_b())
        yield Live(
            data=data,
            hub=hub,
            handoff_id=fresh["handoff_id"],
            handoff_old=fresh["handoff_old"],
            handoff_b=fresh["handoff_b"],
        )
    finally:
        if hub is not None:
            hub.close()
        run_cleanup()


@pytest.fixture
def remark(request: pytest.FixtureRequest):
    def add(text: str) -> None:
        request.node.user_properties.append(("remark", text))

    return add


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]):
    outcome = yield
    report = outcome.get_result()
    if "level_a" not in str(item.fspath):
        return
    if report.when == "setup" and report.failed:
        _record(item, report)
    elif report.when == "call":
        _record(item, report)
    elif report.when == "setup" and report.skipped and not getattr(report, "wasxfail", False):
        _record(item, report)


def _record(item: pytest.Item, report: pytest.TestReport) -> None:
    wasxfail = getattr(report, "wasxfail", False)
    if report.passed:
        result = "bestanden"
    elif report.failed and wasxfail:
        result = "fehlgeschlagen"
    elif report.failed:
        result = "fehlgeschlagen"
    elif report.skipped and wasxfail:
        result = "xfail"
    elif report.skipped:
        result = "übersprungen"
    else:
        result = report.outcome

    remarks = [value for key, value in item.user_properties if key == "remark"]
    if wasxfail and result == "xfail":
        remarks.insert(0, str(wasxfail))
    if report.failed:
        lines = [line.strip() for line in str(report.longrepr).splitlines() if line.strip()]
        if lines:
            remarks.append(lines[-1][:400])
    RESULTS.append({"id": _test_id(item.name), "result": result, "remark": _cell(" ".join(remarks))})


def _cell(text: str) -> str:
    return text.replace("|", "/").replace("\n", " ").strip()


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if not RESULTS:
        return
    if not any("level_a" in str(arg) for arg in session.config.invocation_params.args) and os.environ.get("MCP_LEVEL_A") != "1":
        return
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = date.today().isoformat()
    wanted = session.config.getoption("--mcp-id")
    name = f"ebene_a_{stamp}.md" if not wanted else f"ebene_a_{stamp}_{wanted.strip().upper()}.md"
    path = REPORT_DIR / name
    order = {test_id: index for index, test_id in enumerate(ID_ORDER)}
    rows = sorted(RESULTS, key=lambda row: order.get(row["id"], 1000))
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["result"]] = counts.get(row["result"], 0) + 1
    lines = [
        f"# Ebene A {stamp}",
        "",
        "Umgebung: Dev. Client: pytest + MCP-Python-SDK (Streamable HTTP).",
        "",
        " | ".join(f"{key}: {value}" for key, value in counts.items()),
        "",
        "| ID | Ergebnis | Bemerkung |",
        "|---|---|---|",
    ]
    for row in rows:
        lines.append(f"| {row['id']} | {row['result']} | {row['remark']} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nEbene-A-Bericht: {path}")
