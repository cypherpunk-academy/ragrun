"""Pull finished ragscribe transcripts into inbox/.

DeepSeek chooses only the filename (author + topic). The transcript text is stored as-is.
A job already present in inbox/ (line `Job: <uuid>`) is skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

COLLECTION = "ragscribe"
INBOX = Path(__file__).resolve().parents[1] / "inbox"
SSH_HOST = os.environ.get("TUXAI_SSH_HOST") or "tuxai"
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_CHAT_MODEL") or "deepseek-chat"

_UMLAUT = str.maketrans(
    {
        "ä": "ae",
        "ö": "oe",
        "ü": "ue",
        "Ä": "Ae",
        "Ö": "Oe",
        "Ü": "Ue",
        "ß": "ss",
    }
)
_JOB_ID = re.compile(
    r"^[-*]?\s*Job:\s*([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})\s*$",
    re.MULTILINE,
)

_REMOTE = r"""
import json, os, urllib.request
key = None
with open(os.path.expanduser("~/ragrun-server/.env")) as f:
    for line in f:
        line = line.strip()
        if line.startswith("QDRANT_API_KEY="):
            key = line.split("=", 1)[1].strip().strip("\"'")
            break
if not key:
    raise SystemExit("QDRANT_API_KEY fehlt in ~/ragrun-server/.env")

def post(path, body):
    req = urllib.request.Request(
        "http://127.0.0.1:6333" + path,
        data=json.dumps(body).encode(),
        headers={"api-key": key, "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.load(resp)

offset = None
rows = []
while True:
    body = {"limit": 100, "with_payload": True, "with_vector": False}
    if offset is not None:
        body["offset"] = offset
    page = post("/collections/__COLLECTION__/points/scroll", body)
    res = page.get("result") or {}
    for p in res.get("points") or []:
        pl = p.get("payload") or {}
        rows.append({
            "id": str(p.get("id")),
            "created_at": pl.get("created_at") or "",
            "status": pl.get("status") or "",
            "source": pl.get("source") or "",
            "url": pl.get("url") or "",
            "text": pl.get("text") or "",
        })
    offset = res.get("next_page_offset")
    if not offset:
        break
print(json.dumps(rows, ensure_ascii=False))
""".replace("__COLLECTION__", COLLECTION)


def known_job_ids(inbox: Path) -> set[str]:
    found: set[str] = set()
    if not inbox.is_dir():
        return found
    for path in inbox.glob("*.md"):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        found.update(m.group(1).lower() for m in _JOB_ID.finditer(text))
    return found


def is_ready(job: dict[str, Any]) -> bool:
    return (job.get("status") or "") == "ok" and bool((job.get("text") or "").strip())


def select_new(jobs: list[dict[str, Any]], known: set[str]) -> list[dict[str, Any]]:
    fresh = []
    for job in jobs:
        job_id = str(job.get("id") or "").lower()
        if not job_id or job_id in known or not is_ready(job):
            continue
        fresh.append(job)
    fresh.sort(key=lambda j: j.get("created_at") or "")
    return fresh


def slug_part(value: str, fallback: str) -> str:
    text = (value or "").strip().translate(_UMLAUT)
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^A-Za-z0-9_-]", "", text)
    text = re.sub(r"_+", "_", text).strip("_-")
    return (text or fallback)[:80]


def filename_for(author: str, topic: str) -> str:
    return f"{slug_part(author, 'Unbekannt')}__{slug_part(topic, 'Transkript')}.md"


def unique_path(inbox: Path, name: str, job_id: str) -> Path:
    path = inbox / name
    if not path.exists():
        return path
    stem = path.stem
    suffix = job_id.split("-", 1)[0]
    return inbox / f"{stem}__{suffix}{path.suffix}"


def parse_name_json(raw: str) -> tuple[str, str]:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("DeepSeek-Antwort enthält kein JSON")
    data = json.loads(text[start : end + 1])
    author = str(data.get("author") or "").strip()
    topic = str(data.get("topic") or "").strip()
    if not author or not topic:
        raise ValueError("DeepSeek-Antwort ohne author oder topic")
    return author, topic


def render_markdown(
    job: dict[str, Any],
    author: str,
    topic: str,
    *,
    title: str = "",
    page_url: str = "",
) -> str:
    lines = [
        f"# {author} — {topic}",
        "",
        f"- Autor: {author}",
        f"- Thema: {topic}",
    ]
    if title:
        lines.append(f"- Titel: {title}")
    if page_url:
        lines.append(f"- URL: {page_url}")
    lines.append(f"- Kurzlink: {job.get('url') or ''}")
    lines.append(f"- Job: {job['id']}")
    if job.get("created_at"):
        lines.append(f"- Erfasst: {job['created_at']}")
    lines.append("")
    lines.append((job.get("text") or "").strip())
    lines.append("")
    return "\n".join(lines)


def fetch_jobs(host: str = SSH_HOST) -> list[dict[str, Any]]:
    proc = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=10", host, "python3", "-"],
        input=_REMOTE,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()
        raise RuntimeError(f"Qdrant-Abfrage auf {host} fehlgeschlagen: {detail}")
    return json.loads(proc.stdout)


def video_meta(url: str) -> dict[str, str]:
    if not url:
        return {}
    try:
        proc = subprocess.run(
            ["yt-dlp", "--no-download", "--no-warnings", "-J", url],
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {}
    if proc.returncode != 0 or not proc.stdout.strip():
        return {}
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {}
    return {
        "channel": str(data.get("channel") or data.get("uploader") or ""),
        "title": str(data.get("title") or ""),
        "description": str(data.get("description") or "")[:500],
        "page_url": str(data.get("webpage_url") or ""),
    }


def _meta_block(meta: dict[str, str], url: str) -> str:
    if not meta:
        return f"URL: {url}\nKeine Video-Metadaten."
    return "\n".join(
        [
            f"Kanal: {meta.get('channel') or ''}",
            f"Titel: {meta.get('title') or ''}",
            f"Beschreibung: {meta.get('description') or ''}",
            f"URL: {meta.get('page_url') or url}",
        ]
    )


def propose_name(text: str, meta_block: str, api_key: str) -> tuple[str, str]:
    excerpt = text.strip()
    if len(excerpt) > 4000:
        excerpt = excerpt[:4000]
    user = (
        "Benenne dieses Transkript. Antworte ausschließlich mit JSON:\n"
        '{"author":"Vor- und Nachname","topic":"kurzes Thema auf Deutsch"}\n\n'
        "author: die Person, deren Aussage der Text ist. Bei einem Interview der "
        "Befragte, nicht der Fragesteller und nicht der TikTok-Kanal, wenn der nur "
        "weiterverbreitet.\n"
        "topic: zwei bis sechs Wörter, deutsch, ohne Punkt.\n\n"
        f"{meta_block}\n\nTranskript:\n{excerpt}\n"
    )
    body = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "Du benennst Transkripte. Antworte nur mit JSON.",
            },
            {"role": "user", "content": user},
        ],
        "temperature": 0.1,
        "max_tokens": 200,
    }
    req = urllib.request.Request(
        DEEPSEEK_URL,
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.load(resp)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"DeepSeek HTTP {exc.code}: {detail}") from exc
    content = (
        ((payload.get("choices") or [{}])[0].get("message") or {}).get("content")
        or ""
    )
    return parse_name_json(content)


def deepseek_api_key() -> str:
    key = (
        os.environ.get("DEEPSEEK_REST_API_KEY")
        or os.environ.get("RAGRUN_DEEPSEEK_API_KEY")
        or ""
    ).strip()
    if key.startswith("${") and key.endswith("}"):
        key = (os.environ.get(key[2:-1]) or "").strip()
    return key


def retrieve(inbox: Path, *, dry_run: bool = False, host: str = SSH_HOST) -> int:
    jobs = fetch_jobs(host)
    fresh = select_new(jobs, known_job_ids(inbox))
    if not fresh:
        print(f"Keine neuen Transkripte ({len(jobs)} in {COLLECTION}).")
        return 0
    print(f"{len(fresh)} neue Transkripte.")
    if dry_run:
        for job in fresh:
            print(f"- {job['id']}  {job.get('url') or ''}")
        return 0

    api_key = deepseek_api_key()
    if not api_key:
        print(
            "DEEPSEEK_REST_API_KEY fehlt. Der Dateiname kommt von DeepSeek.",
            file=sys.stderr,
        )
        return 1

    inbox.mkdir(parents=True, exist_ok=True)
    failed = 0
    for job in fresh:
        meta = video_meta(job.get("url") or "")
        try:
            author, topic = propose_name(
                job.get("text") or "",
                _meta_block(meta, job.get("url") or ""),
                api_key,
            )
        except (RuntimeError, ValueError, json.JSONDecodeError, urllib.error.URLError) as exc:
            print(f"Benennen fehlgeschlagen ({job['id']}): {exc}", file=sys.stderr)
            failed += 1
            continue
        path = unique_path(inbox, filename_for(author, topic), str(job["id"]))
        path.write_text(
            render_markdown(
                job,
                author,
                topic,
                title=meta.get("title") or "",
                page_url=meta.get("page_url") or "",
            ),
            encoding="utf-8",
        )
        print(path.relative_to(inbox.parent))
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Neue ragscribe-Transkripte aus Qdrant in die inbox legen."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Neue Jobs nur auflisten, nichts benennen und nichts schreiben.",
    )
    parser.add_argument(
        "--inbox",
        type=Path,
        default=INBOX,
        help="Zielordner (Vorgabe: ragscribe/inbox).",
    )
    args = parser.parse_args(argv)
    try:
        return retrieve(args.inbox, dry_run=args.dry_run)
    except (RuntimeError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
