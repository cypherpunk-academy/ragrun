"""Render the corpus site and upload it to the destination R2 bucket."""
from __future__ import annotations

import json
import logging
import sys
from argparse import Namespace
from typing import Any

from app.config import settings
from app.db.session import get_engine
from app.reader.export import render_segment_html
from app.reader.html import load_book_css, load_reader_js
from app.reader.load import list_segments
from app.reader.r2 import r2_client, r2_configured, upload_bytes
from app.reader.site_sync import (
    diff_site_objects,
    keys_to_delete,
    keys_to_upload,
    md5_hex,
)
from app.reader.urls import chapter_object_key

logger = logging.getLogger(__name__)


def prefixed_key(rel: str) -> str:
    prefix = (settings.r2_prefix or "").strip().strip("/")
    rel = rel.lstrip("/")
    return f"{prefix}/{rel}" if prefix else rel


def managed_prefixes() -> list[str]:
    return [prefixed_key("books") + "/", prefixed_key("assets") + "/"]


def render_site_bodies(engine) -> tuple[dict[str, tuple[bytes, str]], int]:
    """Return object key → (body, content type), and the number of failed chapters."""
    bodies: dict[str, tuple[bytes, str]] = {
        prefixed_key("assets/book.css"): (load_book_css().encode("utf-8"), "text/css"),
        prefixed_key("assets/reader.js"): (
            load_reader_js().encode("utf-8"),
            "text/javascript",
        ),
    }
    failed = 0
    segments = list_segments(engine)
    for index, seg in enumerate(segments, start=1):
        html = render_segment_html(engine, str(seg["first_paragraph_id"]))
        if not html:
            failed += 1
            continue
        key = chapter_object_key(str(seg["source_id"]), str(seg["segment_slug"]))
        bodies[key] = (html.encode("utf-8"), "text/html; charset=utf-8")
        if index % 50 == 0:
            logger.info("rendered %d/%d chapters", index, len(segments))
    return bodies, failed


def list_remote_etags(client, bucket: str, prefixes: list[str]) -> dict[str, str]:
    found: dict[str, str] = {}
    paginator = client.get_paginator("list_objects_v2")
    for prefix in prefixes:
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents") or []:
                key = obj.get("Key")
                if not key or str(key).endswith("/"):
                    continue
                found[str(key)] = str(obj.get("ETag") or "")
    return found


def delete_keys(client, bucket: str, keys: list[str]) -> int:
    deleted = 0
    for start in range(0, len(keys), 1000):
        batch = keys[start : start + 1000]
        client.delete_objects(
            Bucket=bucket,
            Delete={"Objects": [{"Key": key} for key in batch], "Quiet": True},
        )
        deleted += len(batch)
    return deleted


def publish_site(*, strategy: str, apply: bool, prune: bool) -> dict[str, Any]:
    if strategy not in ("replace", "sync"):
        raise ValueError(f"Unknown site strategy {strategy}")
    if not r2_configured():
        raise RuntimeError(
            "R2 is not configured (RAGRUN_R2_ACCOUNT_ID, RAGRUN_R2_ACCESS_KEY_ID, "
            "RAGRUN_R2_SECRET_ACCESS_KEY, RAGRUN_R2_BUCKET)"
        )
    bucket = (settings.r2_bucket or "").strip()
    public_base = (settings.corpus_public_base_url or "").strip().rstrip("/")
    engine = get_engine()
    bodies, failed = render_site_bodies(engine)
    local_md5 = {key: md5_hex(body) for key, (body, _ctype) in bodies.items()}
    client = r2_client()
    remote = list_remote_etags(client, bucket, managed_prefixes())
    diff = diff_site_objects(local_md5, remote)
    upload_keys = keys_to_upload(strategy, diff)  # type: ignore[arg-type]
    delete = keys_to_delete(strategy, diff, prune=prune)  # type: ignore[arg-type]
    uploaded = 0
    deleted = 0
    if apply:
        if failed:
            logger.error(
                "refusing to delete site objects: %d chapters failed to render",
                failed,
            )
            delete = []
        for key in upload_keys:
            body, content_type = bodies[key]
            upload_bytes(key, body, content_type, client=client)
            uploaded += 1
            if uploaded % 50 == 0:
                logger.info("uploaded %d/%d objects", uploaded, len(upload_keys))
        if delete:
            deleted = delete_keys(client, bucket, delete)
    segments = len(bodies) - 2
    return {
        "strategy": strategy,
        "bucket": bucket,
        "public_base": public_base,
        "segments": segments,
        "failed": failed,
        "insert": len(diff["insert"]),
        "update": len(diff["update"]),
        "unchanged": len(diff["unchanged"]),
        "dest_only": len(diff["dest_only"]),
        "uploaded": uploaded,
        "deleted": deleted,
        "apply": apply,
        "prune": strategy == "replace" or prune,
    }


def run_publish_site(args: Namespace) -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(message)s")
    stats = publish_site(
        strategy=str(args.strategy),
        apply=bool(args.apply),
        prune=bool(args.prune),
    )
    print(json.dumps(stats), flush=True)
    if stats["failed"]:
        raise SystemExit(1)
