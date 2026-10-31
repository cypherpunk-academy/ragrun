"""Diff static corpus pages against objects already in the destination bucket."""
from __future__ import annotations

import hashlib
from typing import Literal

SiteStrategy = Literal["replace", "sync"]


def md5_hex(body: bytes) -> str:
    return hashlib.md5(body).hexdigest()


def normalize_etag(etag: str) -> str | None:
    """S3/R2 single-part ETag is the MD5. Multipart ETags contain a hyphen."""
    value = (etag or "").strip().strip('"').lower()
    if not value or "-" in value:
        return None
    return value


def diff_site_objects(
    local_md5: dict[str, str],
    remote_etag: dict[str, str],
) -> dict[str, list[str]]:
    insert: list[str] = []
    update: list[str] = []
    unchanged: list[str] = []
    dest_only: list[str] = []
    remote_md5 = {key: normalize_etag(etag) for key, etag in remote_etag.items()}
    for key in sorted(local_md5):
        remote = remote_md5.get(key)
        if remote is None:
            insert.append(key)
        elif remote == local_md5[key]:
            unchanged.append(key)
        else:
            update.append(key)
    for key in sorted(remote_etag):
        if key not in local_md5:
            dest_only.append(key)
    return {
        "insert": insert,
        "update": update,
        "unchanged": unchanged,
        "dest_only": dest_only,
    }


def keys_to_upload(strategy: SiteStrategy, diff: dict[str, list[str]]) -> list[str]:
    if strategy == "replace":
        return diff["insert"] + diff["update"] + diff["unchanged"]
    return diff["insert"] + diff["update"]


def keys_to_delete(
    strategy: SiteStrategy,
    diff: dict[str, list[str]],
    *,
    prune: bool,
) -> list[str]:
    if strategy == "replace" or prune:
        return list(diff["dest_only"])
    return []
