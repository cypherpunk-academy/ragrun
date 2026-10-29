"""Upload corpus HTML to Cloudflare R2 (S3-compatible)."""
from __future__ import annotations

from app.config import settings


def r2_configured() -> bool:
    return bool(
        (settings.r2_account_id or "").strip()
        and (settings.r2_access_key_id or "").strip()
        and (settings.r2_secret_access_key or "").strip()
        and (settings.r2_bucket or "").strip()
    )


def r2_client():
    import boto3

    account = (settings.r2_account_id or "").strip()
    return boto3.client(
        "s3",
        endpoint_url=f"https://{account}.r2.cloudflarestorage.com",
        aws_access_key_id=(settings.r2_access_key_id or "").strip(),
        aws_secret_access_key=(settings.r2_secret_access_key or "").strip(),
        region_name="auto",
    )


def upload_bytes(
    key: str,
    body: bytes,
    content_type: str,
    *,
    client=None,
) -> None:
    cl = client or r2_client()
    cl.put_object(
        Bucket=(settings.r2_bucket or "").strip(),
        Key=key,
        Body=body,
        ContentType=content_type,
        CacheControl="public, max-age=300",
    )
