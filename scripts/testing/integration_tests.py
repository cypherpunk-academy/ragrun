#!/usr/bin/env python3
"""
Lightweight connectivity checks for DeepSeek (optional) and embedding service.

Replaces the old Personal RAG Server integration suite that depended on removed
legacy modules (app.core.config, DeepSeekService, etc.).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.config import settings
from app.infra.deepseek_client import DeepSeekClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def test_deepseek_chat() -> bool:
    if not settings.deepseek_api_key:
        logger.warning("RAGRUN_DEEPSEEK_API_KEY not set; skipping DeepSeek test")
        return True
    client = DeepSeekClient(
        settings.deepseek_api_key,
        model=settings.deepseek_chat_model or "deepseek-chat",
        timeout=settings.deepseek_timeout_seconds,
        base_url=str(settings.deepseek_base_url),
    )
    reply = await client.chat(
        [{"role": "user", "content": "Say OK in one word."}],
        max_tokens=16,
    )
    logger.info("DeepSeek reply: %s", reply[:200])
    return bool(reply.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description="ragrun smoke checks")
    parser.add_argument(
        "--skip-deepseek",
        action="store_true",
        help="Do not call DeepSeek (only env check)",
    )
    args = parser.parse_args()

    logger.info("Config: embeddings_base_url=%s", settings.embeddings_base_url)

    if args.skip_deepseek:
        return 0

    try:
        ok = asyncio.run(test_deepseek_chat())
    except Exception as exc:  # pragma: no cover - manual script
        logger.error("DeepSeek test failed: %s", exc)
        return 1
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
