#!/usr/bin/env python3
"""
Create local data directories and optionally probe the embedding HTTP service.

Legacy Chroma/Pinecone setup was removed; ragrun uses Qdrant + HTTP embeddings
(see app/config.py and docker-compose).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import httpx

# Project root on path for `app.config`
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from app.config import settings  # noqa: E402


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


def create_directories(logger: logging.Logger) -> bool:
    directories = [
        "data/vector_db",
        "data/backups",
        "data/logs",
        "data/embeddings_cache",
        "config",
    ]
    try:
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.info("Created directory: %s", directory)
        return True
    except OSError as exc:
        logger.error("Failed to create directories: %s", exc)
        return False


def probe_embedding_service(logger: logging.Logger) -> bool:
    base = str(settings.embeddings_base_url).rstrip("/")
    url = f"{base}/api/v1/health/simple"
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(url)
        if response.status_code == 200:
            logger.info("Embedding service reachable: %s", url)
            return True
        logger.warning("Embedding service returned HTTP %s: %s", response.status_code, url)
        return False
    except httpx.RequestError as exc:
        logger.warning("Embedding service not reachable (%s): %s", url, exc)
        return False


def main() -> bool:
    logger = setup_logging()
    logger.info("Setting up local data directories (ragrun / Qdrant stack)...")
    ok = create_directories(logger)
    probe_embedding_service(logger)
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
