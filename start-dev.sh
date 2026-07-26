#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure bind-mount directories exist
mkdir -p data/vector_db data/backups data/logs data/embeddings_cache
mkdir -p personal-embeddings-service/models personal-embeddings-service/logs

docker compose -f docker-compose.yml -f docker-compose.override.yml up -d

echo "Services starting..."
echo "Check: docker compose ps"
echo "Logs:  docker compose logs -f rag-server"