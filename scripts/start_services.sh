#!/usr/bin/env bash
set -euo pipefail

# Starts both the Personal Embeddings Service and the RAG API.
# Environment variables (with defaults):
#   RAGRUN_PORT       (default: 8000)
#   EMBEDDINGS_PORT   (default: 8001)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAGRUN_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EMBED_ROOT="$RAGRUN_ROOT/personal-embeddings-service"

RAGRUN_PORT="${RAGRUN_PORT:-8000}"
EMBEDDINGS_PORT="${EMBEDDINGS_PORT:-8001}"

echo "==> Starting Personal Embeddings Service on port $EMBEDDINGS_PORT"
(
  cd "$EMBED_ROOT"
  # Ensure writable caches for HF/Transformers/Sentence-Transformers
  export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
  export SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME:-$HOME/.cache/sentence-transformers}"
  # Pydantic BaseSettings will map CACHE_DIR -> settings.cache_dir
  export CACHE_DIR="${CACHE_DIR:-$HF_HOME}"
  exec python -m uvicorn app.main:app --host 0.0.0.0 --port "$EMBEDDINGS_PORT"
) &
EMBED_PID=$!

echo "==> Starting RAG API on port $RAGRUN_PORT"
(
  cd "$RAGRUN_ROOT"
  exec python -m uvicorn app.main:app --host 0.0.0.0 --port "$RAGRUN_PORT"
) &
RAGRUN_PID=$!

echo ""
echo "Personal Embeddings Service: http://localhost:$EMBEDDINGS_PORT/api/v1/health"
echo "RAG API:                     http://localhost:$RAGRUN_PORT/api/v1/health"
echo "Press Ctrl+C to stop both services."

cleanup() {
  echo "\n==> Shutting down services..."
  kill "$EMBED_PID" "$RAGRUN_PID" 2>/dev/null || true
  wait "$EMBED_PID" "$RAGRUN_PID" 2>/dev/null || true
}

trap cleanup SIGINT SIGTERM

wait "$EMBED_PID" "$RAGRUN_PID"

