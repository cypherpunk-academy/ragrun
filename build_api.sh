#!/usr/bin/env bash
set -euo pipefail

# Auf dem TUXAI laufen Prod-Qdrant, Staging-Qdrant und der Embedding-Dienst: kein ragkeep, kein ragrun-api.
if [[ "$(hostname -s)" == "TUXAI" ]]; then
  cd "$(dirname "$0")"
  ENV_FILE="${TUXAI_ENV_FILE:-$HOME/ragrun-server/.env}"
  require_tuxai_env() {
    local key=$1
    local line
    line="$(grep -E "^${key}=" "$ENV_FILE" | head -n 1 || true)"
    if [[ -z "$line" || -z "${line#${key}=}" ]]; then
      echo "ERROR: ${key} fehlt oder ist leer in ${ENV_FILE}" >&2
      exit 1
    fi
  }
  for key in QDRANT_API_KEY QDRANT_STORAGE_DIR QDRANT_STORAGE_DIR_STAGING; do
    require_tuxai_env "$key"
  done
  # -p ragrun-server behält Projektnamen, Netz und das Modell-Volume der bestehenden Installation.
  DC=(docker compose -p ragrun-server --env-file "$ENV_FILE" -f deploy/tuxai/compose.yaml)
  "${DC[@]}" build embedding-service
  "${DC[@]}" up -d qdrant qdrant-staging embedding-service
  qkey="$(grep -E '^QDRANT_API_KEY=' "$ENV_FILE" | head -n 1 | cut -d= -f2-)"
  echo "Warte auf den Embedding-Dienst (Modell lädt, kann 1-2 Minuten dauern) ..."
  for _ in $(seq 1 60); do
    if curl -fs http://127.0.0.1:8001/api/v1/health/simple >/dev/null; then
      echo "embedding-service: healthy"
      if curl -fs -H "api-key: $qkey" http://127.0.0.1:6333/healthz >/dev/null; then
        echo "qdrant (prod :6333): healthy"
      else
        echo "WARNUNG: qdrant :6333 nicht healthy" >&2
      fi
      if curl -fs -H "api-key: $qkey" http://127.0.0.1:6334/healthz >/dev/null; then
        echo "qdrant-staging (:6334): healthy"
      else
        echo "WARNUNG: qdrant-staging :6334 nicht healthy" >&2
      fi
      exit 0
    fi
    sleep 5
  done
  echo "WARNUNG: embedding-service meldet nach 5 Minuten noch nicht healthy" >&2
  "${DC[@]}" logs --tail 20 embedding-service >&2
  exit 1
fi

# Init ragkeep submodule only when missing — do not --remote-update: that resets
# ragkeep to origin/main and fails on local edits (products, manifests, chunk caches).
if [[ ! -e "ragkeep/.git" ]]; then
  if ! git submodule update --init ragkeep; then
    echo "ERROR: git submodule update --init ragkeep failed; aborting." >&2
    exit 1
  fi
fi

docker compose build --no-cache ragrun-api
docker compose up -d