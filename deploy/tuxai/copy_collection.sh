#!/usr/bin/env bash
# Auf dem Mac ausführen (ragrun-qdrant auf localhost:6333, ohne API-Key).
# Lädt einen Qdrant-Snapshot auf Tuxai hoch (Cloudflare Tunnel + Access).
#
#   export TUXAI_QDRANT_API_KEY=...   # QDRANT_API_KEY auf dem Pulse
#   export RAGRUN_QDRANT_URL=https://qdrant.ragxxx.com   # optional, Default siehe unten
#   export RAGRUN_CF_ACCESS_CLIENT_ID=... RAGRUN_CF_ACCESS_CLIENT_SECRET=...  # prod- oder staging-tuxai
#
#   ./deploy/tuxai/copy_collection.sh rudolf-steiner-ga
#   ./deploy/tuxai/copy_collection.sh philo-von-freisinn rudolf-steiner-ga

set -euo pipefail

QDRANT_REMOTE="${RAGRUN_QDRANT_URL:-https://qdrant.ragxxx.com}"
QDRANT_REMOTE="${QDRANT_REMOTE%/}"

cf_curl_args() {
  CF_ARGS=()
  if [[ -n "${RAGRUN_CF_ACCESS_CLIENT_ID:-}" && -n "${RAGRUN_CF_ACCESS_CLIENT_SECRET:-}" ]]; then
    CF_ARGS=(
      -H "CF-Access-Client-Id: ${RAGRUN_CF_ACCESS_CLIENT_ID}"
      -H "CF-Access-Client-Secret: ${RAGRUN_CF_ACCESS_CLIENT_SECRET}"
    )
  fi
}

copy_collection() {
  local C="$1"
  echo "== Snapshot von $C auf dem Mac erzeugen =="
  local NAME
  NAME=$(curl -s -X POST "http://localhost:6333/collections/$C/snapshots" | python3 -c "import sys,json; print(json.load(sys.stdin)['result']['name'])") || return 1
  echo "Snapshot: $NAME"
  curl -fS -o "/tmp/$C.snapshot" "http://localhost:6333/collections/$C/snapshots/$NAME" || return 1
  ls -lh "/tmp/$C.snapshot"
  echo "== Hochladen nach $QDRANT_REMOTE =="
  cf_curl_args
  curl -fS -X POST "$QDRANT_REMOTE/collections/$C/snapshots/upload?priority=snapshot" \
    "${CF_ARGS[@]}" -H "api-key: $TUXAI_QDRANT_API_KEY" -F "snapshot=@/tmp/$C.snapshot" || return 1
  echo
  curl -s -X POST "$QDRANT_REMOTE/collections/$C/points/count" \
    "${CF_ARGS[@]}" -H "api-key: $TUXAI_QDRANT_API_KEY" -H "Content-Type: application/json" -d '{"exact":true}'
  echo
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  : "${TUXAI_QDRANT_API_KEY:?TUXAI_QDRANT_API_KEY setzen (QDRANT_API_KEY vom Pulse)}"
  if [[ $# -lt 1 ]]; then
    echo "usage: $0 <collection> [collection...]" >&2
    exit 1
  fi
  for c in "$@"; do
    copy_collection "$c"
  done
fi
