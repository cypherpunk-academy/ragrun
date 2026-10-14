#!/usr/bin/env bash
# Auf dem Mac ausführen (ragrun-qdrant auf localhost:6333, ohne API-Key).
# Lädt einen Qdrant-Snapshot auf den Pulse hoch. TUXAI_QDRANT_API_KEY ist der QDRANT_API_KEY dort.
#
#   ./deploy/tuxai/copy_collection.sh rudolf-steiner-ga
#   ./deploy/tuxai/copy_collection.sh philo-von-freisinn rudolf-steiner-ga

set -euo pipefail

copy_collection() {
  local C="$1"
  echo "== Snapshot von $C auf dem Mac erzeugen =="
  local NAME
  NAME=$(curl -s -X POST "http://localhost:6333/collections/$C/snapshots" | python3 -c "import sys,json; print(json.load(sys.stdin)['result']['name'])") || return 1
  echo "Snapshot: $NAME"
  curl -fS -o "/tmp/$C.snapshot" "http://localhost:6333/collections/$C/snapshots/$NAME" || return 1
  ls -lh "/tmp/$C.snapshot"
  echo "== Hochladen auf den Pulse =="
  curl -fS -X POST "http://100.93.243.73:6333/collections/$C/snapshots/upload?priority=snapshot" \
    -H "api-key: $TUXAI_QDRANT_API_KEY" -F "snapshot=@/tmp/$C.snapshot" || return 1
  echo
  curl -s -X POST "http://100.93.243.73:6333/collections/$C/points/count" \
    -H "api-key: $TUXAI_QDRANT_API_KEY" -H "Content-Type: application/json" -d '{"exact":true}'
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
