#!/usr/bin/env bash
# Qdrant collection via snapshot: source → target.
#
# Mac → Tuxai (Tunnel):
#   export TUXAI_QDRANT_API_KEY=...
#   export RAGRUN_QDRANT_URL=https://qdrant.ragxxx.com
#   export RAGRUN_CF_ACCESS_CLIENT_ID=... RAGRUN_CF_ACCESS_CLIENT_SECRET=...
#   ./deploy/tuxai/copy_collection.sh philo-von-freisinn
#
# Tuxai Prod → Staging (localhost, schnell):
#   export TUXAI_QDRANT_API_KEY=...   # ~/ragrun-server/.env QDRANT_API_KEY
#   export QDRANT_SOURCE_URL=http://127.0.0.1:6333
#   export QDRANT_TARGET_URL=http://127.0.0.1:6334
#   ./deploy/tuxai/copy_collection.sh philo-von-freisinn rudolf-steiner-ga
#
# Staging via Tunnel (vom Mac):
#   export RAGRUN_QDRANT_URL=https://qdrant-staging.ragxxx.com
#   ... (CF + api-key wie oben)

set -euo pipefail

QDRANT_SOURCE="${QDRANT_SOURCE_URL:-http://localhost:6333}"
QDRANT_SOURCE="${QDRANT_SOURCE%/}"
QDRANT_TARGET="${QDRANT_TARGET_URL:-${RAGRUN_QDRANT_URL:-https://qdrant.ragxxx.com}}"
QDRANT_TARGET="${QDRANT_TARGET%/}"

cf_curl_args() {
  CF_ARGS=()
  if [[ -n "${RAGRUN_CF_ACCESS_CLIENT_ID:-}" && -n "${RAGRUN_CF_ACCESS_CLIENT_SECRET:-}" ]]; then
    CF_ARGS=(
      -H "CF-Access-Client-Id: ${RAGRUN_CF_ACCESS_CLIENT_ID}"
      -H "CF-Access-Client-Secret: ${RAGRUN_CF_ACCESS_CLIENT_SECRET}"
    )
  fi
}

api_key_args() {
  API_ARGS=()
  if [[ -n "${TUXAI_QDRANT_API_KEY:-}" ]]; then
    API_ARGS=(-H "api-key: ${TUXAI_QDRANT_API_KEY}")
  fi
}

needs_cf_for_url() {
  [[ "$1" == https://* ]]
}

qdrant_json_field() {
  local field="$1"
  python3 -c "
import json, sys
raw = sys.stdin.read()
if not raw.strip():
    print('Qdrant lieferte leere Antwort (api-key fehlt/falsch oder alte copy_collection.sh?)', file=sys.stderr)
    sys.exit(1)
try:
    data = json.loads(raw)
except json.JSONDecodeError:
    print('Kein JSON von Qdrant:', raw[:500], file=sys.stderr)
    sys.exit(1)
if data.get('status') == 'error':
    print('Qdrant error:', data.get('status'), data.get('result'), file=sys.stderr)
    sys.exit(1)
print(data['result'][field])
"
}

copy_collection() {
  local C="$1"
  echo "== Snapshot von $C ($QDRANT_SOURCE) =="
  cf_curl_args
  api_key_args
  local SRC_CF=() TGT_CF=()
  if needs_cf_for_url "$QDRANT_SOURCE"; then
    SRC_CF=("${CF_ARGS[@]}")
  fi
  if needs_cf_for_url "$QDRANT_TARGET"; then
    TGT_CF=("${CF_ARGS[@]}")
  fi
  local NAME
  NAME=$(
    curl -sS -f -X POST "${SRC_CF[@]}" "${API_ARGS[@]}" \
      "$QDRANT_SOURCE/collections/$C/snapshots" | qdrant_json_field name
  ) || return 1
  echo "Snapshot: $NAME"
  curl -fS -o "/tmp/$C.snapshot" \
    "${SRC_CF[@]}" "${API_ARGS[@]}" \
    "$QDRANT_SOURCE/collections/$C/snapshots/$NAME" || return 1
  ls -lh "/tmp/$C.snapshot"
  echo "== Hochladen nach $QDRANT_TARGET =="
  curl -fS -X POST "$QDRANT_TARGET/collections/$C/snapshots/upload?priority=snapshot" \
    "${TGT_CF[@]}" "${API_ARGS[@]}" -F "snapshot=@/tmp/$C.snapshot" || return 1
  echo
  curl -sS -X POST "$QDRANT_TARGET/collections/$C/points/count" \
    "${TGT_CF[@]}" "${API_ARGS[@]}" -H "Content-Type: application/json" -d '{"exact":true}'
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
