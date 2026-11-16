#!/usr/bin/env bash
# Deploy ragscribe to Tuxai from a committed revision of the synced trees.
#   ./deploy/tuxai/ragscribe/push.sh
# SSH host: TUXAI_SSH_HOST (default tuxai).
# Health wait: RAGSCRIBE_HEALTH_WAIT seconds (default 180).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
HOST="${TUXAI_SSH_HOST:-tuxai}"
HEALTH_WAIT="${RAGSCRIBE_HEALTH_WAIT:-180}"

cd "$ROOT"

dirty="$(git status --porcelain -- ragscribe ':!ragscribe/mobile' deploy/tuxai/ragscribe)"
if [[ -n "$dirty" ]]; then
  echo "Uncommitted changes in the trees that get deployed:" >&2
  echo "$dirty" >&2
  exit 1
fi

rev="$(git rev-parse --short HEAD)"
echo "Deploying ragscribe $rev → $HOST"

rsync -avz --delete \
  --exclude mobile/ \
  --exclude '__pycache__/' \
  --exclude '.venv/' \
  --exclude '*.pyc' \
  --exclude DEPLOYED_REV \
  ragscribe/ "$HOST:~/ragscribe/"

rsync -avz deploy/tuxai/ragscribe/ "$HOST:~/ragrun-server/deploy/tuxai/ragscribe/"

ssh -o ServerAliveInterval=30 "$HOST" bash -s -- "$rev" "$HEALTH_WAIT" <<'EOF'
set -euo pipefail
rev="$1"
health_wait="$2"
app="$HOME/ragscribe"
repo="$HOME/ragrun-server"
env_file="$repo/.env"

if [[ ! -f "$env_file" ]]; then
  echo "Env-Datei fehlt: $env_file" >&2
  exit 1
fi
if ! grep -Eq '^QDRANT_API_KEY=.+' "$env_file"; then
  echo "QDRANT_API_KEY fehlt oder ist leer in $env_file" >&2
  exit 1
fi

printf '%s\n' "$rev" > "$app/DEPLOYED_REV"
cd "$repo"
docker compose --env-file "$env_file" -f deploy/tuxai/ragscribe/compose.yaml up -d --build

deadline=$((SECONDS + health_wait))
while true; do
  body="$(curl -sS -m 5 http://127.0.0.1:8002/health || true)"
  echo "health: ${body:-unreachable}"
  if [[ "$body" =~ \"ready\"[[:space:]]*:[[:space:]]*true ]]; then
    echo "ragscribe $rev ready"
    exit 0
  fi
  if (( SECONDS >= deadline )); then
    echo "health not ready after ${health_wait}s (rev $rev)" >&2
    exit 1
  fi
  sleep 5
done
EOF
