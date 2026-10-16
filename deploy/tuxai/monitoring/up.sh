#!/usr/bin/env bash
# Startet den Monitoring-Stack. Gibt keine Geheimnisse aus.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
ENV_FILE="${TUXAI_ENV_FILE:-$HOME/ragrun-server/.env}"
KEY_FILE="$ROOT/deploy/tuxai/monitoring/qdrant_key"
COMPOSE="$ROOT/deploy/tuxai/monitoring/compose.yaml"

if [[ ! -s "$KEY_FILE" ]]; then
  echo "qdrant_key fehlt oder ist leer. Zuerst deploy/tuxai/monitoring/make-qdrant-key.sh ausführen." >&2
  exit 1
fi
if [[ ! -f "$ENV_FILE" ]]; then
  echo "Env-Datei fehlt: $ENV_FILE" >&2
  exit 1
fi
if ! grep -Eq '^GRAFANA_ADMIN_PASSWORD=.+' "$ENV_FILE"; then
  echo "GRAFANA_ADMIN_PASSWORD fehlt oder ist leer in $ENV_FILE" >&2
  exit 1
fi

cd "$ROOT"
docker compose -p ragrun-monitoring --env-file "$ENV_FILE" -f "$COMPOSE" up -d
