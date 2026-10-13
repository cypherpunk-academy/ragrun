#!/usr/bin/env bash
# Schreibt den Qdrant-Key aus der Pulse-.env in qdrant_key, ohne ihn auszugeben.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
ENV_FILE="${TUXAI_ENV_FILE:-$HOME/ragrun-server/.env}"
OUT="$ROOT/deploy/tuxai/monitoring/qdrant_key"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Env-Datei fehlt: $ENV_FILE" >&2
  exit 1
fi

line="$(grep -E '^QDRANT_API_KEY=' "$ENV_FILE" | head -n 1 || true)"
if [[ -z "$line" ]]; then
  echo "QDRANT_API_KEY fehlt in $ENV_FILE" >&2
  exit 1
fi

value="${line#QDRANT_API_KEY=}"
value="${value%$'\r'}"
if [[ "$value" == \"*\" && "$value" == *\" ]]; then
  value="${value:1:${#value}-2}"
elif [[ "$value" == \'*\' && "$value" == *\' ]]; then
  value="${value:1:${#value}-2}"
fi
if [[ -z "$value" ]]; then
  echo "QDRANT_API_KEY ist leer in $ENV_FILE" >&2
  exit 1
fi

printf '%s' "$value" > "$OUT"
unset value line
# Prometheus im Image läuft als nobody (uid 65534). Der Host-Benutzer kann die
# Datei nicht an diese uid übergeben ohne root. Modus 644 lässt „andere“ lesen,
# also auch uid 65534. 600 wäre im Container nicht lesbar.
chmod 644 "$OUT"
echo "qdrant_key geschrieben, Modus 644."
