#!/usr/bin/env bash
# Production APK (internal distribution) for https://scribe.ragxxx.com.
# Extra args are passed through to eas, e.g. ./build-app.sh --non-interactive
set -euo pipefail
cd "$(dirname "$0")/mobile"

if ! command -v eas >/dev/null 2>&1; then
  echo "eas CLI fehlt. Install: npm install -g eas-cli" >&2
  exit 1
fi

exec eas build -p android --profile production "$@"
