#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# up -d creates missing services; restart alone would not.
docker compose up -d
