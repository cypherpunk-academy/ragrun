#!/usr/bin/env bash
set -euo pipefail

if ! git submodule update --init --remote ragkeep; then
  echo "ERROR: git submodule update failed; aborting." >&2
  exit 1
fi

docker compose build --no-cache ragrun-api
docker compose up -d