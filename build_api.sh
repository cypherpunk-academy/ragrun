#!/usr/bin/env bash
set -euo pipefail

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