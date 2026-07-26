#!/usr/bin/env bash
set -euo pipefail

# Init submodules only when missing — do not --remote-update: that resets ragkeep
# to origin/main and fails on local edits (products, manifests, chunk caches).
for sub in ragkeep steineroriginals; do
  if [[ ! -e "$sub/.git" ]]; then
    if ! git submodule update --init "$sub"; then
      echo "ERROR: git submodule update --init $sub failed; aborting." >&2
      exit 1
    fi
  fi
done

docker compose build --no-cache ragrun-api
docker compose up -d