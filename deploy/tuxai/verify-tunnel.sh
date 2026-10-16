#!/usr/bin/env bash
# Probe tunnel hostnames (Cloudflare Access + Qdrant api-key) from your Mac.
#   export RAGRUN_QDRANT_URL=https://qdrant.ragxxx.com
#   export RAGRUN_EMBEDDINGS_BASE_URL=https://embed.ragxxx.com
#   export RAGRUN_QDRANT_API_KEY=...
#   export RAGRUN_CF_ACCESS_CLIENT_ID=...   # staging-tuxai or prod-tuxai
#   export RAGRUN_CF_ACCESS_CLIENT_SECRET=...
#   ./deploy/tuxai/verify-tunnel.sh
set -euo pipefail

qurl="${RAGRUN_QDRANT_URL:?RAGRUN_QDRANT_URL}"
eurl="${RAGRUN_EMBEDDINGS_BASE_URL:?RAGRUN_EMBEDDINGS_BASE_URL}"
qkey="${RAGRUN_QDRANT_API_KEY:?RAGRUN_QDRANT_API_KEY}"
cid="${RAGRUN_CF_ACCESS_CLIENT_ID:-}"
csec="${RAGRUN_CF_ACCESS_CLIENT_SECRET:-}"

cf_args=()
if [[ -n "$cid" && -n "$csec" ]]; then
  cf_args=(
    -H "CF-Access-Client-Id: $cid"
    -H "CF-Access-Client-Secret: $csec"
  )
fi

echo "== Qdrant $qurl/healthz =="
curl -fsS "${cf_args[@]}" -H "api-key: $qkey" "$qurl/healthz"
echo

embed_health="${eurl%/}/api/v1/health/simple"
echo "== Embed $embed_health =="
curl -fsS "${cf_args[@]}" "$embed_health"
echo
echo "OK"
