#!/usr/bin/env bash
# Deploy this ragscribe tree to Tuxai. Refuses uncommitted changes in the synced paths.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
exec "$root/deploy/tuxai/ragscribe/push.sh" "$@"
