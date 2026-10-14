#!/usr/bin/env sh
set -eu

TS_SOCKET="${TS_SOCKET:-/var/run/tailscale/tailscaled.sock}"
TS_STATE_DIR="${TS_STATE_DIR:-/var/lib/tailscale}"
TS_SOCKS5="${TS_SOCKS5:-127.0.0.1:1055}"

start_tailscale() {
  authkey="${TS_AUTHKEY:-}"
  if [ -z "$authkey" ]; then
    return 0
  fi

  hostname="${TS_HOSTNAME:-ragrun-${RAGRUN_APP_ENV:-railway}}"
  echo "[ragrun] Starting Tailscale (hostname=${hostname})..."

  mkdir -p "$TS_STATE_DIR" /var/run/tailscale

  # Railway has no /dev/net/tun; userspace mode needs SOCKS5 so httpx reaches tailnet IPs (e.g. Tuxai).
  tailscaled \
    --state="${TS_STATE_DIR}/tailscaled.state" \
    --socket="$TS_SOCKET" \
    --tun=userspace-networking \
    --socks5-server="${TS_SOCKS5}" &
  ts_pid=$!

  # Wait for tailscaled API only — "status" fails with NeedsLogin before "up".
  i=0
  while [ "$i" -lt 60 ]; do
    if tailscale --socket="$TS_SOCKET" version >/dev/null 2>&1; then
      break
    fi
    i=$((i + 1))
    sleep 0.5
  done

  if ! tailscale --socket="$TS_SOCKET" version >/dev/null 2>&1; then
    echo "[ragrun] ERROR: tailscaled did not become ready within 30s" >&2
    kill "$ts_pid" 2>/dev/null || true
    exit 1
  fi

  # shellcheck disable=SC2086
  tailscale --socket="$TS_SOCKET" up --authkey="$authkey" --hostname="$hostname" ${TS_EXTRA_ARGS:-}

  i=0
  while [ "$i" -lt 120 ]; do
    if tailscale --socket="$TS_SOCKET" ip -4 2>/dev/null | grep -q .; then
      break
    fi
    i=$((i + 1))
    sleep 0.5
  done

  ts_ip="$(tailscale --socket="$TS_SOCKET" ip -4 2>/dev/null | head -n1 || true)"
  if [ -z "$ts_ip" ]; then
    echo "[ragrun] ERROR: Tailscale has no IPv4 address after up" >&2
    tailscale --socket="$TS_SOCKET" status >&2 || true
    exit 1
  fi

  export ALL_PROXY="socks5://${TS_SOCKS5}/"
  export HTTP_PROXY="socks5://${TS_SOCKS5}/"
  export HTTPS_PROXY="socks5://${TS_SOCKS5}/"
  # Public Supabase (JWKS, Auth API, pooler) must not use Tailscale SOCKS — PyJWKClient uses urllib.
  export NO_PROXY="localhost,127.0.0.1,::1,supabase.co,.supabase.co,.pooler.supabase.com,huggingface.co,.huggingface.co,hf.co"
  export no_proxy="${NO_PROXY}"
  echo "[ragrun] Tailscale up (${ts_ip}); outbound via socks5://${TS_SOCKS5}"
}

start_tailscale

echo "[ragrun] Running DB migrations (alembic upgrade head)..."

attempt=0
max_attempts="${RAGRUN_ALEMBIC_MAX_ATTEMPTS:-30}"
sleep_seconds="${RAGRUN_ALEMBIC_RETRY_SLEEP_SECONDS:-1}"

while [ "$attempt" -lt "$max_attempts" ]; do
  attempt=$((attempt + 1))
  if alembic upgrade head; then
    echo "[ragrun] Migrations applied."
    break
  fi

  echo "[ragrun] Alembic failed (attempt ${attempt}/${max_attempts}); retrying in ${sleep_seconds}s..." >&2
  sleep "$sleep_seconds"
done

if [ "$attempt" -ge "$max_attempts" ]; then
  echo "[ragrun] WARNING: Could not apply migrations after ${max_attempts} attempts; starting API anyway." >&2
fi

exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --log-level info
