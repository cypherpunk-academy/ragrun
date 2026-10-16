#!/usr/bin/env bash
# Schreibt rotierte Secrets an Railway, lokale .env-Dateien, GitHub und optional Tuxai.
# Werte stehen nur in scripts/secrets/rotate.env (gitignored). Dieses Skript gibt sie nie aus.
#
#   ./scripts/secrets/apply.sh              # Dry-Run (Standard)
#   ./scripts/secrets/apply.sh --apply      # wirklich schreiben
#   ./scripts/secrets/apply.sh --apply --redeploy
set -euo pipefail
if [[ -n "${TRACE:-}" ]]; then
  echo "TRACE/set -x ist für dieses Skript verboten (würde Secrets loggen)." >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SECRETS_FILE="${RAGRUN_SECRETS_FILE:-$ROOT/scripts/secrets/rotate.env}"
RAGPREP_ROOT="$(cd "$ROOT/../ragprep" 2>/dev/null && pwd || true)"
RAGAPP_ROOT="$(cd "$ROOT/../ragapp" 2>/dev/null && pwd || true)"

DRY_RUN=1
REDEPLOY=0
SKIP_DEPLOYS=1

usage() {
  cat <<'EOF'
Usage: apply.sh [--dry-run|--apply] [--redeploy] [--file PATH]

Liest scripts/secrets/rotate.env. Leere Werte werden übersprungen.
Standard ist Dry-Run: es werden nur Schlüssel und Ziele genannt, nie Werte.

  --dry-run     nur Plan (Standard)
  --apply       wirklich schreiben
  --redeploy    nach Railway-Set railway restart (sonst --skip-deploys)
  --file PATH   andere Werte-Datei
EOF
}

die() { echo "error: $*" >&2; exit 1; }
log() { echo "$*"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --apply) DRY_RUN=0; shift ;;
    --redeploy) REDEPLOY=1; SKIP_DEPLOYS=0; shift ;;
    --file) SECRETS_FILE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unbekanntes Argument: $1" ;;
  esac
done

[[ -f "$SECRETS_FILE" ]] || die "Datei fehlt: $SECRETS_FILE
Kopiere scripts/secrets/rotate.env.example nach scripts/secrets/rotate.env und trage nur neue Werte ein."

# --- Datei darf nicht im Git liegen und nicht weltlesbar sein ---
if git -C "$ROOT" ls-files --error-unmatch "$SECRETS_FILE" >/dev/null 2>&1; then
  die "$SECRETS_FILE ist von Git getrackt. Abbruch. Datei aus dem Index nehmen, .gitignore prüfen."
fi

chmod 600 "$SECRETS_FILE" 2>/dev/null || true
if stat -f '%Lp' "$SECRETS_FILE" >/dev/null 2>&1; then
  perms="$(stat -f '%Lp' "$SECRETS_FILE")"
else
  perms="$(stat -c '%a' "$SECRETS_FILE")"
fi
if [[ "$perms" != "600" && "$perms" != "400" ]]; then
  die "$SECRETS_FILE hat Rechte $perms, erwartet 600. chmod 600 und erneut."
fi

# Nur KEY=VALUE laden, keine Export-Ausgabe
set -a
# shellcheck disable=SC1090
source "$SECRETS_FILE"
set +a

APPLY_RAILWAY_STAGING="${APPLY_RAILWAY_STAGING:-0}"
APPLY_RAILWAY_PRODUCTION="${APPLY_RAILWAY_PRODUCTION:-0}"
APPLY_LOCAL_RAGRUN="${APPLY_LOCAL_RAGRUN:-0}"
APPLY_LOCAL_RAGPREP="${APPLY_LOCAL_RAGPREP:-0}"
APPLY_GITHUB="${APPLY_GITHUB:-0}"
APPLY_TUXAI="${APPLY_TUXAI:-0}"
APPLY_EAS="${APPLY_EAS:-0}"
RAILWAY_PROJECT_ID="${RAILWAY_PROJECT_ID:-4fc5cea8-9f49-42af-b624-a0f3e5232245}"
RAILWAY_PRODUCTION_SERVICE="${RAILWAY_PRODUCTION_SERVICE:-ragrun}"
RAILWAY_STAGING_SERVICE="${RAILWAY_STAGING_SERVICE:-ragrun-staging}"
GITHUB_REPO="${GITHUB_REPO:-cypherpunk-academy/ragrun}"

mode_label="DRY-RUN"
if [[ "$DRY_RUN" -eq 0 ]]; then
  mode_label="APPLY"
fi
log "== $mode_label  file=$(basename "$SECRETS_FILE")  perms=$perms =="

filled() {
  [[ -n "${1:-}" ]]
}

# Schlüssel ohne Wert — für den Plan
plan() {
  local dest="$1" key="$2" val="${3:-}"
  if ! filled "$val"; then
    return 0
  fi
  log "  $dest  $key"
}

upsert_env_file() {
  local path="$1"
  shift
  # rest: KEY=VALUE pairs via env names passed as KEY:ENVVAR
  python3 - "$path" "$@" <<'PY'
import os, stat, sys
path = sys.argv[1]
pairs = []
for spec in sys.argv[2:]:
    dest_key, src_key = spec.split(":", 1)
    val = os.environ.get(src_key, "")
    if val:
        pairs.append((dest_key, val))
if not pairs:
    sys.exit(0)
try:
    text = open(path, encoding="utf-8").read()
except FileNotFoundError:
    text = ""
lines = text.splitlines()
seen = set()
out = []
for line in lines:
    if not line or line.lstrip().startswith("#") or "=" not in line:
        out.append(line)
        continue
    k = line.split("=", 1)[0]
    replacement = None
    for dest_key, val in pairs:
        if k == dest_key:
            replacement = f"{dest_key}={val}"
            seen.add(dest_key)
            break
    out.append(replacement if replacement is not None else line)
for dest_key, val in pairs:
    if dest_key not in seen:
        out.append(f"{dest_key}={val}")
new = "\n".join(out)
if text and not text.endswith("\n"):
    pass
new += "\n"
with open(path, "w", encoding="utf-8") as fh:
    fh.write(new)
os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
print(f"  wrote {path} ({len(pairs)} keys)", file=sys.stderr)
PY
}

railway_set() {
  local env="$1" service="$2" key="$3" val="$4"
  if ! filled "$val"; then
    return 0
  fi
  plan "railway:$env/$service" "$key" "$val"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    return 0
  fi
  command -v railway >/dev/null || die "railway CLI fehlt"
  if [[ "$SKIP_DEPLOYS" -eq 0 ]]; then
    printf '%s' "$val" | railway variable set "$key" --stdin \
      -p "$RAILWAY_PROJECT_ID" -e "$env" -s "$service" >/dev/null
  else
    printf '%s' "$val" | railway variable set "$key" --stdin \
      -p "$RAILWAY_PROJECT_ID" -e "$env" -s "$service" --skip-deploys >/dev/null
  fi
}

railway_delete() {
  local env="$1" service="$2" key="$3"
  plan "railway:$env/$service" "DELETE $key" x
  if [[ "$DRY_RUN" -eq 1 ]]; then
    return 0
  fi
  command -v railway >/dev/null || die "railway CLI fehlt"
  if railway variable list -p "$RAILWAY_PROJECT_ID" -e "$env" -s "$service" --json 2>/dev/null \
    | python3 -c "import json,sys; raw=json.load(sys.stdin); keys=raw.keys() if isinstance(raw, dict) else {v['name'] for v in raw}; sys.exit(0 if sys.argv[1] in keys else 1)" "$key"; then
    railway variable delete "$key" -p "$RAILWAY_PROJECT_ID" -e "$env" -s "$service" >/dev/null
  fi
}

github_set() {
  local name="$1" val="$2"
  if ! filled "$val"; then
    return 0
  fi
  plan "github:$GITHUB_REPO" "$name" "$val"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    return 0
  fi
  command -v gh >/dev/null || die "gh CLI fehlt"
  printf '%s' "$val" | gh secret set "$name" --repo "$GITHUB_REPO"
}

restart_service() {
  local env="$1" service="$2"
  if [[ "$REDEPLOY" -eq 0 || "$DRY_RUN" -eq 1 ]]; then
    return 0
  fi
  log "  redeploy railway:$env/$service"
  if ! railway redeploy -y -p "$RAILWAY_PROJECT_ID" -e "$env" -s "$service" >/dev/null 2>&1; then
    log "  redeploy failed, trying restart"
    railway restart -y -p "$RAILWAY_PROJECT_ID" -e "$env" -s "$service" >/dev/null || true
  fi
}

# ---------- Plan / Apply ----------
log "-- Railway staging ($RAILWAY_STAGING_SERVICE) --"
if [[ "$APPLY_RAILWAY_STAGING" == "1" ]]; then
  railway_set staging "$RAILWAY_STAGING_SERVICE" RAGRUN_QDRANT_API_KEY "${QDRANT_API_KEY:-}"
  railway_set staging "$RAILWAY_STAGING_SERVICE" RAGRUN_DEEPSEEK_API_KEY "${DEEPSEEK_API_KEY:-}"
  railway_set staging "$RAILWAY_STAGING_SERVICE" RAGRUN_HF_TOKEN "${HF_TOKEN:-}"
  railway_set staging "$RAILWAY_STAGING_SERVICE" RAGRUN_INTERNAL_API_KEY "${INTERNAL_API_KEY:-}"
  railway_set staging "$RAILWAY_STAGING_SERVICE" RAGRUN_SUPABASE_ANON_KEY "${STAGING_SUPABASE_ANON_KEY:-}"
  railway_set staging "$RAILWAY_STAGING_SERVICE" RAGRUN_SUPABASE_SERVICE_ROLE_KEY "${STAGING_SUPABASE_SERVICE_ROLE_KEY:-}"
  railway_set staging "$RAILWAY_STAGING_SERVICE" SUPABASE_SERVICE_ROLE_KEY "${STAGING_SUPABASE_SERVICE_ROLE_KEY:-}"
  railway_set staging "$RAILWAY_STAGING_SERVICE" RAGRUN_SUPABASE_JWT_SECRET "${STAGING_SUPABASE_JWT_SECRET:-}"
  railway_set staging "$RAILWAY_STAGING_SERVICE" RAGRUN_POSTGRES_DSN "${STAGING_POSTGRES_DSN:-}"
  railway_set staging "$RAILWAY_STAGING_SERVICE" RAGRUN_QDRANT_URL "${TUNNEL_QDRANT_URL:-}"
  railway_set staging "$RAILWAY_STAGING_SERVICE" RAGRUN_EMBEDDINGS_BASE_URL "${TUNNEL_EMBED_URL:-}"
  if filled "${TUNNEL_QDRANT_URL:-}" || filled "${TUNNEL_EMBED_URL:-}"; then
    railway_set staging "$RAILWAY_STAGING_SERVICE" RAGRUN_EMBEDDINGS_PROVIDER "http"
  fi
  railway_set staging "$RAILWAY_STAGING_SERVICE" RAGRUN_CF_ACCESS_CLIENT_ID "${STAGING_CF_ACCESS_CLIENT_ID:-}"
  railway_set staging "$RAILWAY_STAGING_SERVICE" RAGRUN_CF_ACCESS_CLIENT_SECRET "${STAGING_CF_ACCESS_CLIENT_SECRET:-}"
  if [[ "${CLEAR_RAILWAY_TAILSCALE:-0}" == "1" ]]; then
    railway_delete staging "$RAILWAY_STAGING_SERVICE" TS_AUTHKEY
    railway_delete staging "$RAILWAY_STAGING_SERVICE" TS_HOSTNAME
  else
    railway_set staging "$RAILWAY_STAGING_SERVICE" TS_AUTHKEY "${STAGING_TS_AUTHKEY:-}"
    if filled "${STAGING_TS_AUTHKEY:-}"; then
      railway_set staging "$RAILWAY_STAGING_SERVICE" TS_HOSTNAME "ragrun-staging"
    fi
  fi
  restart_service staging "$RAILWAY_STAGING_SERVICE"
else
  log "  übersprungen"
fi

log "-- Railway production ($RAILWAY_PRODUCTION_SERVICE) --"
if [[ "$APPLY_RAILWAY_PRODUCTION" == "1" ]]; then
  railway_set production "$RAILWAY_PRODUCTION_SERVICE" RAGRUN_QDRANT_API_KEY "${QDRANT_API_KEY:-}"
  railway_set production "$RAILWAY_PRODUCTION_SERVICE" RAGRUN_DEEPSEEK_API_KEY "${DEEPSEEK_API_KEY:-}"
  railway_set production "$RAILWAY_PRODUCTION_SERVICE" RAGRUN_HF_TOKEN "${HF_TOKEN:-}"
  railway_set production "$RAILWAY_PRODUCTION_SERVICE" RAGRUN_INTERNAL_API_KEY "${INTERNAL_API_KEY:-}"
  railway_set production "$RAILWAY_PRODUCTION_SERVICE" RAGRUN_SUPABASE_ANON_KEY "${PRODUCTION_SUPABASE_ANON_KEY:-}"
  railway_set production "$RAILWAY_PRODUCTION_SERVICE" RAGRUN_SUPABASE_SERVICE_ROLE_KEY "${PRODUCTION_SUPABASE_SERVICE_ROLE_KEY:-}"
  railway_set production "$RAILWAY_PRODUCTION_SERVICE" SUPABASE_SERVICE_ROLE_KEY "${PRODUCTION_SUPABASE_SERVICE_ROLE_KEY:-}"
  railway_set production "$RAILWAY_PRODUCTION_SERVICE" RAGRUN_SUPABASE_JWT_SECRET "${PRODUCTION_SUPABASE_JWT_SECRET:-}"
  railway_set production "$RAILWAY_PRODUCTION_SERVICE" RAGRUN_POSTGRES_DSN "${PRODUCTION_POSTGRES_DSN:-}"
  railway_set production "$RAILWAY_PRODUCTION_SERVICE" RAGRUN_QDRANT_URL "${TUNNEL_QDRANT_URL:-}"
  railway_set production "$RAILWAY_PRODUCTION_SERVICE" RAGRUN_EMBEDDINGS_BASE_URL "${TUNNEL_EMBED_URL:-}"
  if filled "${TUNNEL_QDRANT_URL:-}" || filled "${TUNNEL_EMBED_URL:-}"; then
    railway_set production "$RAILWAY_PRODUCTION_SERVICE" RAGRUN_EMBEDDINGS_PROVIDER "http"
  fi
  railway_set production "$RAILWAY_PRODUCTION_SERVICE" RAGRUN_CF_ACCESS_CLIENT_ID "${PRODUCTION_CF_ACCESS_CLIENT_ID:-}"
  railway_set production "$RAILWAY_PRODUCTION_SERVICE" RAGRUN_CF_ACCESS_CLIENT_SECRET "${PRODUCTION_CF_ACCESS_CLIENT_SECRET:-}"
  if [[ "${CLEAR_RAILWAY_TAILSCALE:-0}" == "1" ]]; then
    railway_delete production "$RAILWAY_PRODUCTION_SERVICE" TS_AUTHKEY
    railway_delete production "$RAILWAY_PRODUCTION_SERVICE" TS_HOSTNAME
  else
    railway_set production "$RAILWAY_PRODUCTION_SERVICE" TS_AUTHKEY "${PRODUCTION_TS_AUTHKEY:-}"
    if filled "${PRODUCTION_TS_AUTHKEY:-}"; then
      railway_set production "$RAILWAY_PRODUCTION_SERVICE" TS_HOSTNAME "ragrun-production"
    fi
  fi
  restart_service production "$RAILWAY_PRODUCTION_SERVICE"
else
  log "  übersprungen"
fi

log "-- lokale ragrun .env --"
if [[ "$APPLY_LOCAL_RAGRUN" == "1" ]]; then
  for f in "$ROOT/.env.staging"; do
    if filled "${QDRANT_API_KEY:-}"; then plan "local:$f" RAGRUN_QDRANT_API_KEY x; fi
    if filled "${DEEPSEEK_API_KEY:-}"; then plan "local:$f" RAGRUN_DEEPSEEK_API_KEY x; fi
    if filled "${STAGING_SUPABASE_ANON_KEY:-}"; then plan "local:$f" RAGRUN_SUPABASE_ANON_KEY x; fi
    if filled "${STAGING_SUPABASE_SERVICE_ROLE_KEY:-}"; then
      plan "local:$f" RAGRUN_SUPABASE_SERVICE_ROLE_KEY x
      plan "local:$f" SUPABASE_SERVICE_ROLE_KEY x
    fi
    if filled "${STAGING_SUPABASE_JWT_SECRET:-}"; then plan "local:$f" RAGRUN_SUPABASE_JWT_SECRET x; fi
    if filled "${STAGING_POSTGRES_DSN:-}"; then plan "local:$f" RAGRUN_POSTGRES_DSN x; fi
    if filled "${TUNNEL_QDRANT_URL:-}"; then plan "local:$f" RAGRUN_QDRANT_URL x; fi
    if filled "${TUNNEL_EMBED_URL:-}"; then plan "local:$f" RAGRUN_EMBEDDINGS_BASE_URL x; fi
    if filled "${STAGING_CF_ACCESS_CLIENT_ID:-}"; then plan "local:$f" RAGRUN_CF_ACCESS_CLIENT_ID x; fi
    if filled "${STAGING_CF_ACCESS_CLIENT_SECRET:-}"; then plan "local:$f" RAGRUN_CF_ACCESS_CLIENT_SECRET x; fi
    if [[ "$DRY_RUN" -eq 0 ]]; then
      upsert_env_file "$f" \
        RAGRUN_QDRANT_API_KEY:QDRANT_API_KEY \
        RAGRUN_DEEPSEEK_API_KEY:DEEPSEEK_API_KEY \
        RAGRUN_SUPABASE_ANON_KEY:STAGING_SUPABASE_ANON_KEY \
        RAGRUN_SUPABASE_SERVICE_ROLE_KEY:STAGING_SUPABASE_SERVICE_ROLE_KEY \
        SUPABASE_SERVICE_ROLE_KEY:STAGING_SUPABASE_SERVICE_ROLE_KEY \
        RAGRUN_SUPABASE_JWT_SECRET:STAGING_SUPABASE_JWT_SECRET \
        RAGRUN_POSTGRES_DSN:STAGING_POSTGRES_DSN \
        RAGRUN_QDRANT_URL:TUNNEL_QDRANT_URL \
        RAGRUN_EMBEDDINGS_BASE_URL:TUNNEL_EMBED_URL \
        RAGRUN_CF_ACCESS_CLIENT_ID:STAGING_CF_ACCESS_CLIENT_ID \
        RAGRUN_CF_ACCESS_CLIENT_SECRET:STAGING_CF_ACCESS_CLIENT_SECRET
    fi
  done
  for f in "$ROOT/.env.production" "$ROOT/.env.prod"; do
    [[ -e "$f" || "$DRY_RUN" -eq 0 ]] || continue
    if filled "${QDRANT_API_KEY:-}"; then plan "local:$f" RAGRUN_QDRANT_API_KEY x; fi
    if filled "${DEEPSEEK_API_KEY:-}"; then plan "local:$f" RAGRUN_DEEPSEEK_API_KEY x; fi
    if filled "${PRODUCTION_SUPABASE_ANON_KEY:-}"; then
      plan "local:$f" RAGRUN_SUPABASE_ANON_KEY x
      plan "local:$f" EXPO_PUBLIC_SUPABASE_ANON_KEY x
    fi
    if filled "${PRODUCTION_SUPABASE_SERVICE_ROLE_KEY:-}"; then
      plan "local:$f" RAGRUN_SUPABASE_SERVICE_ROLE_KEY x
      plan "local:$f" SUPABASE_SERVICE_ROLE_KEY x
    fi
    if filled "${PRODUCTION_SUPABASE_JWT_SECRET:-}"; then plan "local:$f" RAGRUN_SUPABASE_JWT_SECRET x; fi
    if filled "${PRODUCTION_POSTGRES_DSN:-}"; then plan "local:$f" RAGRUN_POSTGRES_DSN x; fi
    if filled "${TUNNEL_QDRANT_URL:-}"; then plan "local:$f" RAGRUN_QDRANT_URL x; fi
    if filled "${TUNNEL_EMBED_URL:-}"; then plan "local:$f" RAGRUN_EMBEDDINGS_BASE_URL x; fi
    if filled "${PRODUCTION_CF_ACCESS_CLIENT_ID:-}"; then plan "local:$f" RAGRUN_CF_ACCESS_CLIENT_ID x; fi
    if filled "${PRODUCTION_CF_ACCESS_CLIENT_SECRET:-}"; then plan "local:$f" RAGRUN_CF_ACCESS_CLIENT_SECRET x; fi
    if [[ "$DRY_RUN" -eq 0 && -e "$f" ]]; then
      upsert_env_file "$f" \
        RAGRUN_QDRANT_API_KEY:QDRANT_API_KEY \
        RAGRUN_DEEPSEEK_API_KEY:DEEPSEEK_API_KEY \
        RAGRUN_SUPABASE_ANON_KEY:PRODUCTION_SUPABASE_ANON_KEY \
        EXPO_PUBLIC_SUPABASE_ANON_KEY:PRODUCTION_SUPABASE_ANON_KEY \
        RAGRUN_SUPABASE_SERVICE_ROLE_KEY:PRODUCTION_SUPABASE_SERVICE_ROLE_KEY \
        SUPABASE_SERVICE_ROLE_KEY:PRODUCTION_SUPABASE_SERVICE_ROLE_KEY \
        RAGRUN_SUPABASE_JWT_SECRET:PRODUCTION_SUPABASE_JWT_SECRET \
        RAGRUN_POSTGRES_DSN:PRODUCTION_POSTGRES_DSN \
        RAGRUN_QDRANT_URL:TUNNEL_QDRANT_URL \
        RAGRUN_EMBEDDINGS_BASE_URL:TUNNEL_EMBED_URL \
        RAGRUN_CF_ACCESS_CLIENT_ID:PRODUCTION_CF_ACCESS_CLIENT_ID \
        RAGRUN_CF_ACCESS_CLIENT_SECRET:PRODUCTION_CF_ACCESS_CLIENT_SECRET
    fi
  done
else
  log "  übersprungen"
fi

log "-- lokale ragprep .env --"
if [[ "$APPLY_LOCAL_RAGPREP" == "1" ]]; then
  if [[ -z "$RAGPREP_ROOT" ]]; then
    log "  ragprep-Ordner nicht gefunden, übersprungen"
  else
    if filled "${STAGING_POSTGRES_DSN:-}"; then plan "local:$RAGPREP_ROOT/.env.staging" RAGRUN_POSTGRES_DSN x; fi
    if filled "${PRODUCTION_POSTGRES_DSN:-}"; then plan "local:$RAGPREP_ROOT/.env.production" RAGRUN_POSTGRES_DSN x; fi
    if filled "${QDRANT_API_KEY:-}"; then plan "local:$RAGPREP_ROOT/.env.production" RAGRUN_QDRANT_API_KEY x; fi
    if [[ "$DRY_RUN" -eq 0 ]]; then
      [[ -f "$RAGPREP_ROOT/.env.staging" ]] && upsert_env_file "$RAGPREP_ROOT/.env.staging" \
        RAGRUN_POSTGRES_DSN:STAGING_POSTGRES_DSN
      [[ -f "$RAGPREP_ROOT/.env.production" ]] && upsert_env_file "$RAGPREP_ROOT/.env.production" \
        RAGRUN_POSTGRES_DSN:PRODUCTION_POSTGRES_DSN \
        RAGRUN_QDRANT_API_KEY:QDRANT_API_KEY
    fi
  fi
else
  log "  übersprungen"
fi

log "-- GitHub secret --"
if [[ "$APPLY_GITHUB" == "1" ]]; then
  github_set SUPABASE_STAGING_ANON_KEY "${STAGING_SUPABASE_ANON_KEY:-}"
else
  log "  übersprungen"
fi

log "-- ragapp eas.json (anon keys, git-tracked) --"
if [[ "$APPLY_EAS" == "1" ]]; then
  if [[ -z "$RAGAPP_ROOT" ]]; then
    log "  ragapp-Ordner nicht gefunden"
  elif filled "${STAGING_SUPABASE_ANON_KEY:-}" || filled "${PRODUCTION_SUPABASE_ANON_KEY:-}"; then
    plan "git:$RAGAPP_ROOT/eas.json" EXPO_PUBLIC_SUPABASE_ANON_KEY x
    if [[ "$DRY_RUN" -eq 0 ]]; then
      python3 - "$RAGAPP_ROOT/eas.json" <<'PY'
import json, os, sys
path = sys.argv[1]
with open(path, encoding="utf-8") as fh:
    data = json.load(fh)
changed = False
staging = os.environ.get("STAGING_SUPABASE_ANON_KEY", "")
prod = os.environ.get("PRODUCTION_SUPABASE_ANON_KEY", "")
builds = data.setdefault("build", {})
if staging and "staging" in builds:
    builds["staging"].setdefault("env", {})["EXPO_PUBLIC_SUPABASE_ANON_KEY"] = staging
    changed = True
if prod and "production" in builds:
    builds["production"].setdefault("env", {})["EXPO_PUBLIC_SUPABASE_ANON_KEY"] = prod
    changed = True
if changed:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")
    print(f"  wrote {path} (commit + EAS rebuild nötig)", file=sys.stderr)
PY
    fi
  fi
else
  log "  übersprungen (anon keys sind öffentlich; nach Änderung EAS-Build nötig)"
fi

log "-- Tuxai / Pulse --"
if [[ "$APPLY_TUXAI" == "1" ]]; then
  if filled "${QDRANT_API_KEY:-}"; then plan "tuxai:~/ragrun-server/.env" QDRANT_API_KEY x; fi
  if filled "${GRAFANA_ADMIN_PASSWORD:-}"; then plan "tuxai:~/ragrun-server/.env" GRAFANA_ADMIN_PASSWORD x; fi
  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "  Cutover: erst Railway --skip-deploys, dann Qdrant-Key auf dem Pulse, dann --redeploy"
  elif [[ -n "${TUXAI_SSH_HOST:-}" ]]; then
    log "  SSH $TUXAI_SSH_HOST — Key-Zeilen werden ohne Echo ersetzt"
    remote="${TUXAI_ENV_REMOTE:-~/ragrun-server/.env}"
    python3 - "$remote" "${TUXAI_SSH_HOST}" "${QDRANT_API_KEY:-}" "${GRAFANA_ADMIN_PASSWORD:-}" <<'PY'
import subprocess, sys
remote, host, qkey, gpass = sys.argv[1:5]
# Werte nicht in argv der Remote-Shell: Datei per stdin an python auf dem Host.
script = r'''
import os, re, sys
path = sys.argv[1]
raw = sys.stdin.read()
parts = raw.split("\n", 2)
qkey, gpass = parts[0], parts[1] if len(parts) > 1 else ""
text = open(os.path.expanduser(path), encoding="utf-8").read().splitlines()
def upsert(lines, key, val):
    if not val:
        return lines
    out, seen = [], False
    for line in lines:
        if line.startswith(key + "="):
            out.append(f"{key}={val}")
            seen = True
        else:
            out.append(line)
    if not seen:
        out.append(f"{key}={val}")
    return out
text = upsert(text, "QDRANT_API_KEY", qkey)
text = upsert(text, "GRAFANA_ADMIN_PASSWORD", gpass)
open(os.path.expanduser(path), "w", encoding="utf-8").write("\n".join(text) + "\n")
print("updated", path, file=sys.stderr)
'''
payload = qkey + "\n" + gpass + "\n"
proc = subprocess.run(
    ["ssh", host, "python3", "-c", script, remote],
    input=payload,
    text=True,
    check=True,
)
print(proc.stderr or "  tuxai env updated", file=sys.stderr)
PY
    log "  Qdrant neu starten (ein Key, harter Cutover):"
    log "    ssh $TUXAI_SSH_HOST 'cd ~/ragrun-server && docker compose up -d qdrant'"
    log "  Danach: deploy/tuxai/monitoring/make-qdrant-key.sh && up.sh  (auf dem Pulse)"
  else
    log "  TUXAI_SSH_HOST leer — manuell auf dem Pulse:"
    log "    QDRANT_API_KEY und ggf. GRAFANA_ADMIN_PASSWORD in ~/ragrun-server/.env"
    log "    docker compose up -d qdrant"
    log "    deploy/tuxai/monitoring/make-qdrant-key.sh && ./deploy/tuxai/monitoring/up.sh"
  fi
else
  log "  übersprungen"
fi

log
if [[ "$DRY_RUN" -eq 1 ]]; then
  log "Nichts geschrieben. Bei leerem Plan: Werte in rotate.env eintragen."
  log "Dann:  $0 --apply"
  log "Qdrant-Key: Railway zuerst mit --apply (ohne --redeploy), dann Pulse, dann $0 --apply --redeploy"
  log "Danach rotate.env löschen oder in den Passwortmanager schieben."
else
  log "Fertig. rotate.env löschen, sobald Railway und Pulse laufen:"
  log "  rm -P scripts/secrets/rotate.env"
fi
