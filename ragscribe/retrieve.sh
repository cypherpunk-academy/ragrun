#!/usr/bin/env bash
# Neue fertige Transkripte aus der Qdrant-Collection ragscribe (Tuxai) nach inbox/ holen.
# DeepSeek bestimmt nur den Dateinamen (Autor__Thema.md). Der Text bleibt unverändert.
# Jobs, die in inbox/ schon als "Job: <uuid>" stehen, werden übersprungen.
#
#   ./retrieve.sh
#   ./retrieve.sh --dry-run
#
# Voraussetzung: SSH-Host tuxai (oder TUXAI_SSH_HOST), DEEPSEEK_REST_API_KEY.
set -euo pipefail
cd "$(dirname "$0")"

if [[ -z "${DEEPSEEK_REST_API_KEY:-}${RAGRUN_DEEPSEEK_API_KEY:-}" ]]; then
  for env_file in "../.env" "../../ragprep/.env"; do
    [[ -f "$env_file" ]] || continue
    line="$(grep -E '^(DEEPSEEK_REST_API_KEY|RAGRUN_DEEPSEEK_API_KEY)=' "$env_file" | tail -1 || true)"
    [[ -n "$line" ]] || continue
    key="${line%%=*}"
    val="${line#*=}"
    val="${val%\"}"; val="${val#\"}"
    val="${val%\'}"; val="${val#\'}"
    if [[ "$val" == \$\{*\} ]]; then
      ref="${val#\$\{}"
      ref="${ref%\}}"
      val="${!ref:-}"
    fi
    if [[ -n "$val" ]]; then
      export "$key=$val"
    fi
  done
fi
if [[ -z "${DEEPSEEK_REST_API_KEY:-}" && -n "${RAGRUN_DEEPSEEK_API_KEY:-}" ]]; then
  export DEEPSEEK_REST_API_KEY="$RAGRUN_DEEPSEEK_API_KEY"
fi

export PYENV_VERSION="${RAGSCRIBE_PYENV:-reniets}"
exec python -m app.retrieve "$@"
