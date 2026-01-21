#!/usr/bin/env sh
set -eu

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

exec uvicorn app.main:app --host 0.0.0.0 --port 8000

