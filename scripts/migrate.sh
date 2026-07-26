#!/usr/bin/env bash
# Apply ragapp Supabase migrations (001–014) to a target project.
#
# Usage:
#   ./scripts/migrate.sh "postgresql://postgres.<ref>:<password>@aws-0-eu-central-1.pooler.supabase.com:5432/postgres"
#
# The DATABASE_URL can also be set as environment variable:
#   DATABASE_URL="postgresql://..." ./scripts/migrate.sh
#
# Migration 000_reset.sql is intentionally skipped (destructive — drops all tables).
# All other migrations are idempotent (IF NOT EXISTS / CREATE OR REPLACE).

set -euo pipefail

DATABASE_URL="${1:-${DATABASE_URL:-}}"

if [[ -z "$DATABASE_URL" ]]; then
  echo "ERROR: No DATABASE_URL provided." >&2
  echo "Usage: $0 \"postgresql://postgres.<ref>:<password>@host:5432/postgres\"" >&2
  exit 1
fi

MIGRATIONS_DIR="$(cd "$(dirname "$0")/../" && pwd)/../ragapp/supabase/migrations"

if [[ ! -d "$MIGRATIONS_DIR" ]]; then
  echo "ERROR: Migrations directory not found: $MIGRATIONS_DIR" >&2
  echo "Run this script from the ragrun repo root." >&2
  exit 1
fi

echo "Applying migrations from: $MIGRATIONS_DIR"
echo "Target: ${DATABASE_URL%%@*}@..."
echo ""

for file in $(ls "$MIGRATIONS_DIR"/*.sql | sort); do
  filename=$(basename "$file")
  if [[ "$filename" == "000_reset.sql" ]]; then
    echo "  SKIP  $filename  (destructive reset — skipped)"
    continue
  fi
  echo -n "  APPLY $filename ... "
  psql "$DATABASE_URL" -f "$file" -v ON_ERROR_STOP=1 --quiet
  echo "done"
done

echo ""
echo "All migrations applied successfully."
