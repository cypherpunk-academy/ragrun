#!/usr/bin/env bash
# Copy read-only RAG corpus tables from Supabase Staging → Production.
# Does NOT touch user data (app_notes, rag_talks turns, connector_grants, auth).
#
# Prerequisites:
#   1. Production schema ≥ Staging for these tables (run ragkeep/sql/019_lecture_catalog.sql
#      on Prod if rag_lecture_catalog is missing; apply ragapp migrations as on Staging).
#   2. .env.staging and .env.production (or .env.prod) with RAGRUN_POSTGRES_DSN (pooler URI).
#
# Usage:
#   ./scripts/sync_staging_to_prod.sh              # dry-run (dump only, no write)
#   CONFIRM=1 ./scripts/sync_staging_to_prod.sh   # truncate + restore on Production
#
# Requires: psql, pg_dump

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

load_env() {
  local file="$1"
  if [[ -f "$file" ]]; then
    while IFS= read -r line; do
      [[ "$line" =~ ^#.*$ || -z "$line" ]] && continue
      [[ "$line" =~ ^([^=]+)=(.*)$ ]] && export "${BASH_REMATCH[1]}"="${BASH_REMATCH[2]}"
    done <"$file"
  fi
}

load_env "$ROOT/.env.staging"
if [[ -f "$ROOT/.env.production" ]]; then
  load_env "$ROOT/.env.production"
elif [[ -f "$ROOT/.env.prod" ]]; then
  load_env "$ROOT/.env.prod"
fi

STAGING_DSN="${RAGRUN_POSTGRES_DSN_STAGING:-${STAGING_POSTGRES_DSN:-}}"
PROD_DSN="${RAGRUN_POSTGRES_DSN_PROD:-${PRODUCTION_POSTGRES_DSN:-${RAGRUN_POSTGRES_DSN:-}}}"

# If we loaded .env.staging last, RAGRUN_POSTGRES_DSN may be staging — prefer explicit prod file.
if [[ -f "$ROOT/.env.production" ]]; then
  PROD_DSN="$(grep -E '^RAGRUN_POSTGRES_DSN=' "$ROOT/.env.production" | head -1 | cut -d= -f2- | tr -d '"')"
elif [[ -f "$ROOT/.env.prod" ]]; then
  PROD_DSN="$(grep -E '^RAGRUN_POSTGRES_DSN=' "$ROOT/.env.prod" | head -1 | cut -d= -f2- | tr -d '"')"
fi
STAGING_DSN="$(grep -E '^RAGRUN_POSTGRES_DSN=' "$ROOT/.env.staging" | head -1 | cut -d= -f2- | tr -d '"')"

if [[ -z "$STAGING_DSN" ]]; then
  echo "❌ Staging DSN missing (.env.staging → RAGRUN_POSTGRES_DSN)" >&2
  exit 1
fi
if [[ -z "$PROD_DSN" ]]; then
  echo "❌ Production DSN missing (.env.production or .env.prod → RAGRUN_POSTGRES_DSN)" >&2
  exit 1
fi

TABLES=(
  rag_lecture_cycles
  rag_lecture_catalog
  rag_sources
  rag_paragraphs
  rag_chunks
  app_paragraph_chunk
  rag_references
  app_starter_prompts
  llm_pricing
)

echo "🔄 Corpus sync: Staging → Production"
echo "   Source: staging Supabase (from .env.staging)"
echo "   Target: production Supabase"
echo "   Tables: ${TABLES[*]}"
echo ""

DUMP_FILE=$(mktemp /tmp/ragrun_staging_dump_XXXXXX.sql)
trap 'rm -f "$DUMP_FILE"' EXIT

TABLES_ARGS=()
for t in "${TABLES[@]}"; do
  TABLES_ARGS+=(-t "$t")
done

echo "📦 Dumping from staging..."
pg_dump \
  --data-only \
  --no-owner \
  --no-privileges \
  "${TABLES_ARGS[@]}" \
  "$STAGING_DSN" >"$DUMP_FILE"
echo "   Dump size: $(du -sh "$DUMP_FILE" | cut -f1)"

if [[ "${CONFIRM:-}" != "1" ]]; then
  echo ""
  echo "Dry-run only. To write to Production:"
  echo "  CONFIRM=1 $0"
  exit 0
fi

echo "🗑️  Truncating production corpus tables (CASCADE)..."
TRUNCATE_LIST=$(IFS=, ; echo "${TABLES[*]}")
psql "$PROD_DSN" -v ON_ERROR_STOP=1 -c "TRUNCATE TABLE ${TRUNCATE_LIST} CASCADE;"

echo "📥 Restoring into production..."
# Supabase pooler user cannot ALTER TABLE … DISABLE TRIGGER; replica role skips FK/trigger checks.
psql "$PROD_DSN" --single-transaction -v ON_ERROR_STOP=1 <<EOF
SET session_replication_role = replica;
\\i $DUMP_FILE
SET session_replication_role = DEFAULT;
EOF

echo ""
echo "✅ Done. Row counts on production:"
for t in "${TABLES[@]}"; do
  COUNT=$(psql "$PROD_DSN" -t -A -c "SELECT COUNT(*) FROM $t" 2>/dev/null || echo "?")
  printf "   %-28s %s\n" "$t" "$COUNT"
done
