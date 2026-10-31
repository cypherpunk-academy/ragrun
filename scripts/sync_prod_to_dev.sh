#!/usr/bin/env bash
# DEPRECATED: Production is no longer a sync source.
# Use ragprep instead:
#   rp deploy:analyse staging dev
#   rp deploy:sync staging dev --supabase [--qdrant] [--assistant <name>] --confirm
#
# Former behaviour: corpus tables from Production → Dev.

set -euo pipefail

echo "DEPRECATED: scripts/sync_prod_to_dev.sh" >&2
echo "  Production → Dev is removed. Refresh Dev from Staging:" >&2
echo "    rp deploy:sync staging dev --supabase --confirm" >&2
exit 1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# --- Load env files ---
load_env() {
  local file="$1"
  if [[ -f "$file" ]]; then
    while IFS= read -r line; do
      [[ "$line" =~ ^#.*$ || -z "$line" ]] && continue
      [[ "$line" =~ ^([^=]+)=(.*)$ ]] && export "${BASH_REMATCH[1]}"="${BASH_REMATCH[2]}"
    done < "$file"
  fi
}

load_env "$ROOT/.env"
load_env "$ROOT/.env.dev"

PROD_DSN="${RAGRUN_POSTGRES_DSN_PROD:-}"
DEV_DSN="${RAGRUN_POSTGRES_DSN:-}"

if [[ -z "$PROD_DSN" ]]; then
  echo "❌ RAGRUN_POSTGRES_DSN_PROD not set. Add it to .env:"
  echo "   RAGRUN_POSTGRES_DSN_PROD=postgresql://postgres.rmdqihhjjyizbuhxkxhn:<password>@aws-0-eu-west-1.pooler.supabase.com:5432/postgres"
  exit 1
fi
if [[ -z "$DEV_DSN" ]]; then
  echo "❌ RAGRUN_POSTGRES_DSN not set in .env.dev"
  exit 1
fi

# Tables to sync (corpus/config only — no user data)
TABLES=(
  rag_chunks
  rag_paragraphs
  app_paragraph_chunk
  rag_references
  rag_sources
  rag_talks
  app_starter_prompts
  llm_pricing
  alembic_version
)

echo "🔄 Syncing corpus tables: prod → reniets-dev"
echo "   Source: ragrun-production"
echo "   Target: reniets-dev"
echo ""

DUMP_FILE=$(mktemp /tmp/ragrun_prod_dump_XXXXXX.sql)
trap "rm -f $DUMP_FILE" EXIT

# Dump tables from production
echo "📦 Dumping from production..."
TABLES_ARGS=()
for t in "${TABLES[@]}"; do
  TABLES_ARGS+=(-t "$t")
done

pg_dump \
  --data-only \
  --no-owner \
  --no-privileges \
  --disable-triggers \
  "${TABLES_ARGS[@]}" \
  "$PROD_DSN" \
  > "$DUMP_FILE"

echo "   Dump size: $(du -sh "$DUMP_FILE" | cut -f1)"

# Truncate target tables in reverse order (FK safety)
echo "🗑️  Truncating target tables..."
TRUNCATE_SQL=""
for t in "${TABLES[@]}"; do
  TRUNCATE_SQL+="TRUNCATE TABLE IF EXISTS $t CASCADE; "
done
psql "$DEV_DSN" -c "$TRUNCATE_SQL"

# Restore into dev
echo "📥 Restoring into reniets-dev..."
psql "$DEV_DSN" \
  --single-transaction \
  -f "$DUMP_FILE"

echo ""
echo "✅ Sync complete. Verifying row counts..."
for t in "${TABLES[@]}"; do
  COUNT=$(psql "$DEV_DSN" -t -c "SELECT COUNT(*) FROM $t" 2>/dev/null | tr -d ' ')
  printf "   %-30s %s rows\n" "$t" "$COUNT"
done
