#!/usr/bin/env node
/**
 * DEPRECATED: Production is no longer a sync source.
 * Use: rp deploy:sync staging dev --supabase --confirm
 */

console.error('DEPRECATED: scripts/sync_prod_to_dev.mjs');
console.error('  Production → Dev is removed. Refresh Dev from Staging:');
console.error('    rp deploy:sync staging dev --supabase --confirm');
process.exit(1);
