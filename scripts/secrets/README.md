# Secrets: Staging und Production

Wenn du kritische Tokens neu erzeugst, stehen sie an mehreren Stellen. Dieses Verzeichnis ist die Liste und das Eintragsskript.

**Werte nie in Git, nie in den Chat, nie ins Skript selbst.** Nur in `rotate.env` (gitignored, `chmod 600`).

## Ist das sicher?

Ein Eintragsskript ist **akzeptabel als Cutover-Hilfe**, nicht als Tresor.

| Ja | Nein |
|----|------|
| Werte-Datei ist gitignored und `600` | Secrets ins `.sh` oder ins Example schreiben |
| Standard ist Dry-Run, Ausgabe nur Schlüsselnamen | `set -x` / Werte loggen |
| Railway und GitHub bekommen den Wert über stdin, nicht als CLI-Argument | `railway variable set KEY=geheim` (landet in `ps`) |
| Nach dem Lauf `rotate.env` löschen | Datei in iCloud/Time Machine/Chat liegen lassen |
| Railway-Login lokal, kein Token in der Datei | Unattended auf einem geteilten Rechner |

Besser als Secrets ins Repo oder in diesen Chat zu legen. Schwächer als 1Password/Bitwarden (`op inject`). Die bestehenden `.env.staging` / `.env.production` auf dem Mac sind schon lokale Kopien — `rotate.env` ist nur die zweite, temporäre.

Reihenfolge bei einem **Qdrant-Key** (ein Key auf Tuxai, Staging und Production teilen ihn):

1. `apply.sh --apply` ohne `--redeploy` (Railway-Variablen setzen, alter Prozess läuft weiter)
2. Key auf dem Pulse in `~/ragrun-server/.env`, Qdrant neu starten
3. `apply.sh --apply --redeploy`

Supabase-Keys zuerst im Dashboard erzeugen, dann hier eintragen, dann alte Keys dort ungültig machen (sobald Railway den neuen Wert hat).

**Cloudflare Tunnel + Access:** Zone `ragxxx.com`; **ein** Service-Auth-Token für alle Tunnel-Apps. Railway Staging und Prod bekommen **dieselben** `RAGRUN_CF_ACCESS_CLIENT_ID` / `SECRET` (in `rotate.env` kann `STAGING_CF_ACCESS_*` und `PRODUCTION_CF_ACCESS_*` denselben Wert tragen). Rotation manuell — `deploy/tuxai/plan-tuxai-cloudflare-railway.md`.

## Woher kommen neue Werte

| Secret | Neu erzeugen | Sensitivität |
|--------|----------------|--------------|
| Staging/Prod Anon Key | Supabase → Project Settings → API Keys (publishable) | niedrig (öffentlicher Client-Key) |
| Staging/Prod Service Role | dasselbe, secret `service_role` | **hoch** — volle DB |
| Staging/Prod JWT Secret | Settings → JWT Keys (Legacy). Staging nutzt oft nur JWKS | **hoch** — invalidiert Sessions |
| `RAGRUN_POSTGRES_DSN` | Settings → Database → Passwort rotieren, Pooler-URI neu bauen | **hoch** |
| `QDRANT_API_KEY` | `openssl rand -hex 32`, dann Pulse | **hoch** — gemeinsamer Korpus |
| `DEEPSEEK_API_KEY` | DeepSeek Console → API keys | **hoch** — Kosten |
| `TUNNEL_QDRANT_URL` / `TUNNEL_EMBED_URL` | feste Tunnel-URLs (apply → beide Railway-Services) | niedrig (öffentlich) |
| `STAGING_CORPUS_PUBLIC_BASE_URL` / `PRODUCTION_CORPUS_PUBLIC_BASE_URL` | R2 Custom Domains `corpus-staging` / `corpus` | niedrig (öffentlich) |
| `R2_ACCOUNT_ID` / `R2_ACCESS_KEY_ID` / `R2_SECRET_ACCESS_KEY` | Cloudflare → R2 → Manage API tokens | **hoch** — Objekt-Schreiben |
| `STAGING_R2_BUCKET` / `PRODUCTION_R2_BUCKET` | `philo-corpus-staging` / `philo-corpus` | niedrig |
| `STAGING_CF_ACCESS_*` / `PRODUCTION_CF_ACCESS_*` | Zero Trust → **ein** Service Token (oft identischer Wert) | **hoch** |
| `HF_TOKEN` | Hugging Face (optional; bei `EMBEDDINGS_PROVIDER=http` nicht nötig auf Railway) | mittel |
| `RAGRUN_INTERNAL_API_KEY` | `openssl rand -hex 32` (heute leer; sperrt `/api/v1/rag` und `/admin`) | **hoch** |
| `GRAFANA_ADMIN_PASSWORD` | selbst wählen, Pulse `~/ragrun-server/.env` | hoch |
| GitHub `SUPABASE_STAGING_ANON_KEY` | derselbe Staging-Anon-Key | niedrig |

Postgres-Passwort in der URI URL-encoden (`*` → `%2A`, `!` → `%21`). Railway darf `postgresql+psycopg://…` mit unencoded Passwort nutzen; die lokalen Dateien nutzen `postgresql://…`.

## Wohin schreiben

**Tuxai:** Staging und Production teilen **einen** Embedder (`embed.ragxxx.com` → `:8001`). **Qdrant getrennt:** Prod `qdrant.ragxxx.com` → `:6333`, Staging `qdrant-staging.ragxxx.com` → `:6334` (eigener Container/Storage). Tunnel: `deploy/tuxai/cloudflare-tunnel.md`. Nutzerdaten: getrennte Supabase-Projekte.

### Staging — Supabase `duhtxxbynkilpxdpcpsk`

| Variable | Ziele |
|----------|--------|
| `RAGRUN_SUPABASE_ANON_KEY` | Railway Service `ragrun-staging`; `.env.staging`; GitHub Secret `SUPABASE_STAGING_ANON_KEY`; optional `ragapp/eas.json` Build `staging` |
| `RAGRUN_SUPABASE_SERVICE_ROLE_KEY` + `SUPABASE_SERVICE_ROLE_KEY` | Railway `ragrun-staging`; `.env.staging` (ragkeep-Hooks) |
| `RAGRUN_SUPABASE_JWT_SECRET` | optional Railway + `.env.staging` (heute oft leer, JWKS über die URL) |
| `RAGRUN_POSTGRES_DSN` | Railway `ragrun-staging`; `.env.staging`; `ragprep/.env.staging` |
| `RAGRUN_QDRANT_URL` / `RAGRUN_EMBEDDINGS_BASE_URL` | Tunnel-HTTPS-URLs (siehe `cloudflare-tunnel.md`) |
| `RAGRUN_QDRANT_API_KEY` | Railway `ragrun-staging`; `.env.staging`; Pulse `QDRANT_API_KEY` |
| `RAGRUN_CF_ACCESS_CLIENT_ID` / `RAGRUN_CF_ACCESS_CLIENT_SECRET` | **Gleiches** Service-Token wie Prod (alle Tunnel-Apps) |
| `RAGRUN_DEEPSEEK_API_KEY` | Railway `ragrun-staging`; `.env.staging` |
| `RAGRUN_HF_TOKEN` | Railway `ragrun-staging` (Legacy) |
| `RAGRUN_INTERNAL_API_KEY` | Railway `ragrun-staging`, wenn du die internen Routen schließen willst |
| `RAGRUN_CORPUS_PUBLIC_BASE_URL` | Railway `ragrun-staging`; `.env.staging` → `https://corpus-staging.ragxxx.com` |
| `RAGRUN_R2_*` | Railway `ragrun-staging`; `.env.staging` (Bucket `philo-corpus-staging`) |

API-URL (kein Secret): `https://staging.ragxxx.com`. Keepalive: `.github/workflows/supabase-staging-keepalive.yml`.

Bekannte Falle: `.env.staging` hatte zeitweise den **Dev**-Service-Role zum Staging-URL. Nach einer Rotation müssen URL, Anon, Service-Role und DSN **dasselbe** Projekt meinen.

### Production — Supabase `rmdqihhjjyizbuhxkxhn`

| Variable | Ziele |
|----------|--------|
| `RAGRUN_SUPABASE_ANON_KEY` | Railway Service `ragrun`; `.env.production`; optional `EXPO_PUBLIC_*` dort und `ragapp/eas.json` Build `production` |
| `RAGRUN_SUPABASE_SERVICE_ROLE_KEY` + `SUPABASE_SERVICE_ROLE_KEY` | Railway `ragrun`; `.env.production` |
| `RAGRUN_SUPABASE_JWT_SECRET` | Railway `ragrun`; `.env.production` |
| `RAGRUN_POSTGRES_DSN` | Railway `ragrun`; `.env.production`; `ragprep/.env.production`; `.env.dev` Feld `RAGRUN_POSTGRES_DSN_PROD` (Prod→Dev-Sync) |
| `RAGRUN_QDRANT_API_KEY` | Railway `ragrun`; `.env.production`; Pulse (gleicher Key wie Staging) |
| `RAGRUN_QDRANT_URL` / `RAGRUN_EMBEDDINGS_BASE_URL` | Tunnel-HTTPS-URLs (siehe `cloudflare-tunnel.md`) |
| `RAGRUN_CF_ACCESS_CLIENT_ID` / `RAGRUN_CF_ACCESS_CLIENT_SECRET` | Cloudflare Zero Trust → Service Auth (shared token) |
| `RAGRUN_DEEPSEEK_API_KEY` | Railway `ragrun`; `.env.production` |
| `RAGRUN_HF_TOKEN` | Railway `ragrun` |
| `RAGRUN_INTERNAL_API_KEY` | Railway `ragrun` |
| `RAGRUN_CORPUS_PUBLIC_BASE_URL` | Railway `ragrun`; `.env.production` → `https://corpus.ragxxx.com` |
| `RAGRUN_R2_*` | Railway `ragrun`; `.env.production` (Bucket `philo-corpus`) |

API-URL (kein Secret): `https://api.ragxxx.com`.

`ragprep/.env.production` kann noch eine alte Qdrant-Cloud-URI haben. Nach Tuxai-Umstellung dort URL+Key auf den Pulse setzen, nicht den Cloud-Key neu ausrollen.

### Pulse (Tuxai)

| Variable | Datei | Danach |
|----------|--------|--------|
| `QDRANT_API_KEY` | `~/ragrun-server/.env` | `docker compose up -d qdrant` |
| `GRAFANA_ADMIN_PASSWORD` | dieselbe Datei | Monitoring-`up.sh` |
| Prometheus-Keydatei | `deploy/tuxai/monitoring/qdrant_key` | `make-qdrant-key.sh` (gitignored) |

Railway-Service `qdrant-staging` ist tot (G2b). Nicht mehr befüllen.

### Was das Skript nicht anfasst

- `.env` / `.env.dev` / `ragapp/.env.local` — Development
- Expo-Store-Builds, bis du `APPLY_EAS=1` setzt und neu baust
- ragrun-UI (`NEXTAUTH_SECRET`, `GITHUB_CLIENT_*`), falls die UI noch als eigener Railway-Service existiert — im aktuellen Projekt `ragrun` gibt es nur `ragrun` und `ragrun-staging`
- DeepSeek/Supabase/HF **am Anbieter** — dort manuell erzeugen bzw. alte Keys widerrufen

## Skript

```bash
cd /Users/michael/Reniets/Ai/ragrun
cp scripts/secrets/rotate.env.example scripts/secrets/rotate.env
chmod 600 scripts/secrets/rotate.env
# Neue Werte eintragen, Unverändertes leer lassen

./scripts/secrets/apply.sh                 # Plan: nur Schlüssel + Ziele
./scripts/secrets/apply.sh --apply         # schreiben, Railway ohne Restart
./scripts/secrets/apply.sh --apply --redeploy
rm -P scripts/secrets/rotate.env
```

Voraussetzungen: `railway` eingeloggt (`railway whoami`), für GitHub `gh auth status`. Projekt-ID und Service-Namen stehen schon im Example.

`ragkeep` pre-push (`sync_sources`, `upload_covers`) liest `RAGRUN_POSTGRES_DSN` und `SUPABASE_SERVICE_ROLE_KEY` aus der jeweils geladenen ragrun-Env — sobald die lokalen Dateien stimmen, sind die Hooks mitgedreht.
