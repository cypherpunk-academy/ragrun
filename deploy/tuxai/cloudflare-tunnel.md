# Railway → Tuxai via Cloudflare Tunnel (Option B)

Betriebsplan inkl. Rotationsstrategie: [`plan-tuxai-cloudflare-railway.md`](./plan-tuxai-cloudflare-railway.md).

Railway talks **HTTPS** to public hostnames; `cloudflared` on Tuxai forwards to `127.0.0.1:6333` and `:8001`.

Use this when Staging MCP searches time out or fail with `search failed` while OAuth works.

## Architecture

```
Claude → staging.ragxxx.com (Railway ragrun-staging)
              → https://qdrant-staging.ragxxx.com ──cloudflared──► 127.0.0.1:6334  (Staging Qdrant)
              → https://embed.ragxxx.com          ──cloudflared──► 127.0.0.1:8001  (shared)
              → Supabase                          (direct)

Production ragrun → qdrant.ragxxx.com → 127.0.0.1:6333, same embed.ragxxx.com → :8001
```

Tuxai `compose.yaml` binds Qdrant and the embedder on `127.0.0.1`. The tunnel uses that localhost origin.

## 0. DNS

**Fix (Plan):** Zone **`ragxxx.com`** bei Cloudflare (Weg A) — Railway, **GitHub Pages**, MX und Tunnel-Hostnames in **einer** Zone. CNAME-only bei Chris auf `cfargotunnel.com` reicht für Railway **nicht** (fd10-AAAA). Details: [`plan-tuxai-cloudflare-railway.md`](./plan-tuxai-cloudflare-railway.md) § DNS.

## 1. Cloudflare Zero Trust (tunnel)

1. **Networking → Tunnels** → create **`ragrun-tuxai`** (cloudflared on Tuxai).
2. **Published application** routes:

   | Public hostname | Service URL | Nutzer |
   |-----------------|-------------|--------|
   | `qdrant.ragxxx.com` | `http://127.0.0.1:6333` | Production |
   | `qdrant-staging.ragxxx.com` | `http://127.0.0.1:6334` | Staging |
   | `embed.ragxxx.com` | `http://127.0.0.1:8001` | **Staging + Production** (ein Embedder) |

   Kein `embed-staging` — zweites e5-Modell im RAM entfällt.

   In the form: Subdomain + Domain `ragxxx.com`, or **custom hostname** if `ragxxx.com` is not in the dropdown.

   Example config: `deploy/tuxai/cloudflared/config.yml.example`

3. **Access (required):** Zero Trust → **Access** → **Applications** → **Service Auth** only.

   **Ein Service Token** für alle Tunnel-Apps (`qdrant`, `qdrant-staging`, `embed`, ggf. `ssh`). Dieselben `RAGRUN_CF_ACCESS_CLIENT_ID` / `RAGRUN_CF_ACCESS_CLIENT_SECRET` auf Railway **`ragrun-staging`** und **`ragrun`** (Production).

   Pro App: Policy **Include → Service Token** → dieses eine Token auswählen. Keine getrennten Staging-/Prod-Tokens nötig.

## 2. Tuxai: run cloudflared

```bash
# systemd: deploy/tuxai/cloudflared/cloudflared.service.example
# manual test:
cloudflared tunnel --config /etc/cloudflared/config.yml run
```

From your Mac: `./deploy/tuxai/verify-tunnel.sh` (set `RAGRUN_*` + CF Access headers env vars).

Qdrant expects header `api-key: <QDRANT_API_KEY>`.

Embed: `GET https://embed.ragxxx.com/api/v1/health/simple`

## 3. Railway `ragrun-staging`

**Set:**

| Variable | Example |
|----------|---------|
| `RAGRUN_QDRANT_URL` | `https://qdrant-staging.ragxxx.com` |
| `RAGRUN_QDRANT_API_KEY` | same as Tuxai `QDRANT_API_KEY` |
| `RAGRUN_EMBEDDINGS_BASE_URL` | `https://embed.ragxxx.com` |
| `RAGRUN_EMBEDDINGS_PROVIDER` | `http` |
| `RAGRUN_CF_ACCESS_CLIENT_ID` | Service token Client ID (if Access enabled) |
| `RAGRUN_CF_ACCESS_CLIENT_SECRET` | Service token secret |

Redeploy `ragrun-staging`.

## 4. Verify

```bash
curl -s https://staging.ragxxx.com/healthz | jq '.dependencies.qdrant, .dependencies.embedding_service'
```

Both should be `healthy` with **https** endpoints on your tunnel hostnames.

The same path, including Supabase and a vector search, is `pytest tests/staging` (reads `.env.staging`, stays out of a plain `pytest`).

From repo root (local, with staging Supabase vars if needed):

```bash
export $(grep -v '^#' .env.staging | grep -v '<' | xargs)
export RAGRUN_QDRANT_URL=https://qdrant-staging.ragxxx.com
export RAGRUN_EMBEDDINGS_BASE_URL=https://embed.ragxxx.com
# plus RAGRUN_CF_ACCESS_* if used
.venv/bin/python scripts/benchmark_retrieval_latency.py --label staging-via-cf
```

Expect hybrid/search timings similar to **tuxai-from-mac** (~0.2–0.8 s), not minutes.

## 5. Production

Same pattern on service `ragrun` when Staging is stable (shared hostnames, separate Access tokens).

## Troubleshooting

| Symptom | Check |
|---------|--------|
| `403` from tunnel URL | Cloudflare Access: set `RAGRUN_CF_ACCESS_*` on Railway |
| `401` Qdrant | `RAGRUN_QDRANT_API_KEY` mismatch |
| NXDOMAIN / SSL | Chris CNAME not set or not propagated |
| SSL errors | URLs must be `https://` on Railway |
