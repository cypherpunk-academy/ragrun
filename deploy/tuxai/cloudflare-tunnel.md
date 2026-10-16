# Railway → Tuxai via Cloudflare Tunnel (Option B)

Betriebsplan inkl. Rotationsstrategie: [`plan-tuxai-cloudflare-railway.md`](./plan-tuxai-cloudflare-railway.md).

Replaces **Tailscale SOCKS on Railway** for Qdrant and the embedding service. Railway talks **HTTPS** to public hostnames; `cloudflared` on Tuxai forwards to `127.0.0.1:6333` and `:8001`. No `TS_AUTHKEY`, no `ALL_PROXY`.

Use this when Staging MCP searches time out or fail with `search failed` while OAuth works.

## Architecture

```
Claude → staging.ragxxx.com (Railway ragrun)
              → https://qdrant.ragxxx.com   ──cloudflared──► 127.0.0.1:6333
              → https://embed.ragxxx.com    ──cloudflared──► 127.0.0.1:8001
              → Supabase / Hugging Face       (direct, no tunnel)
```

Tuxai `compose.yaml` already binds Qdrant and embedder on `127.0.0.1` plus Tailscale IP — tunnel only needs localhost.

## 0. DNS

**Fix (Plan):** Zone **`ragxxx.com`** bei Cloudflare (Weg A) — Railway, **GitHub Pages**, MX und Tunnel-Hostnames in **einer** Zone. CNAME-only bei Chris auf `cfargotunnel.com` reicht für Railway **nicht** (fd10-AAAA). Details: [`plan-tuxai-cloudflare-railway.md`](./plan-tuxai-cloudflare-railway.md) § DNS.

## 1. Cloudflare Zero Trust (tunnel)

1. **Networking → Tunnels** → create **`ragrun-tuxai`** (cloudflared on Tuxai).
2. **Published application** routes (twice):

   | Public hostname | Service URL |
   |-----------------|-------------|
   | `qdrant.ragxxx.com` | `http://127.0.0.1:6333` |
   | `embed.ragxxx.com` | `http://127.0.0.1:8001` |

   In the form: Subdomain `qdrant` / `embed`, Domain `ragxxx.com`, or **custom hostname** if `ragxxx.com` is not in the dropdown.

   Example config: `deploy/tuxai/cloudflared/config.yml.example`

3. **Access (required):** Zero Trust → **Access** → **Applications** → add both hostnames. Policy: **Service Auth** only — **no** human login.

   Create **two** service tokens:

   | Token name | Railway service |
   |------------|-----------------|
   | `staging-tuxai` | `ragrun-staging` |
   | `prod-tuxai` | `ragrun` |

   Both tokens must be allowed on **both** applications. Store Client ID + Secret per token; never mix staging/prod on the wrong service.

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

**Remove** (or leave empty so entrypoint skips Tailscale):

- `TS_AUTHKEY`
- `TS_HOSTNAME`

**Set:**

| Variable | Example |
|----------|---------|
| `RAGRUN_QDRANT_URL` | `https://qdrant.ragxxx.com` |
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

From repo root (local, with staging Supabase vars if needed):

```bash
export $(grep -v '^#' .env.staging | grep -v '<' | xargs)
export RAGRUN_QDRANT_URL=https://qdrant.ragxxx.com
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
| Health OK, MCP still slow | Redeploy after **unset** `TS_AUTHKEY`; confirm logs have **no** `Tailscale up` / `socks5` |
| SSL errors | URLs must be `https://` on Railway |

## Relation to Tailscale

**Removed entirely** with the tunnel cutover (see plan § Tailscale abbauen). No Tailnet, no `TS_AUTHKEY` on Railway, no fallback path.
