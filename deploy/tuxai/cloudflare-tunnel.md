# Railway → Tuxai via Cloudflare Tunnel (Option B)

Replaces **Tailscale SOCKS on Railway** for Qdrant and the embedding service. Railway talks **HTTPS** to public hostnames; `cloudflared` on Tuxai forwards to `127.0.0.1:6333` and `:8001`. No `TS_AUTHKEY`, no `ALL_PROXY`.

Use this when Staging MCP searches time out or fail with `search failed` while OAuth works.

## Architecture

```
Claude → staging.ragxxx.com (Railway ragrun)
              → https://tuxai-qdrant.<zone>   ──cloudflared──► 127.0.0.1:6333
              → https://tuxai-embed.<zone>    ──cloudflared──► 127.0.0.1:8001
              → Supabase / Hugging Face       (direct, no tunnel)
```

Tuxai `compose.yaml` already binds Qdrant and embedder on `127.0.0.1` plus Tailscale IP — tunnel only needs localhost.

## 1. Cloudflare (once)

1. Add your zone (e.g. `ragxxx.com`) to Cloudflare if not already.
2. **Zero Trust** → **Networks** → **Tunnels** → **Create tunnel** → **Cloudflared**.
3. Install `cloudflared` on Tuxai (Pulse); authenticate with the token Cloudflare shows.
4. **Public hostnames** (two routes):

   | Public hostname | Service | Origin |
   |-----------------|---------|--------|
   | `tuxai-qdrant.ragxxx.com` | HTTP | `http://127.0.0.1:6333` |
   | `tuxai-embed.ragxxx.com` | HTTP | `http://127.0.0.1:8001` |

   Example config: `deploy/tuxai/cloudflared/config.yml.example`

5. **Access (recommended):** Zero Trust → **Access** → **Applications** → add both hostnames. Policy: **Service Auth** → create a **Service Token**. Note **Client ID** and **Client Secret**.

   Without Access, anyone who discovers the URL could hit Qdrant if they guess the API key — use Access + existing `QDRANT_API_KEY`.

## 2. Tuxai: run cloudflared

```bash
# systemd or compose sidecar; minimal manual test:
cloudflared tunnel --config /path/to/config.yml run
```

Ensure `curl -s https://tuxai-qdrant.ragxxx.com/healthz` works from your Mac (with Access headers if enabled).

Qdrant expects header `api-key: <QDRANT_API_KEY>`.

Embed: `POST https://tuxai-embed.ragxxx.com/api/v1/health/simple`

## 3. Railway `ragrun-staging`

**Remove** (or leave empty so entrypoint skips Tailscale):

- `TS_AUTHKEY`
- `TS_HOSTNAME`

**Set:**

| Variable | Example |
|----------|---------|
| `RAGRUN_QDRANT_URL` | `https://tuxai-qdrant.ragxxx.com` |
| `RAGRUN_QDRANT_API_KEY` | same as Tuxai `QDRANT_API_KEY` |
| `RAGRUN_EMBEDDINGS_BASE_URL` | `https://tuxai-embed.ragxxx.com` |
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
export RAGRUN_QDRANT_URL=https://tuxai-qdrant.ragxxx.com
export RAGRUN_EMBEDDINGS_BASE_URL=https://tuxai-embed.ragxxx.com
# plus RAGRUN_CF_ACCESS_* if used
.venv/bin/python scripts/benchmark_retrieval_latency.py --label staging-via-cf
```

Expect hybrid/search timings similar to **tuxai-from-mac** (~0.2–0.8 s), not minutes.

## 5. Production

Same pattern on service `ragrun` when Staging is stable: tunnel hostnames can be shared (same Tuxai corpus) or separate subdomains per env.

## Troubleshooting

| Symptom | Check |
|---------|--------|
| `403` from tunnel URL | Cloudflare Access: set `RAGRUN_CF_ACCESS_*` on Railway |
| `401` Qdrant | `RAGRUN_QDRANT_API_KEY` mismatch |
| Health OK, MCP still slow | Redeploy after **unset** `TS_AUTHKEY`; confirm logs have **no** `Tailscale up` / `socks5` |
| SSL errors | URLs must be `https://` on Railway |

## Relation to Tailscale

- **Tuxai:** Tailscale on `100.93.243.73` can stay for **your Mac**, Grafana, manual ops.
- **Railway:** should **not** use Tailscale SOCKS for Qdrant/embed (see `railway-tailscale.md` legacy note).
