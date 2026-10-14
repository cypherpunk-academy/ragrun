# Railway → Tuxai over Tailscale

Staging (`ragrun-staging`) and Production (`ragrun`) reach Qdrant and the embedder on the Pulse only via the tailnet (`100.93.243.73:6333`, `:8001`). The container joins the tailnet at boot when `TS_AUTHKEY` is set.

## 1. Tailscale admin (once)

1. [Admin console](https://login.tailscale.com/admin/settings/keys) → **Generate auth key**
   - Reusable: **yes** (one key per environment is fine; rotate separately)
   - Ephemeral: **no** (stable hostname in ACL logs)
   - Tags (recommended): e.g. `tag:ragrun` — create the tag in ACL first if you use tags
2. **ACL**: allow `tag:ragrun` (or the Railway hostnames) to reach the Pulse on TCP `6333` and `8001`. Example fragment:

```json
"acls": [
  {"action": "accept", "src": ["tag:ragrun"], "dst": ["100.93.243.73:6333", "100.93.243.73:8001"]}
]
```

Adjust if the Pulse uses a machine name instead of the IP in your tailnet.

3. Confirm the Pulse (`tuxai`) is on the same tailnet and still binds Qdrant/embeddings to the Tailscale IP (see `deploy/tuxai/compose.yaml`).

## 2. Railway variables

Set on **each** service (values never in git):

| Variable | Staging (`ragrun-staging`) | Production (`ragrun`) |
|----------|----------------------------|------------------------|
| `TS_AUTHKEY` | auth key (stdin) | auth key (stdin) |
| `TS_HOSTNAME` | `ragrun-staging` | `ragrun-production` |
| `RAGRUN_QDRANT_URL` | `http://100.93.243.73:6333` | same |
| `RAGRUN_QDRANT_API_KEY` | Pulse key | same |
| `RAGRUN_EMBEDDINGS_PROVIDER` | `http` | same |
| `RAGRUN_EMBEDDINGS_BASE_URL` | `http://100.93.243.73:8001` | same |

```bash
cd ragrun
echo "$TS_AUTHKEY" | railway variable set TS_AUTHKEY --stdin -e staging -s ragrun-staging
railway variable set TS_HOSTNAME=ragrun-staging -e staging -s ragrun-staging
# repeat for production / ragrun
railway redeploy -e staging -s ragrun-staging -y
railway redeploy -e production -s ragrun -y
```

Or use `scripts/secrets/rotate.env` + `apply.sh` (see `scripts/secrets/README.md`).

**Do not** set `TS_AUTHKEY` on local Dev/ngrok — only Railway.

## 3. Verify

```bash
curl -s https://staging.ragxxx.com/healthz | jq '.dependencies.qdrant, .dependencies.embedding_service'
curl -s https://api.ragxxx.com/healthz | jq '.dependencies.qdrant, .dependencies.embedding_service'
```

Both should be `healthy` with endpoints under `100.93.243.73`.

Deploy logs should contain `[ragrun] Tailscale up (100.93.243.73)` (or similar).

## 4. Troubleshooting

| Symptom | Check |
|---------|--------|
| `tailscale up` fails in logs | Key expired/revoked; tag not allowed on key |
| Qdrant unhealthy, Tailscale OK | Wrong `RAGRUN_QDRANT_API_KEY` vs Pulse `QDRANT_API_KEY` |
| Timeout on health | ACL blocks Pulse ports; Pulse offline |
| Works on Mac, not Railway | `TS_AUTHKEY` missing on that service |

Local Docker and Dev MCP stay on `localhost:6333` / `localhost:8001` — no Tailscale in those images unless you export `TS_AUTHKEY` yourself.

**Userspace on Railway:** `tailscaled` runs with `--socks5-server=127.0.0.1:1055`; the entrypoint sets `ALL_PROXY` / `HTTP_PROXY` so httpx (health checks, Qdrant, embedder) uses the tailnet.
