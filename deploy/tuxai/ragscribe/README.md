# ragscribe on Tuxai

Production: `https://scribe.ragxxx.com` → Docker on `127.0.0.1:8002` via tunnel route on **`ragrun-tuxai`**. Jobs and transcripts live in Qdrant collection **`ragscribe`** on the production Qdrant (`qdrant:6333`).

No staging hostname. Access is Cloudflare Access **Service Auth** (same service token as ragrun). Token values stay in 1Password / EAS secrets — never in git.

## Architecture

```
Android APK
  → https://scribe.ragxxx.com  (CF Access)
  → cloudflared (ragrun-tuxai)
  → http://127.0.0.1:8002  (container ragscribe)
       → Qdrant collection ragscribe (via Docker net ragrun-server_default)
```

Jobs API: `POST/GET/DELETE /api/v1/jobs` (poll; no long streams for the app).

## 1. Deploy the container

From the Mac (repo root), sync the server tree only:

```bash
rsync -avz --delete \
  --exclude mobile/ \
  --exclude '__pycache__/' \
  --exclude '.venv/' \
  --exclude '*.pyc' \
  ragscribe/ tuxai:~/ragscribe/
rsync -avz deploy/tuxai/ragscribe/ tuxai:~/ragrun-server/deploy/tuxai/ragscribe/
```

On Tuxai, export `QDRANT_API_KEY` (same as `~/ragrun-server/.env`), then:

```bash
cd ~/ragrun-server
# Prefer building from synced ~/ragscribe — override context if needed:
docker compose -f deploy/tuxai/ragscribe/compose.yaml up -d --build
```

Or from `~/ragscribe/docker-compose.yml` (must join `ragrun-server_default` and set `QDRANT_URL` / `QDRANT_API_KEY`).

```bash
curl -s http://127.0.0.1:8002/health
# expect ready: true
```

## 2. Cloudflare tunnel (reuse `ragrun-tuxai`)

Prefer **one extra published route** on the existing healthy tunnel **`ragrun-tuxai`**
(same connectors already on TUXAI). No second `cloudflared` process, no
`cloudflared service install`.

1. Zero Trust → **Networks → Tunnels** → **`ragrun-tuxai`** → **+ Add route**.
2. Type: **Published application**.
3. Hostname: `scribe` + domain `ragxxx.com` → public URL `https://scribe.ragxxx.com`.
4. Service: `http://127.0.0.1:8002` (HTTP).
5. Save. If DNS shows a manual **CNAME** instead of type **Tunnel**, delete it and recreate
   under **DNS → Records** as type **Tunnel** → content `ragrun-tuxai` (same as `ssh` /
   `qdrant-staging`). A bare CNAME can yield Cloudflare **1033** until that is fixed.

A separate tunnel + second systemd unit is only needed if you deliberately want
isolation; examples remain in
[`cloudflared.yml.example`](./cloudflared.yml.example) and
[`cloudflared-ragscribe.service.example`](./cloudflared-ragscribe.service.example).

## 3. Cloudflare Access

1. Zero Trust → **Access → Applications** → self-hosted app for `scribe.ragxxx.com`.
2. Policy: **Service Auth** only; Include → **Service Token** → same token as ragrun
   (`RAGRUN_CF_ACCESS_CLIENT_ID` / `RAGRUN_CF_ACCESS_CLIENT_SECRET` in Railway / `.env`).
   No separate `ragscribe` token required.
3. Put those same values into EAS production env as
   `EXPO_PUBLIC_CF_ACCESS_CLIENT_ID` / `EXPO_PUBLIC_CF_ACCESS_CLIENT_SECRET`
   (sensitive). Do not commit them.

## 4. Verify from the Mac

```bash
# Without token → 403
curl -sI https://scribe.ragxxx.com/health | head -1

# With token → 200, ready: true
curl -s https://scribe.ragxxx.com/health \
  -H "CF-Access-Client-Id: $RAGSCRIBE_CF_ACCESS_CLIENT_ID" \
  -H "CF-Access-Client-Secret: $RAGSCRIBE_CF_ACCESS_CLIENT_SECRET"
```

## Android APK

See [`ragscribe/mobile/README.md`](../../../ragscribe/mobile/README.md). Production EAS profile builds an internal APK with default URL `https://scribe.ragxxx.com` and Access headers baked in via EAS secrets.

## Files

| File | Role |
|------|------|
| [`compose.yaml`](./compose.yaml) | Docker service, localhost bind, Whisper volume |
| [`cloudflared.yml.example`](./cloudflared.yml.example) | Tunnel ingress for `scribe.ragxxx.com` |
| [`cloudflared-ragscribe.service.example`](./cloudflared-ragscribe.service.example) | Second systemd unit |
