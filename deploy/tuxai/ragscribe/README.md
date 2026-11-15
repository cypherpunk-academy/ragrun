# ragscribe on Tuxai

Production: `https://scribe.ragxxx.com` → Docker on `127.0.0.1:8002` via a **separate** Cloudflare tunnel (`ragscribe`). Does not touch `ragrun-tuxai` / `cloudflared-ragrun`.

No staging hostname. Access is Cloudflare Access **Service Auth** only (token name `ragscribe`). Token values stay in 1Password / EAS secrets — never in git.

## Architecture

```
Android APK
  → https://scribe.ragxxx.com  (CF Access: CF-Access-Client-Id / Secret)
  → cloudflared-ragscribe
  → http://127.0.0.1:8002  (container ragscribe)
```

## 1. Deploy the container

From the Mac (repo root), sync the server tree only:

```bash
rsync -avz --delete \
  --exclude mobile/ \
  --exclude '__pycache__/' \
  --exclude '.venv/' \
  --exclude '*.pyc' \
  ragscribe/ tuxai:~/ragscribe/
```

On Tuxai, check free RAM (Whisper `medium` plus the existing e5 embedder), then build and start. Prefer the repo compose if the full `ragrun` tree is present:

```bash
# If deploy files live under ~/ragrun-server (or similar):
cd ~/ragrun-server   # adjust to your clone path
docker compose -f deploy/tuxai/ragscribe/compose.yaml up -d --build
```

Or a minimal compose next to the synced tree (`~/ragscribe/docker-compose.yml`):

```yaml
services:
  ragscribe:
    build: .
    container_name: ragscribe
    restart: unless-stopped
    ports:
      - "127.0.0.1:8002:8002"
    environment:
      WHISPER_MODEL: medium
      WHISPER_DEVICE: cpu
      WHISPER_DOWNLOAD_ROOT: /root/.cache/whisper
    volumes:
      - ragscribe-whisper-cache:/root/.cache/whisper
volumes:
  ragscribe-whisper-cache:
```

```bash
cd ~/ragscribe && docker compose up -d --build
curl -s http://127.0.0.1:8002/health
# expect ready: true (first start downloads the Whisper model)
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
