# ragscribe

Turn spoken videos (YouTube, TikTok, …) into text. Standalone: no ragrun API, Qdrant, or Postgres required.

## Requirements

- Python 3.11+ with [openai-whisper](https://github.com/openai/whisper) (default: pyenv env `reniets`)
- `ffmpeg` on `PATH`
- `yt-dlp` (installed via `requirements.txt`)

## Start the server

```bash
cd ragscribe
./start.sh
```

Listens on `0.0.0.0:8002` so a phone on the same LAN can reach it.

- Health: `GET http://127.0.0.1:8002/health`
- Inputs: `GET http://127.0.0.1:8002/api/v1/inputs`
- Scribe: `POST http://127.0.0.1:8002/api/v1/scribe`

### Example

```bash
curl -N -X POST http://127.0.0.1:8002/api/v1/scribe \
  -H 'Content-Type: application/json' \
  -d '{"input":"video_url","url":"https://www.youtube.com/watch?v=…","language":"de"}'
```

NDJSON events: `fetch` (download), `start` / `progress` / `done` (transcription), `error`, `heartbeat`.

### Phone

Production default in the app is `https://scribe.ragxxx.com` (Docker on Tuxai + Cloudflare tunnel; see [`deploy/tuxai/ragscribe/README.md`](../deploy/tuxai/ragscribe/README.md)). The release APK sends Cloudflare Access headers for that host.

For local phone testing: start the server with `./start.sh`, then either use the LAN URL `http://<your-LAN-IP>:8002`, or temporarily point ngrok at port 8002:

```bash
ngrok http 8002 --domain=shana-peatier-not.ngrok-free.dev
```

(`ngr` always tunnels port 8000 for Philo Dev. Same reserved domain — only one of Philo Dev / ragscribe ngrok at a time.)

## Mobile app

See [`mobile/`](mobile/) — Expo app for Android and iOS that accepts shared links, shows progress, and shares the transcript.

## Env

| Variable | Default | Meaning |
|----------|---------|---------|
| `WHISPER_MODEL` | `medium` | Whisper model name |
| `WHISPER_DEVICE` | auto | `cpu` / `cuda` |
| `WHISPER_DOWNLOAD_ROOT` | — | Model cache directory |
| `WHISPER_LOG_LEVEL` | `INFO` | Log level |
