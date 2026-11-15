#!/usr/bin/env bash
# Start ragscribe on the host (pyenv reniets). Model loads once, then stays in memory.
# Binds 0.0.0.0 so phones on the LAN can reach the server.
set -euo pipefail
cd "$(dirname "$0")"
export PYENV_VERSION="${PYENV_VERSION:-reniets}"
export WHISPER_MODEL="${WHISPER_MODEL:-medium}"
export WHISPER_LOG_LEVEL="${WHISPER_LOG_LEVEL:-INFO}"

if ! python -c 'import fastapi, uvicorn, whisper, yt_dlp' 2>/dev/null; then
  echo "Installing ragscribe deps into ${PYENV_VERSION} …" >&2
  pip install -r requirements.txt
fi

if ! python -c 'import whisperx' 2>/dev/null; then
  echo "whisperx fehlt in ${PYENV_VERSION} — Diarize fällt auf openai-whisper zurück." >&2
  echo "Install: PYENV_VERSION=${PYENV_VERSION} pip install 'whisperx>=3.3.0'" >&2
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg fehlt im PATH (für yt-dlp Audio-Extraktion)." >&2
  exit 1
fi

echo "ragscribe listening on http://0.0.0.0:8002 (phone: http://<LAN-IP>:8002)" >&2
exec uvicorn app.main:app --host 0.0.0.0 --port 8002 --timeout-keep-alive 43200
