git submodule update --init --remote ragkeep
docker compose build --no-cache ragrun-api && docker compose up -d 