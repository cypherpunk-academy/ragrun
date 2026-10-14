FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y ca-certificates curl build-essential git \
    && curl -fsSL https://pkgs.tailscale.com/stable/debian/bookworm.noarmor.gpg \
      -o /usr/share/keyrings/tailscale-archive-keyring.gpg \
    && curl -fsSL https://pkgs.tailscale.com/stable/debian/bookworm.tailscale-keyring.list \
      -o /etc/apt/sources.list.d/tailscale.list \
    && apt-get update && apt-get install -y tailscale \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY alembic.ini ./alembic.ini
COPY app ./app
# Ensure we only ship assistant prompts in one canonical location:
# `/app/ragkeep/assistants` (matches `settings.assistants_root` default).
RUN rm -rf /app/app/assistants

# Clone public assistants. philo-von-freisinn is a nested submodule of
# ragkeep, so a shallow clone leaves it empty — clone it explicitly.
RUN git clone --depth=1 --no-tags \
      https://github.com/cypherpunk-academy/ragkeep.git /tmp/ragkeep && \
    rm -rf /tmp/ragkeep/assistants/philo-von-freisinn && \
    git clone --depth=1 --no-tags \
      https://github.com/cypherpunk-academy/philo-von-freisinn.git \
      /tmp/ragkeep/assistants/philo-von-freisinn && \
    test -f /tmp/ragkeep/assistants/philo-von-freisinn/prompts/instruction.prompt && \
    mkdir -p ./ragkeep && \
    cp -r /tmp/ragkeep/assistants ./ragkeep/assistants && \
    rm -rf /tmp/ragkeep

COPY entrypoint.sh ./entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 8000

CMD ["/app/entrypoint.sh"]
