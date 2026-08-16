FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY alembic.ini ./alembic.ini
COPY app ./app
# Ensure we only ship assistant prompts in one canonical location:
# `/app/ragkeep/assistants` (matches `settings.assistants_root` default).
RUN rm -rf /app/app/assistants

# Public assistants from ragkeep (shallow). philo-von-freisinn is a private
# nested submodule — clone would need credentials — so overlay runtime files
# vendored under docker/assistants/ (prompts + manifest only).
RUN git clone --depth=1 --no-tags \
      https://github.com/cypherpunk-academy/ragkeep.git /tmp/ragkeep && \
    mkdir -p ./ragkeep && \
    cp -r /tmp/ragkeep/assistants ./ragkeep/assistants && \
    rm -rf /tmp/ragkeep ./ragkeep/assistants/philo-von-freisinn
COPY docker/assistants/philo-von-freisinn ./ragkeep/assistants/philo-von-freisinn
RUN test -f ./ragkeep/assistants/philo-von-freisinn/prompts/instruction.prompt

COPY entrypoint.sh ./entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 8000

CMD ["/app/entrypoint.sh"]
