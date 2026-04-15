FROM python:3.13-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg git \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir poetry

WORKDIR /app

# 1. Install dependencies first (cached unless pyproject.toml/lock change).
COPY pyproject.toml poetry.lock ./
# Poetry needs at least the package source dir to exist for install.
RUN mkdir -p src/main/python
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --only main --no-root

# 2. Copy source (changes here don't re-install deps).
COPY src/ src/
COPY bin/ bin/
COPY build/ build/

# 3. Install the project itself (fast — deps already installed).
RUN poetry install --no-interaction --only main

# Make entry point executable.
RUN chmod +x bin/beq-designer

ENV PYTHONPATH=/app/src/main/python:/app/src/test/python
# Tell bin/beq-designer we're already in the venv (no poetry re-exec).
ENV VIRTUAL_ENV=/usr/local
# Default BEQ shared working directory (matches compose volume mount).
ENV BEQ_SHARED_DIR=/beq-shared
# Backward compat alias.
ENV BEQ_DIR=/beq-shared
# Auto-discover media roots under /media (all mounted volumes).
ENV BEQ_MEDIA_DIR=/media

ENTRYPOINT ["python3", "bin/beq-designer"]
CMD []
