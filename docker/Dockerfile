FROM python:3.13-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg git \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir poetry

WORKDIR /app

# Copy project files for poetry install.
COPY pyproject.toml poetry.lock ./
COPY src/ src/
COPY bin/ bin/
COPY build/ build/

# Install dependencies (no dev deps, no interactive prompts).
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --only main

# Make entry point executable.
RUN chmod +x bin/beq-designer

ENV PYTHONPATH=/app/src/main/python:/app/src/test/python
# Tell bin/beq-designer we're already in the venv (no poetry re-exec).
ENV VIRTUAL_ENV=/usr/local
# Default BEQ working directory (matches compose volume mount).
ENV BEQ_DIR=/beq

ENTRYPOINT ["python3", "bin/beq-designer"]
CMD ["--help"]
