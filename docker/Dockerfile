FROM python:3.13-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only the Python source needed — no heavy deps (scipy, PyQt, etc).
# All modules used by extract_lfe.py are stdlib-only.
COPY src/main/python/model/wav_integrity.py src/main/python/model/
COPY src/main/python/model/auto_beq_catalogue.py src/main/python/model/
COPY src/main/python/model/media_constants.py src/main/python/model/
# No spike/ modules needed — extract_lfe.py imports only from model.*

# Make Python packages importable.
RUN touch src/main/python/model/__init__.py

COPY scripts/extract_lfe.py scripts/
COPY scripts/verify_wav_cache.py scripts/
# wav_cache_status.py needs numpy (via spike imports) — run on dev machine, not Docker

ENV PYTHONPATH=/app/src/main/python:/app/src/test/python

ENTRYPOINT ["python3", "scripts/extract_lfe.py"]
