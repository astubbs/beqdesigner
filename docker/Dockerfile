FROM python:3.13-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the full source tree — simpler than cherry-picking files.
COPY src/main/python/ src/main/python/
COPY src/test/python/ src/test/python/
COPY bin/ bin/

# Make Python packages importable.
RUN touch src/main/python/model/__init__.py \
    && touch src/main/python/cli/__init__.py

ENV PYTHONPATH=/app/src/main/python:/app/src/test/python

ENTRYPOINT ["python3", "src/main/python/cli/extract.py"]
