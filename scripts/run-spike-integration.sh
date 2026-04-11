#!/usr/bin/env bash
# Integration spike test runner — tests that touch real external resources.
#
# Runs only tests marked `@pytest.mark.integration`. These need at least
# some of the following to be useful:
#   - real media files on disk (library sweep)
#   - network access to the TMDb API (metadata tests)
#   - running Ollama hosts (LLM advisor tests)
#   - a populated `~/.config/beqdesigner/auto_beq_sweep.json` (library sweep
#     — run `poetry run python -m spike.sweep_discover` first to generate it)
#
# Runs that can't find their resources skip rather than fail thanks to
# per-test `@pytest.mark.skipif` decorators. Set env vars to point at your
# local setup.
#
# Supported env vars (defaults shown):
#   AUTO_BEQ_ADVISOR=measurement
#   AUTO_BEQ_SWEEP_CONFIG=~/.config/beqdesigner/auto_beq_sweep.json
#   AUTO_BEQ_SWEEP_LIMIT=10
#   SPIKE_TEST=src/test/python/spike/
#   SPIKE_VERBOSE=0
set -euo pipefail
cd "$(dirname "$0")/.."

export PATH="/opt/homebrew/bin:$PATH"
export PYTHONPATH="./src/main/python:./src/test/python"
export QT_QPA_PLATFORM="offscreen"
export AUTO_BEQ_ADVISOR="${AUTO_BEQ_ADVISOR:-measurement}"

target="${SPIKE_TEST:-src/test/python/spike/}"
verbose_flag=""
if [[ "${SPIKE_VERBOSE:-0}" == "1" ]]; then
  verbose_flag="-s"
fi

LOG_FILE=".pytest_cache/spike_integration.log"
mkdir -p .pytest_cache

echo "[run-spike-integration] advisor=$AUTO_BEQ_ADVISOR target=$target"
echo "[run-spike-integration] log file: $LOG_FILE"
echo "[run-spike-integration] NOTE: tests that don't find their resources will skip"
poetry run pytest "$target" -v -m integration $verbose_flag 2>&1 | tee "$LOG_FILE"
