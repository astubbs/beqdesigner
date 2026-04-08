#!/usr/bin/env bash
# Wrapper so repeated spike-test runs don't each need a fresh approval.
# Edit the env vars below or set them via a wrapper shell - the invocation
# itself (`bash scripts/run-spike-tests.sh`) is always identical, so the
# harness only needs to approve it once.
#
# Supported env vars (defaults shown):
#   AUTO_BEQ_ADVISOR=measurement       # heuristic | measurement | mock | ollama
#   SPIKE_TEST=src/test/python/spike/  # pytest selector
#   SPIKE_VERBOSE=0                    # 1 = include -s
#   OLLAMA_MODEL=llama3.1:8b
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

LOG_FILE=".pytest_cache/spike_tests.log"
mkdir -p .pytest_cache

echo "[run-spike-tests] advisor=$AUTO_BEQ_ADVISOR target=$target"
echo "[run-spike-tests] log file: $LOG_FILE"
poetry run pytest "$target" -v $verbose_flag 2>&1 | tee "$LOG_FILE"
