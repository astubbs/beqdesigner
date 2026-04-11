#!/usr/bin/env bash
# Fast unit-only spike test runner.
#
# Filters out tests marked `integration` (need real media/TMDb/Ollama) and
# `experiment` (minutes-long model retraining) so the default run stays
# hermetic and quick. Use the sibling scripts to opt into the slower groups:
#   scripts/run-spike-integration.sh  — `-m integration`
#   scripts/run-spike-experiments.sh  — `-m experiment`
#
# Repeated runs share a single permission approval because the invocation
# (`bash scripts/run-spike-tests.sh`) is always identical; configure via
# env vars instead of positional args.
#
# Supported env vars (defaults shown):
#   AUTO_BEQ_ADVISOR=measurement       # heuristic | measurement | mock | ollama
#   SPIKE_TEST=src/test/python/spike/  # pytest selector
#   SPIKE_VERBOSE=0                    # 1 = include -s
#   SPIKE_MARKERS="not integration and not experiment"   # override marker filter
#   OLLAMA_MODEL=llama3.1:8b
set -euo pipefail
cd "$(dirname "$0")/.."

export PATH="/opt/homebrew/bin:$PATH"
export PYTHONPATH="./src/main/python:./src/test/python"
export QT_QPA_PLATFORM="offscreen"
export AUTO_BEQ_ADVISOR="${AUTO_BEQ_ADVISOR:-measurement}"

target="${SPIKE_TEST:-src/test/python/spike/}"
marker_filter="${SPIKE_MARKERS:-not integration and not experiment}"
verbose_flag=""
if [[ "${SPIKE_VERBOSE:-0}" == "1" ]]; then
  verbose_flag="-s"
fi

LOG_FILE=".pytest_cache/spike_tests.log"
mkdir -p .pytest_cache

echo "[run-spike-tests] advisor=$AUTO_BEQ_ADVISOR target=$target"
echo "[run-spike-tests] markers: $marker_filter"
echo "[run-spike-tests] log file: $LOG_FILE"
poetry run pytest "$target" -v -m "$marker_filter" $verbose_flag 2>&1 | tee "$LOG_FILE"
