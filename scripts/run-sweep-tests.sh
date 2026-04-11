#!/usr/bin/env bash
# Run the auto-BEQ pipeline across your discovered media library.
#
# Usage:
#   bash scripts/run-sweep-tests.sh              # default: process up to 10 files
#   AUTO_BEQ_SWEEP_LIMIT=50 bash scripts/run-sweep-tests.sh  # process up to 50
#   bash scripts/run-sweep-tests.sh --parallel    # use all configured Ollama hosts
#
# Prerequisite: run `bash scripts/run-sweep-discover.sh` first to discover
# your media and match it against the BEQ catalogue.
set -euo pipefail
cd "$(dirname "$0")/.."

export PATH="/opt/homebrew/bin:$PATH"
export PYTHONPATH="./src/main/python:./src/test/python"
export QT_QPA_PLATFORM="offscreen"
export AUTO_BEQ_ADVISOR="${AUTO_BEQ_ADVISOR:-measurement}"

# Log to file as well as terminal so user can tail it.
LOG_FILE=".pytest_cache/sweep_run.log"
mkdir -p .pytest_cache

if [[ "${1:-}" == "--parallel" ]]; then
    test_selector="src/test/python/spike/test_auto_beq_library_sweep.py::test_library_sweep_parallel"
    shift
else
    test_selector="src/test/python/spike/test_auto_beq_library_sweep.py::test_library_sweep"
fi

echo "[run-sweep-tests] advisor=$AUTO_BEQ_ADVISOR limit=${AUTO_BEQ_SWEEP_LIMIT:-10}"
echo "[run-sweep-tests] log file: $LOG_FILE (tail -f $LOG_FILE in another terminal)"
echo ""

poetry run pytest "$test_selector" -v -s "$@" 2>&1 | tee "$LOG_FILE"
