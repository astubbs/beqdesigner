#!/usr/bin/env bash
# Experiment spike test runner — retrains models from scratch.
#
# Runs only tests marked `@pytest.mark.experiment`: the F/G/H/I experiment
# batches (`test_auto_beq_nn_experiments.py`), real-audio training regime
# (`test_auto_beq_nn_real.py`), chunked-strategy comparison
# (`test_auto_beq_nn_chunked.py`), and E31 extract+validate
# (`test_auto_beq_nn_extract.py`). Each test retrains one or more models
# end-to-end — minutes per test, sometimes 30+ minutes for the full
# `test_g_experiment_comparison` sweep.
#
# Not for CI. Run on demand when iterating on model config or reproducing
# an experiment from the log.
#
# Supported env vars (defaults shown):
#   AUTO_BEQ_ADVISOR=measurement
#   SPIKE_TEST=src/test/python/spike/  # narrow to a specific test file/id
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

LOG_FILE=".pytest_cache/spike_experiments.log"
mkdir -p .pytest_cache

echo "[run-spike-experiments] advisor=$AUTO_BEQ_ADVISOR target=$target"
echo "[run-spike-experiments] log file: $LOG_FILE"
echo "[run-spike-experiments] WARNING: retrains models — minutes to hours per test"
poetry run pytest "$target" -v -m experiment $verbose_flag 2>&1 | tee "$LOG_FILE"
