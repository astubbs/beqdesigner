#!/usr/bin/env bash
# Wrapper so repeated spike-test runs don't each need a fresh approval.
# Env vars you can set before invoking:
#   AUTO_BEQ_ADVISOR      heuristic | mock | ollama (default: mock)
#   SPIKE_TEST            pytest selector (default: all spike tests)
#   SPIKE_VERBOSE         1 to include -s (stdout from tests)
#   OLLAMA_MODEL          e.g. llama3.1:8b (default: llama3.1:8b)
set -euo pipefail
cd "$(dirname "$0")/.."

export PATH="/opt/homebrew/bin:$PATH"
export PYTHONPATH="./src/main/python"
export QT_QPA_PLATFORM="offscreen"
export AUTO_BEQ_ADVISOR="${AUTO_BEQ_ADVISOR:-mock}"

args=("src/test/python/spike/")
if [[ -n "${SPIKE_TEST:-}" ]]; then
  args=("${SPIKE_TEST}")
fi
if [[ "${SPIKE_VERBOSE:-0}" == "1" ]]; then
  args+=("-s")
fi
args+=("-v")

poetry run pytest "${args[@]}"
