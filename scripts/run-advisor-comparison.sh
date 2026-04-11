#!/usr/bin/env bash
# Compare all advisor implementations side-by-side on the sweep corpus.
#
# Runs the sweep test with each advisor and prints a summary table.
# Uses propose_filters_from_measured (NOT propose_or_lookup) —
# catalogue is the answer key, never the shortcut.
#
# Usage:
#   bash scripts/run-advisor-comparison.sh
#   AUTO_BEQ_SWEEP_LIMIT=20 bash scripts/run-advisor-comparison.sh
set -euo pipefail
cd "$(dirname "$0")/.."

export PATH="/opt/homebrew/bin:$PATH"
export PYTHONPATH="./src/main/python:./src/test/python"
export QT_QPA_PLATFORM="offscreen"

ADVISORS="measurement topology slope_extension"
LOG_DIR=".pytest_cache"
mkdir -p "$LOG_DIR"

echo "=== Advisor comparison sweep ==="
echo "Advisors: $ADVISORS"
echo "Sweep limit: ${AUTO_BEQ_SWEEP_LIMIT:-10}"
echo ""

for advisor in $ADVISORS; do
    echo "── Running: $advisor ──"
    AUTO_BEQ_ADVISOR="$advisor" \
    poetry run pytest \
        src/test/python/spike/test_auto_beq_library_sweep.py::test_library_sweep \
        -v -s 2>&1 \
        | tee "$LOG_DIR/comparison_${advisor}.log" \
        | grep -E "verdict=|passed|failed|skipped" \
        | tail -5
    echo ""
done

echo "=== Summary ==="
for advisor in $ADVISORS; do
    logfile="$LOG_DIR/comparison_${advisor}.log"
    pass=$(grep -c "verdict=PASS" "$logfile" 2>/dev/null || true)
    marg=$(grep -c "verdict=MARGINAL" "$logfile" 2>/dev/null || true)
    fail=$(grep -c "verdict=FAIL" "$logfile" 2>/dev/null || true)
    pass=${pass:-0}; marg=${marg:-0}; fail=${fail:-0}
    total=$((pass + marg + fail))
    printf "  %-20s  %d PASS  %d MARGINAL  %d FAIL  (of %d)\n" "$advisor" "$pass" "$marg" "$fail" "$total"
done
