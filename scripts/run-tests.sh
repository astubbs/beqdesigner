#!/usr/bin/env bash
# Run the full default test suite.
#
# Two-pass execution: torch tests run in a separate pytest invocation
# to avoid a SIGSEGV caused by torch + PyQt6 sharing a process on macOS
# (torch's MPS backend conflicts with Qt's Metal usage).
#
# Usage:
#   scripts/run-tests.sh              # default suite (unit tests only)
#   scripts/run-tests.sh -m integration   # integration tests
#   scripts/run-tests.sh -m experiment    # experiment tests
#   scripts/run-tests.sh -k test_name     # specific test by name

set -euo pipefail

MARKER_ARGS=()
EXTRA_ARGS=()

# Default: exclude integration and experiment tests.
if [[ $# -eq 0 ]]; then
    MARKER_ARGS=(-m "not integration and not experiment")
else
    EXTRA_ARGS=("$@")
fi

echo "=== Pass 1: all tests except torch ==="
poetry run pytest \
    --ignore=src/test/python/spike/test_auto_beq_torch.py \
    "${MARKER_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" \
    -v

echo ""
echo "=== Pass 2: torch tests (separate process) ==="
poetry run pytest \
    src/test/python/spike/test_auto_beq_torch.py \
    "${EXTRA_ARGS[@]}" \
    -v

echo ""
echo "=== All tests passed ==="
