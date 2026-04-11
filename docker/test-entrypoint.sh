#!/usr/bin/env bash
# test-entrypoint.sh — dispatcher that runs the spike test suites in
# the right order based on TEST_SCOPE.
#
# TEST_SCOPE values:
#   unit         → run-spike-tests.sh                   (~1 min)
#   integration  → run-spike-tests.sh + run-spike-integration.sh
#   all          → run-spike-tests.sh + run-spike-integration.sh + run-spike-experiments.sh
#                  (default; can take minutes to hours — see AGENTS.md:173)
#
# Exit code semantics: every stage runs even if an earlier one failed
# (so we collect all three log files on a single build), and the final
# exit code is the worst (max) exit code seen across the stages. This
# means a unit-test failure does NOT mask whether the experiment suite
# also breaks — both failures get reported on the same run.
#
# Designed to be invoked as the ENTRYPOINT of docker/Dockerfile.test,
# but also runnable directly on any host that has the repo + poetry.
set -u
cd "$(dirname "$0")/.."

SCOPE="${TEST_SCOPE:-all}"
worst=0

run_stage() {
    local label="$1" script="$2"
    echo "::group::${label}"
    echo "[test-entrypoint] starting ${label} (scope=${SCOPE})"
    if bash "${script}"; then
        echo "[test-entrypoint] ${label} passed"
    else
        local rc=$?
        echo "[test-entrypoint] ${label} FAILED (exit=${rc})"
        if (( rc > worst )); then
            worst=${rc}
        fi
    fi
    echo "::endgroup::"
}

case "${SCOPE}" in
    unit)
        run_stage "spike-unit" "scripts/run-spike-tests.sh"
        ;;
    integration)
        run_stage "spike-unit" "scripts/run-spike-tests.sh"
        run_stage "spike-integration" "scripts/run-spike-integration.sh"
        ;;
    all|"")
        run_stage "spike-unit" "scripts/run-spike-tests.sh"
        run_stage "spike-integration" "scripts/run-spike-integration.sh"
        run_stage "spike-experiments" "scripts/run-spike-experiments.sh"
        ;;
    *)
        echo "[test-entrypoint] unknown TEST_SCOPE='${SCOPE}' (expected: unit|integration|all)" >&2
        exit 64
        ;;
esac

echo "[test-entrypoint] done, worst exit code = ${worst}"
exit "${worst}"
