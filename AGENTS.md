# Agent instructions

## Running spike tests

**Always use `bash scripts/run-spike-tests.sh` to run spike tests.** Never
invoke `poetry run pytest` directly for spike tests. The wrapper exists
so that repeated runs share a single permission approval — each
distinct `poetry run pytest ...` command line requires a fresh
approval, which is disruptive during iteration.

The wrapper honours these env vars (set them inline on the same line):

| Var | Purpose | Default |
|---|---|---|
| `SPIKE_TEST` | pytest selector (file path or nodeid) | `src/test/python/spike/` (all) |
| `AUTO_BEQ_ADVISOR` | advisor impl: heuristic / mock / ollama / measurement | `mock` |
| `SPIKE_VERBOSE` | `1` enables `-s` (no capture) | `0` |
| Any test-specific env var | passed through to pytest | — |

Canonical invocation pattern (works from the repo root):

```bash
SPIKE_TEST=src/test/python/spike/test_auto_beq.py \
  bash scripts/run-spike-tests.sh
```

For a single test within a file, use `::`:

```bash
SPIKE_TEST='src/test/python/spike/test_auto_beq.py::test_synthetic_roundtrip' \
  bash scripts/run-spike-tests.sh
```

Stack env vars for configuration:

```bash
AUTO_BEQ_SWEEP_CONFIG=/tmp/sweep.json \
AUTO_BEQ_SWEEP_LIMIT=3 \
AUTO_BEQ_ADVISOR=measurement \
SPIKE_TEST=src/test/python/spike/test_auto_beq_library_sweep.py \
  bash scripts/run-spike-tests.sh
```

**Do not** invoke the venv python, `poetry run python`, or `poetry run
pytest` directly for spike tests. If a new script/runner is needed, add
it under `scripts/` with a fixed command line.

## Verifying behaviour

**Always verify behaviour with a test, not ad-hoc code.** When checking
that something works (edge cases, parsing, formatting, etc.), write a
test that captures the expectation. Don't run throwaway Python snippets
or inline assertions — if it's worth verifying, it's worth a test.
