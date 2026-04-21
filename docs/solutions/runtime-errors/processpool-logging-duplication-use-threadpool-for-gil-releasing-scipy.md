---
title: ProcessPoolExecutor causes logging spam - use ThreadPoolExecutor for scipy/numpy
date: 2026-04-20
category: runtime-errors
module: cli-training-pipeline
problem_type: runtime_error
component: tooling
severity: medium
symptoms:
  - "Every log message printed N times (once per ProcessPoolExecutor worker) during beq-designer dev train"
  - "Startup banner, config validation, and discovery messages all repeated for each parallel worker"
  - "Switching to spawn start context did not resolve the duplication"
  - "Logging initializer suppression did not help because it runs after module imports"
root_cause: config_error
resolution_type: code_fix
related_components:
  - background_job
tags:
  - multiprocessing
  - threading
  - logging
  - processpool
  - threadpool
  - scipy
  - numpy
  - gil
  - python
  - cli
---

# ProcessPoolExecutor causes logging spam - use ThreadPoolExecutor for scipy/numpy

## Problem

Running `bin/beq-designer dev train` produced every log message N times (once per worker process). The training pipeline used `ProcessPoolExecutor` for parallel WAV feature extraction (scipy Welch FFT), and each forked/spawned child process re-imported modules containing `logging.basicConfig()` at module level. This made the CLI unusable - the banner, config validation, and discovery messages all repeated for each worker, producing walls of duplicate output.

## Symptoms

- Banner (`BEQ Designer CLI`) printed 10+ times at training start
- "checking WAV cache configuration...", "validating media root(s)...", "checking for production model..." repeated per worker
- "discover_wav_catalogue_pairs: cache hit" and "BEQ working directory" messages repeated
- Terminal output became unreadable within seconds of training starting
- The problem occurred both when launching `dev train` from the interactive menu and as a direct CLI subcommand

## What Didn't Work

- **`mp.get_context("spawn")` on the ProcessPoolExecutor**: Spawned children start clean but still re-import all modules. Any module with `logging.basicConfig()` at the top level adds a new handler in every child. The spam source shifted from "inherited handlers" to "freshly-added handlers" but the effect was identical. (session history)

- **`initializer=_suppress_worker_logging` on the ProcessPoolExecutor**: The initializer function runs AFTER module imports complete. By the time `logging.disable(CRITICAL)` executes in the child, the module-level `basicConfig()` has already added a handler and the initial import-time log messages have already been emitted.

- **Local initializer function with `spawn` context**: Defined `_worker_init` as a local function inside the extraction function. `spawn` requires pickling the initializer, but local functions can't be pickled - raised `AttributeError: Can't get local object '_extract_features_parallel.<locals>._worker_init'`. Moving to module level fixed the pickle error but didn't fix the logging spam.

- **Moving `basicConfig()` inside `main()` with handler guard**: `if not logging.getLogger().handlers: basicConfig(...)` correctly prevented double-adding in the parent process, but `spawn` children start with a fresh logger (no handlers), so the guard passes in every child and adds a handler anyway. (session history)

## Solution

Replace `ProcessPoolExecutor` with `ThreadPoolExecutor`. scipy and numpy release the GIL during C-level computation (Welch FFT, smoothing, array operations), so threads get real parallelism without any fork/spawn/re-import overhead.

**Before:**

```python
from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor(max_workers=max_workers) as executor:
    for wav_str, features in executor.map(_extract_one_wav, work):
        results[wav_str] = features
```

**After:**

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    for wav_str, features in executor.map(_extract_one_wav, work):
        results[wav_str] = features
```

Also moved `logging.basicConfig()` from module level to inside `main()` as defense-in-depth:

**Before** (module level in `train_production_model.py`):

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s - %(message)s",
)
```

**After** (inside `main()`):

```python
def main(argv=None):
    ...
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)-5s %(name)s - %(message)s",
        )
```

## Why This Works

The root cause had two layers:

1. **Module-level `logging.basicConfig()`**: Python's `basicConfig()` adds a `StreamHandler` to the root logger. When called at module scope, it executes during import - every process that imports the module gets its own handler. With `ProcessPoolExecutor`, each of N workers imports the module independently, creating N handlers total.

2. **Wrong parallelism primitive**: `ProcessPoolExecutor` creates separate OS processes. Each process has its own Python interpreter, its own module imports, its own logging handlers. `ThreadPoolExecutor` creates threads within the SAME process - one set of handlers, one interpreter, shared memory.

The key insight: scipy and numpy release the GIL during their C-level computation (FFT, array operations, smoothing). This means `ThreadPoolExecutor` gives true parallel execution for these workloads - threads aren't serialized by the GIL because the actual computation happens in C extensions that explicitly release it. There's no performance reason to use processes here.

The pattern generalizes: **for any Python workload where the heavy computation is in C extensions that release the GIL (numpy, scipy, pandas, pillow, etc.), `ThreadPoolExecutor` is strictly better than `ProcessPoolExecutor`** - same parallelism, no fork/spawn overhead, no module re-import side effects. (auto memory [claude])

## Prevention

- **Never place `logging.basicConfig()` at module level.** Put it inside `main()` or behind a `if not logging.getLogger().handlers:` guard. Module-level side effects are invisible landmines for any code that uses multiprocessing.

- **Default to `ThreadPoolExecutor` for scipy/numpy/pandas workloads.** Only use `ProcessPoolExecutor` when the work is pure Python (GIL-bound) with no C extension computation. The comment in the original code said "Uses ProcessPoolExecutor since scipy Welch is single-threaded" - this reasoning was wrong. scipy Welch IS single-threaded per call, but it releases the GIL, so multiple threads calling it concurrently get true parallelism.

- **When using `ProcessPoolExecutor`, always use `spawn` context + `initializer` that disables logging.** If processes are genuinely needed (pure Python work), both protections are required because `spawn` re-imports modules (triggering module-level code) and `fork` inherits parent state (including handlers).

- **Test parallel code paths with N>1 workers.** The logging spam only manifested with multiple workers. A single-worker test would have passed silently.

## Related Issues

- Same class of problem appeared in `run_e86_experiment.py` - module-level code without `__name__ == "__main__"` guard re-executed in spawn children (session history)
- `pytest-forked` was tried for Qt/Torch test isolation and failed similarly - forked processes inherit Qt state (session history)
- XGBoost training was parallelized with `ThreadPoolExecutor` and `n_jobs=1` inside XGBoost to prevent nested parallelism contention (session history)

**Files changed:**
- `src/main/python/model/audio_extraction.py` (was `test_auto_beq_nn_real.py`) - ProcessPoolExecutor -> ThreadPoolExecutor
- `src/main/python/cli/train_production_model.py` - module-level basicConfig -> inside main()
- `src/main/python/cli/train_torch_model.py` - same
- `src/main/python/model/auto_beq_nn.py` - spawn context added (separate ProcessPoolExecutor site)
