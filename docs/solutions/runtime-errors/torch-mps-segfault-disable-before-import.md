---
title: torch SIGSEGV on macOS - disable MPS before importing torch
date: 2026-04-20
category: runtime-errors
module: ml-training
problem_type: runtime_error
component: tooling
severity: critical
symptoms:
  - "SIGSEGV (Address boundary error) when E85 differentiable DSP runs after E82/E83/E84"
  - "Crash at 'building target responses for 448 training entries' - first torch.tensor() call"
  - "Only crashes in combined runs, not when E85 runs in isolation"
  - "device='cpu' explicitly set but crash still occurs"
root_cause: config_error
resolution_type: code_fix
tags:
  - torch
  - mps
  - metal
  - macos
  - segfault
  - sigsegv
  - gpu
  - apple-silicon
---

# torch SIGSEGV on macOS - disable MPS before importing torch

## Problem

Running `bin/beq-designer dev reassess` crashed with SIGSEGV during the E85 differentiable DSP experiment. The crash occurred at the first `torch.tensor()` call after E82 (XGBoost), E83 (Whisper), and E84 (self-training) had already completed in the same process. E85 worked perfectly when run in isolation.

## Symptoms

- `fish: Job 1, './bin/beq-designer' terminated by signal SIGSEGV (Address boundary error)`
- Crash at "E85 prep: building target responses for 448 training entries" - the `torch.tensor()` call
- E85 config explicitly sets `device="cpu"` but crashes anyway
- No crash when E85 runs alone (no prior experiments in the same process)
- No crash when running via the two-pass test runner (separate pytest invocation)

## What Didn't Work

- **Checking for Qt imports**: Confirmed zero Qt modules loaded in the CLI path. The `model/xy_data.py` decoupling from the Qt refactor was working correctly. This was a red herring. (session history)

- **Running E85 in isolation**: No segfault, confirming it's a process-state interaction, not a standalone torch bug.

- **Checking for scipy/numpy stack overflow**: Built 448 target responses with `np.stack` in isolation - no crash. The data itself was fine.

- **`pytest-forked`**: Tried earlier for the torch+PyQt6 variant of this problem. `os.fork()` itself conflicts with macOS MPS inside torch - the fork inherits the Metal state and the child segfaults. (session history)

- **Setting `device="cpu"` in the training config**: E85 already used `device="cpu"` but torch's MPS backend auto-initializes on `import torch` regardless of the requested device. The env var must be set BEFORE the import. (session history)

- **Putting the env var in the experiment script**: Initially placed `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` in `experiments/run_tier1_comparison.py`. User correctly flagged this as a red flag: "If you have to set anything special in a script separate from the main code, it's a red flag." Scripts should be thin frontends. Reverted immediately.

- **Uninstalling numba/llvmlite**: An earlier session found that numba's bundled LLVM conflicted with xgboost's native libs. This fixed THAT specific crash but was unrelated to the MPS issue. (session history)

## Solution

Set `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` in `model/auto_beq_torch.py:_maybe_import_nn()` - the lazy-import gate that all torch code goes through - BEFORE importing torch:

```python
def _maybe_import_nn():
    """Lazy import of torch.nn so this module can be imported without torch.

    Disables MPS (Metal Performance Shaders) on macOS to prevent SIGSEGV
    when torch shares a process with scipy/xgboost that also use Metal.
    """
    import os as _os
    _os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    try:
        import torch
        import torch.nn as nn
        return torch, nn
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for E85 differentiable DSP.",
        ) from exc
```

## Why This Works

On macOS with Apple Silicon, torch auto-detects the MPS (Metal Performance Shaders) backend and initializes it during `import torch`, even when the caller specifies `device="cpu"`. When other libraries (scipy, xgboost) have already used Metal resources in the same process, torch's MPS initialization conflicts with the existing Metal state and crashes with SIGSEGV.

Setting `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` before the import tells torch not to allocate any GPU memory for MPS, effectively disabling the Metal backend. torch still imports and works, but only on CPU - which is what E85 requests anyway.

The env var must be set BEFORE `import torch` because MPS initialization happens during the import, not when tensors are first allocated. Setting it after import has no effect.

The fix is in `_maybe_import_nn()` rather than in a script because:
1. `_maybe_import_nn()` is the single gate function that all torch imports go through
2. Any caller - the tier1 comparison, the production trainer, unit tests - gets the fix automatically
3. Scripts are thin frontends; workarounds belong in the library that owns the dependency (auto memory [claude])

## Prevention

- **Environment workarounds for library imports belong in the model/library layer, not in scripts.** The library code that imports the dependency is responsible for configuring it safely. Callers should not need to know about import-time side effects.

- **Test torch code in a separate process on macOS.** The two-pass test runner (`bin/beq-designer dev test`) already handles this: pass 1 runs non-torch tests, pass 2 runs torch tests in a fresh process. This avoids the MPS conflict entirely for tests.

- **When `device="cpu"` isn't enough, check for import-time GPU initialization.** Many ML frameworks (torch, JAX, TensorFlow) auto-detect GPUs on import. If the GPU backend conflicts with other process state, the `device` parameter alone won't help - you need to disable the backend before import.

- **Document the MPS workaround in the module docstring.** Future developers modifying `_maybe_import_nn()` need to understand why the env var is there and that removing it will cause SIGSEGV on macOS.

## Related Issues

- `docs/solutions/runtime-errors/processpool-logging-duplication-use-threadpool-for-gil-releasing-scipy.md` - same module (`auto_beq_torch.py`) had a parallel ProcessPoolExecutor issue, solved differently (ThreadPoolExecutor)
- `docs/solutions/runtime-errors/qt-dependency-chain-blocks-cli-docker-extract-xy-data.md` - the Qt decoupling work was initially suspected as the segfault cause but confirmed to be working correctly
- The two-pass test runner in `cli/main.py` (torch tests in separate pytest invocation) is the test-time equivalent of this fix
- Earlier sessions traced the same segfault through three different root-cause threads (torch+XGBoost dylib, numba/llvmlite LLVM, MPS auto-init) before converging on MPS as the definitive cause (session history)

**Files changed:**
- `src/main/python/model/auto_beq_torch.py:_maybe_import_nn()` - added env var before torch import
