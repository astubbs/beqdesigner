---
title: CLI/Docker crashes with libEGL.so.1 not found - decouple model layer from Qt
date: 2026-04-20
category: runtime-errors
module: model-layer
problem_type: runtime_error
component: tooling
severity: high
symptoms:
  - "ImportError: libEGL.so.1: cannot open shared object file in Docker container"
  - "Profile generation crashed on headless Docker/NAS with no display server"
  - "Importing cli/generate.py pulled in the entire PyQt6 GUI stack"
root_cause: wrong_api
resolution_type: code_fix
tags:
  - qt
  - pyqt6
  - docker
  - headless
  - import-chain
  - decoupling
  - cli
---

# CLI/Docker crashes with libEGL.so.1 not found - decouple model layer from Qt

## Problem

Running profile generation in a headless Docker container crashed with `ImportError: libEGL.so.1` because the import chain `cli/generate.py` -> `model/auto_beq.py` -> `model/iir.py` -> `model/xy.py` -> `model/preferences.py` -> `qtawesome` -> `PyQt6` required a display server that doesn't exist in Docker.

The actual computation (biquad filters, transfer functions, complex data) had zero Qt dependency. The contamination came from `model/xy.py` importing `Preferences` from `model/preferences.py` and instantiating `QSettings` at module level.

## Symptoms

- `docker compose run beq-designer` showed a Rich traceback ending in `ImportError: libEGL.so.1`
- The traceback showed the chain: `profile.py:33` -> `generate.py:23` -> `auto_beq.py:21` -> `iir.py:13` -> `xy.py:6` -> `preferences.py:8` -> `qtawesome` -> `PyQt6`
- All CLI commands that touched profile generation failed in Docker
- The GUI app still worked fine (Qt is available on the desktop)

## What Didn't Work

- **Lazy import in cli/profile.py**: Changed `from cli.generate import generate_profile` to a function-level import inside `_run_single()`. This was a band-aid - it deferred the crash from startup to when the user selected "Generate profile" from the menu, but the underlying import chain was still broken.

## Solution

Extracted the pure data classes (`ComplexData`, `MagnitudeData`) and smoothing functions from `model/xy.py` into a new Qt-free module `model/xy_data.py`:

```python
# model/xy_data.py - pure numpy, no Qt
class MagnitudeData:
    ...

class ComplexData:
    ...

def smooth(x, y, smooth_type=None):
    ...
```

`model/xy.py` now re-exports from `xy_data` for backward compatibility with GUI code:

```python
# model/xy.py - keeps Qt-dependent interp() and QSettings
from model.xy_data import ComplexData, MagnitudeData, smooth  # re-export

preferences = Preferences(QSettings("3ll3d00d", "beqdesigner"))

def interp(x1, y1, x2):  # Qt-dependent (reads smooth preference)
    ...
```

`model/iir.py` imports from `xy_data` instead of `xy`, breaking the Qt chain:

```python
# Before:
from model.xy import ComplexData  # pulls in Qt

# After:
from model.xy_data import ComplexData  # Qt-free
```

## Why This Works

The Qt dependency was incidental - `iir.py` only needed `ComplexData` (a pure numpy data container), but importing it from `xy.py` triggered `xy.py`'s module-level `preferences = Preferences(QSettings(...))` which required PyQt6.

By extracting the pure data classes to `xy_data.py`, the CLI import chain becomes: `generate.py` -> `auto_beq.py` -> `iir.py` -> `xy_data.py` (numpy only). The Qt-dependent `xy.py` is only imported by GUI code that already has Qt available.

The `MagnitudeData.normalise()` and `.filter()` methods call `interp()` from `xy.py` via lazy import (`from model.xy import interp`) inside the method body. These methods are only called by GUI code, so the lazy import is safe.

## Prevention

- **Never put Qt/GUI imports or instantiation at module level in `model/` code.** The `model/` layer must be importable without a display server.
- **AGENTS.md rule added**: "CLI and service code must never depend on Qt. The model/ layer and cli/ layer must be importable without PyQt6."
- **Verify with**: `python -c "from model.iir import HighShelf, LowShelf, PeakingEQ"` - must work without Qt/display.
- **The `check_production_model()` function was also moved** from `cli/profile.py` (which imports Qt) to `spike/_auto_beq_helpers.py` (Qt-free) for the same reason.

## Related Issues

- The original `model/preferences.py` is a Qt preferences dialog - it's inherently GUI code. The problem was that pure computation modules depended on it for a single constant (`DISPLAY_SMOOTH_GRAPHS`) and a module-level `Preferences` instance.
