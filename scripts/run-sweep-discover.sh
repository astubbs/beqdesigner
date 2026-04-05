#!/usr/bin/env bash
# Run the library sweep discovery CLI.
#
# Usage:
#   bash scripts/run-sweep-discover.sh --library /path/to/movies [--yes] [--refresh]
#
# Sets PYTHONPATH so the spike package and model package are importable,
# then forwards all arguments to sweep_discover.main().
set -euo pipefail
cd "$(dirname "$0")/.."

export PATH="/opt/homebrew/bin:$PATH"
export PYTHONPATH="./src/main/python:./src/test/python"
export QT_QPA_PLATFORM="offscreen"

poetry run python -m spike.sweep_discover "$@"
