#!/usr/bin/env python3
"""Regenerate Python sources from Qt Designer ``.ui`` files.

Run via ``poetry run ui-gen`` (see ``pyproject.toml``) or directly:

    poetry run python scripts/regen_ui.py [path-to-ui-dir]

By default the script scans ``src/main/python/ui`` relative to the repository
root, invokes ``pyuic6`` for every ``.ui`` file it finds, and writes the
matching ``.py`` next to it. The script is idempotent -- running it on a clean
checkout should produce no diff.

Pre-commit hook and CI sync check both call this module to guarantee the
committed ``.py`` files never drift from their ``.ui`` sources.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_UI_DIR = REPO_ROOT / "src" / "main" / "python" / "ui"


def regenerate(ui_dir: Path) -> int:
    if not ui_dir.is_dir():
        print(f"error: {ui_dir} is not a directory", file=sys.stderr)
        return 2

    ui_files = sorted(ui_dir.glob("*.ui"))
    if not ui_files:
        print(f"warning: no .ui files found in {ui_dir}", file=sys.stderr)
        return 0

    # pyuic6 embeds the ``.ui`` path it was invoked with into the generated
    # file's header comment. Run from inside the UI directory so only the
    # bare filename is embedded -- that matches how the pre-existing files
    # were generated and keeps the output stable across machines.
    failed: list[Path] = []
    for ui_file in ui_files:
        py_file = ui_file.with_suffix(".py")
        print(f"pyuic6 {ui_file.name} -> {py_file.name}")
        result = subprocess.run(
            ["pyuic6", ui_file.name, "-o", py_file.name],
            cwd=ui_dir,
        )
        if result.returncode != 0:
            failed.append(ui_file)

    if failed:
        print(f"\nerror: pyuic6 failed for {len(failed)} file(s):", file=sys.stderr)
        for ui_file in failed:
            print(f"  {ui_file.relative_to(REPO_ROOT)}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "ui_dir",
        nargs="?",
        default=str(DEFAULT_UI_DIR),
        help=f"directory containing .ui files (default: {DEFAULT_UI_DIR.relative_to(REPO_ROOT)})",
    )
    args = parser.parse_args(argv)
    return regenerate(Path(args.ui_dir).resolve())


if __name__ == "__main__":
    sys.exit(main())
