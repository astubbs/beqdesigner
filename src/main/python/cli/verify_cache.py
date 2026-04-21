#!/usr/bin/env python3
"""Standalone WAV cache integrity checker.

Scans a WAV cache directory, validates every .wav file, and optionally
deletes corrupt files so they get re-extracted on the next run.

Usage:
    # Check only (report corrupt files):
    python3 verify_wav_cache.py ~/Downloads/beqdesigner/audio-cache

    # Check + delete corrupt files:
    python3 verify_wav_cache.py ~/Downloads/beqdesigner/audio-cache --delete

    # Verbose (show valid files too):
    python3 verify_wav_cache.py ~/Downloads/beqdesigner/audio-cache -v

Requires PYTHONPATH to include src/main/python (uses model.wav_integrity).
Run via: bin/beq-designer verify ...
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running from repo root without setting PYTHONPATH.
_REPO_ROOT = Path(__file__).resolve().parents[4]  # cli/ -> python/ -> main/ -> src/ -> repo
_SRC = _REPO_ROOT / "src" / "main" / "python"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from model.wav_integrity import validate_wav_header, verify_cache

log = logging.getLogger("verify_wav_cache")


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Verify WAV cache integrity — find and optionally delete corrupt files.",
    )
    parser.add_argument(
        "cache_root", type=Path,
        help="Root directory of the WAV cache to scan.",
    )
    parser.add_argument(
        "--delete", action="store_true",
        help="Delete corrupt WAV files (so they get re-extracted).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show valid files too, not just corrupt ones.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    if not args.cache_root.exists():
        log.error("cache root does not exist: %s", args.cache_root)
        sys.exit(1)

    valid, corrupt = verify_cache(args.cache_root)

    if args.verbose:
        for p in valid:
            try:
                rel = p.relative_to(args.cache_root)
            except ValueError:
                rel = p
            ok, reason = validate_wav_header(p)
            log.info("  VALID  %s — %s", rel, reason)

    # Summary.
    log.info("")
    log.info("=" * 60)
    log.info("  WAV Cache Verification")
    log.info("  Root:    %s", args.cache_root)
    log.info("  Scanned: %d files", len(valid) + len(corrupt))
    log.info("  Valid:   %d", len(valid))
    log.info("  Corrupt: %d", len(corrupt))
    log.info("=" * 60)

    if corrupt and args.delete:
        log.info("")
        for p in corrupt:
            log.info("  Deleting: %s", p.name)
            p.unlink()
            try:
                p.parent.rmdir()
            except OSError:
                pass
        log.info("  Deleted %d corrupt file(s).", len(corrupt))
    elif corrupt:
        log.info("")
        log.info("  Run with --delete to remove corrupt files.")

    sys.exit(1 if corrupt else 0)


if __name__ == "__main__":
    main()
