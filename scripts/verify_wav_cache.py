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
"""

from __future__ import annotations

import argparse
import logging
import struct
import sys
import wave
from pathlib import Path

log = logging.getLogger("verify_wav_cache")


def verify_wav(wav_path: Path) -> tuple[bool, str]:
    """Validate a WAV file's header matches its actual file size."""
    try:
        file_size = wav_path.stat().st_size
    except OSError as exc:
        return False, f"cannot stat: {exc}"
    if file_size == 0:
        return False, "empty file (0 bytes)"
    try:
        with wave.open(str(wav_path), "rb") as wf:
            n_frames = wf.getnframes()
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            frame_rate = wf.getframerate()
    except (wave.Error, EOFError, struct.error) as exc:
        return False, f"invalid WAV header: {exc}"

    expected_data = n_frames * n_channels * sample_width
    if file_size - expected_data < 0:
        return False, (
            f"TRUNCATED: header declares {n_frames} frames "
            f"({expected_data:,} data bytes) but file is only "
            f"{file_size:,} bytes ({expected_data - file_size:,} bytes short)"
        )

    duration_s = n_frames / frame_rate if frame_rate > 0 else 0
    return True, (
        f"ok: {n_frames:,} frames, {frame_rate} Hz, "
        f"{duration_s:.0f}s ({duration_s / 60:.1f} min), "
        f"{file_size:,} bytes"
    )


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

    wav_files = sorted(args.cache_root.rglob("*.wav"))
    if not wav_files:
        log.info("no WAV files found in %s", args.cache_root)
        return

    valid_count = 0
    corrupt_count = 0
    corrupt_paths = []

    for wav in wav_files:
        ok, reason = verify_wav(wav)
        try:
            rel = wav.relative_to(args.cache_root)
        except ValueError:
            rel = wav

        if ok:
            valid_count += 1
            if args.verbose:
                log.info("  VALID  %s — %s", rel, reason)
        else:
            corrupt_count += 1
            corrupt_paths.append(wav)
            log.warning("  CORRUPT  %s — %s", rel, reason)

    # Summary.
    log.info("")
    log.info("=" * 60)
    log.info("  WAV Cache Verification")
    log.info("  Root:    %s", args.cache_root)
    log.info("  Scanned: %d files", len(wav_files))
    log.info("  Valid:   %d", valid_count)
    log.info("  Corrupt: %d", corrupt_count)
    log.info("=" * 60)

    if corrupt_paths and args.delete:
        log.info("")
        for p in corrupt_paths:
            log.info("  Deleting: %s", p.name)
            p.unlink()
            # Clean up empty parent dirs.
            try:
                p.parent.rmdir()
            except OSError:
                pass
        log.info("  Deleted %d corrupt file(s).", len(corrupt_paths))
    elif corrupt_paths:
        log.info("")
        log.info("  Run with --delete to remove corrupt files.")

    sys.exit(1 if corrupt_count > 0 else 0)


if __name__ == "__main__":
    main()
