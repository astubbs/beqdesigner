"""Integration tests for WAV file integrity validation.

These protect the WAV database cache from silent corruption caused by
interrupted ffmpeg extractions, partial network transfers, or disk errors.

All tests use synthetic WAV data — no NAS or real media needed.
"""

from __future__ import annotations

import struct
import wave
from pathlib import Path

import pytest
from model.wav_integrity import (
    validate_wav,
    validate_wav_duration,
    validate_wav_header,
    verify_cache,
)


def _make_wav(path: Path, n_frames: int, fs: int = 1000, channels: int = 1, sample_width: int = 2):
    """Create a valid WAV file with the given number of frames."""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(fs)
        # Write silence (zero samples).
        wf.writeframes(b"\x00" * (n_frames * channels * sample_width))


# ---------------------------------------------------------------------------
# 1. Truncated WAV detected by sample count
# ---------------------------------------------------------------------------


def test_truncated_wav_detected_by_sample_count(tmp_path):
    """A WAV file whose actual data is shorter than the header declares."""
    wav_path = tmp_path / "truncated.wav"

    # Create a valid 60-second WAV (60,000 frames at 1000 Hz).
    _make_wav(wav_path, n_frames=60_000)

    # Verify it's valid first.
    ok, reason = validate_wav_header(wav_path)
    assert ok, f"fresh WAV should be valid: {reason}"

    # Truncate: chop off the last 50% of the file.
    original_size = wav_path.stat().st_size
    with wav_path.open("r+b") as f:
        f.truncate(original_size // 2)

    # Now the header still says 60,000 frames but the file is half the size.
    ok, reason = validate_wav_header(wav_path)
    assert not ok, "truncated WAV should be detected"
    assert "truncated" in reason.lower() or "short" in reason.lower(), (
        f"reason should mention truncation: {reason}"
    )


# ---------------------------------------------------------------------------
# 2. Truncated WAV detected by duration
# ---------------------------------------------------------------------------


def test_truncated_wav_detected_by_duration(tmp_path):
    """A WAV that's valid structurally but way too short for the movie runtime."""
    wav_path = tmp_path / "too_short.wav"

    # Create a 10-second WAV (10,000 frames at 1000 Hz).
    _make_wav(wav_path, n_frames=10_000)

    # It should be structurally valid.
    ok, reason = validate_wav_header(wav_path)
    assert ok, f"structurally valid WAV: {reason}"

    # But for a 120-minute movie, 10 seconds is way too short.
    ok, reason = validate_wav_duration(wav_path, expected_runtime_min=120)
    assert not ok, "10s WAV should fail 120-min duration check"
    assert "too short" in reason.lower(), f"reason should mention duration: {reason}"

    # Combined check should also fail.
    ok, reason = validate_wav(wav_path, expected_runtime_min=120)
    assert not ok


def test_duration_check_passes_for_matching_wav(tmp_path):
    """A WAV whose duration matches the expected runtime passes."""
    wav_path = tmp_path / "full_length.wav"

    # 90-minute movie → 5,400,000 frames at 1000 Hz.
    _make_wav(wav_path, n_frames=5_400_000)

    ok, reason = validate_wav_duration(wav_path, expected_runtime_min=90)
    assert ok, f"matching duration should pass: {reason}"


def test_duration_check_skipped_when_no_runtime(tmp_path):
    """Duration check is skipped when expected runtime is 0 or not provided."""
    wav_path = tmp_path / "short.wav"
    _make_wav(wav_path, n_frames=100)

    ok, reason = validate_wav_duration(wav_path, expected_runtime_min=0)
    assert ok, f"should skip check with runtime=0: {reason}"


# ---------------------------------------------------------------------------
# 3. Interrupted ffmpeg produces no final WAV (atomic write)
# ---------------------------------------------------------------------------


def test_atomic_write_prevents_corrupt_cache(tmp_path):
    """Simulates the atomic write pattern: .tmp exists but final .wav doesn't.

    The extraction script writes to .tmp first, then renames on success.
    If interrupted, only .tmp exists — the cache never sees a corrupt .wav.
    """
    final_path = tmp_path / "Movies" / "A" / "Alien (1979) [tmdb-348]" / "lfe-1000hz.wav"
    tmp_path_wav = final_path.with_suffix(".tmp")

    # Simulate interrupted extraction: .tmp exists with partial data.
    tmp_path_wav.parent.mkdir(parents=True, exist_ok=True)
    tmp_path_wav.write_bytes(b"RIFF" + b"\x00" * 100)  # partial WAV header

    assert tmp_path_wav.exists()
    assert not final_path.exists(), "final .wav should NOT exist after interrupt"

    # Cleanup: the extraction script should delete .tmp on next run.
    # Simulate that.
    for orphan in tmp_path.rglob("*.tmp"):
        orphan.unlink()

    assert not tmp_path_wav.exists(), ".tmp should be cleaned up"


# ---------------------------------------------------------------------------
# 4. Valid WAV passes all checks
# ---------------------------------------------------------------------------


def test_valid_wav_passes_all_checks(tmp_path):
    """A properly extracted WAV passes both header and duration checks."""
    wav_path = tmp_path / "valid.wav"

    # 110-minute movie at 1000 Hz.
    _make_wav(wav_path, n_frames=110 * 60 * 1000)

    ok, reason = validate_wav(wav_path, expected_runtime_min=110)
    assert ok, f"valid WAV should pass: {reason}"
    assert "all checks passed" in reason


# ---------------------------------------------------------------------------
# 5. verify_cache finds corrupt files
# ---------------------------------------------------------------------------


def test_verify_cache_finds_corrupt(tmp_path):
    """verify_cache correctly separates valid and corrupt WAVs."""
    # Create 3 valid WAVs.
    for name in ("a.wav", "b.wav", "c.wav"):
        _make_wav(tmp_path / name, n_frames=60_000)

    # Truncate one.
    corrupt = tmp_path / "b.wav"
    original_size = corrupt.stat().st_size
    with corrupt.open("r+b") as f:
        f.truncate(original_size // 4)

    valid, bad = verify_cache(tmp_path)
    assert len(valid) == 2
    assert len(bad) == 1
    assert bad[0] == corrupt


# ---------------------------------------------------------------------------
# 6. Empty file detected
# ---------------------------------------------------------------------------


def test_empty_file_detected(tmp_path):
    """A zero-byte file is detected as corrupt."""
    wav_path = tmp_path / "empty.wav"
    wav_path.write_bytes(b"")

    ok, reason = validate_wav_header(wav_path)
    assert not ok
    assert "empty" in reason.lower()


# ---------------------------------------------------------------------------
# 7. Garbage file detected
# ---------------------------------------------------------------------------


def test_garbage_file_detected(tmp_path):
    """A file with random bytes (not a WAV) is detected."""
    wav_path = tmp_path / "garbage.wav"
    wav_path.write_bytes(b"this is not a wav file at all" * 100)

    ok, reason = validate_wav_header(wav_path)
    assert not ok
    assert "invalid" in reason.lower() or "error" in reason.lower()


# ---------------------------------------------------------------------------
# 8. Media DB ID extraction from paths
# ---------------------------------------------------------------------------


class TestMediaIdExtraction:
    """Test extract_media_id() from scripts/extract_lfe.py."""

    def _extract(self, path_str: str) -> tuple[str, str] | None:
        """Import and call extract_media_id with a synthetic Path."""
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
        from extract_lfe import extract_media_id
        return extract_media_id(Path(path_str))

    def test_tmdb_in_filename(self):
        result = self._extract(
            "/media/Movies/Blade Runner (1982) [tmdb-78]/Blade Runner (1982) [tmdb-78].mkv"
        )
        assert result == ("tmdb", "78")

    def test_tvdb_in_parent_dir(self):
        """TV show with [tvdb-NNN] in parent dir, not in filename."""
        result = self._extract(
            "/media/Anime TV/86 - Eighty Six (2021) [tvdb-378609]/Season 01/"
            "86 - Eighty Six (2021) - S01E02 - 002 - Spearhead [Bluray-2160p].mkv"
        )
        assert result is not None
        assert result == ("tvdb", "378609")

    def test_tvdb_in_grandparent_dir(self):
        """TV show with [tvdb-NNN] in grandparent (title dir above Season dir)."""
        result = self._extract(
            "/media/TV/Blue Eye Samurai (2023) [tvdb-434151]/Season 1/"
            "Blue Eye Samurai (2023) - S01E01 - Hammerscale.mkv"
        )
        assert result is not None
        assert result == ("tvdb", "434151")

    def test_imdb_id(self):
        result = self._extract(
            "/media/Movies/Some Movie (2020) [imdb-tt1234567]/Some Movie.mkv"
        )
        assert result == ("imdb", "tt1234567")

    def test_filename_takes_priority_over_dir(self):
        """If both filename and dir have IDs, filename wins."""
        result = self._extract(
            "/media/Show (2021) [tvdb-111]/Season 01/Show S01E01 [tmdb-222].mkv"
        )
        assert result == ("tmdb", "222")

    def test_no_id_returns_none(self):
        result = self._extract(
            "/media/Movies/Some Movie (2020)/Some Movie.mkv"
        )
        assert result is None
