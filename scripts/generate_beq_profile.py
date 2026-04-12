#!/usr/bin/env python3
"""Generate complete BEQ profiles for uncatalogued media.

End-to-end pipeline: extract LFE → NN inference → biquad coefficients →
spectrograph images → TMDb metadata → catalogue-compatible JSON profile.

Usage:
    # Single episode:
    poetry run python3 scripts/generate_beq_profile.py \
      --media "/path/to/Show S01E01.mkv" \
      --output profile.json

    # Batch (directory of episodes):
    poetry run python3 scripts/generate_beq_profile.py \
      --media-dir "/path/to/Show (2020)/Season 01/" \
      --output-dir profiles/

    # With author style:
    poetry run python3 scripts/generate_beq_profile.py \
      --media "/path/to/movie.mkv" \
      --author aron7awol \
      --output profile.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

# Allow running from repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (_REPO_ROOT / "src" / "main" / "python", _REPO_ROOT / "src" / "test" / "python"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np

from model.auto_beq import DEFAULT_GRID, evaluate_filter_chain, smooth_fractional_octave
from model.auto_beq_advisor import MediaMetadata, extract_curve_features
from model.auto_beq_catalogue import _fetch_or_cache
from model.auto_beq_metadata import enrich_media_metadata, fetch_metadata_batch, load_cache
from model.auto_beq_nn import (
    build_feature_vector,
    catalogue_entry_to_labels,
    deduplicate_by_title,
    labels_to_filters,
    load_model,
    train_late_fusion,
)
from model.iir import HighShelf, LowShelf, PeakingEQ
from spike._auto_beq_helpers import beq_dir
from spike.sweep_discover import parse_media_filename

log = logging.getLogger("generate_beq_profile")

_BIQUAD_FS = 96000  # Sample rate for biquad coefficients (catalogue standard)


# ---------------------------------------------------------------------------
# ffprobe JSON parsing
# ---------------------------------------------------------------------------


def _parse_ffprobe_streams(probe_json: str) -> dict:
    """Extract the first audio stream dict from an ffprobe JSON output.

    ffprobe with ``-select_streams a:0`` populates either the top-level
    ``streams`` key (standalone mkv/mp4 files) or the nested
    ``programs[0].streams`` key (container formats like MPEG-TS).  Both
    may also be present but with one of them empty.

    Returns ``{}`` when no audio stream is found or when the JSON is
    malformed.
    """
    import json as _json
    try:
        parsed = _json.loads(probe_json)
    except (ValueError, TypeError):
        return {}
    streams = parsed.get("streams") or []
    if not streams:
        programs = parsed.get("programs") or []
        if programs:
            streams = programs[0].get("streams") or []
    return streams[0] if streams else {}


# ---------------------------------------------------------------------------
# Biquad computation
# ---------------------------------------------------------------------------


def _compute_biquads(filter_dict: dict) -> dict:
    """Compute biquad coefficients for a filter at 96000 Hz.

    Returns the filter dict augmented with a 'biquads' field matching
    the catalogue schema.
    """
    ftype = filter_dict["type"]
    freq = filter_dict["freq"]
    gain = filter_dict["gain"]
    q = filter_dict["q"]

    if ftype == "LowShelf":
        biquad = LowShelf(_BIQUAD_FS, freq, q, gain)
    elif ftype == "HighShelf":
        biquad = HighShelf(_BIQUAD_FS, freq, q, gain)
    elif ftype == "PeakingEQ":
        biquad = PeakingEQ(_BIQUAD_FS, freq, q, gain)
    else:
        return filter_dict

    result = dict(filter_dict)
    result["biquads"] = {
        str(_BIQUAD_FS): {
            "b": [str(x) for x in biquad.b],
            "a": [str(x) for x in biquad.a[1:]],  # catalogue omits a[0] (always 1.0)
        }
    }
    return result


# ---------------------------------------------------------------------------
# Spectrograph generation
# ---------------------------------------------------------------------------


def _generate_spectrographs(
    measured_curve: np.ndarray,
    freqs: np.ndarray,
    filters: list[dict],
    output_dir: Path,
    title: str,
) -> list[str]:
    """Generate before/after spectrograph images.

    Returns list of image file paths.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    filter_response = evaluate_filter_chain(filters, freqs, fs=1000)
    corrected = measured_curve + filter_response

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"BEQ Profile: {title}", fontsize=14, fontweight="bold")

    # Before (measured rolloff).
    ax1.semilogx(freqs, measured_curve, "b-", linewidth=1.5)
    ax1.set_title("Before BEQ (measured LFE)")
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("dB (relative to 80 Hz)")
    ax1.set_xlim(5, 200)
    ax1.set_ylim(-30, 10)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    # After (with predicted BEQ applied).
    ax2.semilogx(freqs, measured_curve, "b-", linewidth=0.8, alpha=0.4, label="Before")
    ax2.semilogx(freqs, corrected, "r-", linewidth=1.5, label="After BEQ")
    ax2.semilogx(freqs, filter_response, "g--", linewidth=1, alpha=0.6, label="Filter response")
    ax2.set_title("After BEQ (predicted filters applied)")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("dB (relative to 80 Hz)")
    ax2.set_xlim(5, 200)
    ax2.set_ylim(-30, 10)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax2.legend(loc="lower right", fontsize=8)

    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_title = "".join(c if c.isalnum() or c in " -_()" else "_" for c in title)
    img_path = output_dir / f"{safe_title}_spectrograph.png"
    fig.savefig(img_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    log.info("spectrograph saved: %s", img_path)
    return [str(img_path)]


# ---------------------------------------------------------------------------
# Model loading / fallback training
# ---------------------------------------------------------------------------


def _load_or_train_model() -> "tuple[object, str]":
    """Load the production BEQ model, or train a legacy fallback inline.

    Resolution order:
      1. ``AUTO_BEQ_ADVISOR=torch_differentiable`` + ``AUTO_BEQ_TORCH_MODEL_PATH``
         or ``{beq-dir}/e85_torch_filter.pt`` — E85 diff-DSP champion (1.49 dB)
      2. ``AUTO_BEQ_MODEL_PATH`` env var (explicit XGBoost override)
      3. ``{beq-dir}/production_model.joblib`` (E82 auto-discover)
      4. Inline late-fusion α=0.3 training (legacy fallback + warning)

    Returns ``(model, source)`` where ``source`` is ``"E85-diff-dsp"``,
    ``"production"``, or ``"inline-late-fusion"`` for provenance.
    """
    # E85 torch path — the new champion.
    advisor_name = os.environ.get("AUTO_BEQ_ADVISOR", "").lower()
    if advisor_name == "torch_differentiable":
        torch_path = os.environ.get("AUTO_BEQ_TORCH_MODEL_PATH")
        if not torch_path:
            try:
                torch_path = str(beq_dir() / "e85_torch_filter.pt")
            except Exception:
                torch_path = None
        if torch_path and Path(torch_path).exists():
            from model.auto_beq_torch import load_torch_predictor
            log.info("loading E85 torch model: %s", torch_path)
            return load_torch_predictor(torch_path), "E85-diff-dsp"
        raise FileNotFoundError(
            f"AUTO_BEQ_ADVISOR=torch_differentiable but no model found "
            f"at {torch_path}. Run `scripts/train_torch_model.py` first.",
        )

    # E82 XGBoost path.
    override = os.environ.get("AUTO_BEQ_MODEL_PATH")
    if override:
        prod_path = Path(override)
        if not prod_path.exists():
            raise FileNotFoundError(
                f"AUTO_BEQ_MODEL_PATH={override} does not exist",
            )
        log.info("loading production model (via AUTO_BEQ_MODEL_PATH): %s", prod_path)
        return load_model(str(prod_path)), "production"

    prod_path = beq_dir() / "production_model.joblib"
    if prod_path.exists():
        log.info("loading production model: %s", prod_path)
        return load_model(str(prod_path)), "production"

    log.warning(
        "production model not found at %s — falling back to inline training. "
        "Run `poetry run python3 scripts/train_production_model.py` to train "
        "and save the production model (E82 50:1 weighted hybrid, 1.99 dB).",
        prod_path,
    )
    log.info("training inline fallback (late fusion α=0.3)...")
    catalogue = _fetch_or_cache()
    deduped = [e for e in deduplicate_by_title(catalogue) if e.get("filters")]
    tmdb_cache = fetch_metadata_batch(deduped, cache=load_cache())

    def _synthetic_features(entry, freqs):
        correction = evaluate_filter_chain(entry["filters"], freqs, fs=1000)
        rolloff = -correction
        a = int(np.argmin(np.abs(freqs - 80.0)))
        rolloff -= rolloff[a]
        return extract_curve_features(rolloff, freqs)

    X_train, Y_train = [], []
    for e in deduped:
        f = _synthetic_features(e, DEFAULT_GRID)
        m = enrich_media_metadata(e, tmdb_cache)
        X_train.append(build_feature_vector(f, m))
        Y_train.append(catalogue_entry_to_labels(e))
    X_train = np.array(X_train, dtype=np.float32)
    Y_train = np.array(Y_train, dtype=np.float32)

    return train_late_fusion(X_train, Y_train, alpha=0.3), "inline-late-fusion"


# ---------------------------------------------------------------------------
# Profile assembly
# ---------------------------------------------------------------------------


def _compute_digest(filters: list[dict]) -> str:
    """Compute SHA256 digest of the filter chain (catalogue convention)."""
    canonical = json.dumps(filters, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _compute_mv_adjust(filters: list[dict], freqs: np.ndarray) -> float:
    """Estimate master volume adjustment (peak gain of the filter chain)."""
    response = evaluate_filter_chain(filters, freqs, fs=1000)
    return round(float(response.max()), 1)


def generate_profile(
    media_path: Path,
    model: object,
    author: str = "auto",
    output_path: Path | None = None,
    output_dir: Path | None = None,
    model_source: str = "production",
) -> dict:
    """Generate a complete BEQ profile for a media file.

    ``model`` must be a pre-loaded NN model compatible with ``.predict(X)``
    returning a ``(1, N_OUTPUT)`` label vector — typically the saved
    production model from ``_load_or_train_model()``.

    Returns the profile dict (also saved to output_path if specified).
    """
    log.info("generating profile for: %s", media_path.name)

    # Parse title/year/episode from filename.
    parsed = parse_media_filename(media_path)
    if parsed is None:
        raise ValueError(f"cannot parse title/year from: {media_path.name}")

    title = parsed.title
    year = parsed.year
    season = parsed.season
    episode = parsed.episode

    log.info("  parsed: %s (%d) S%sE%s", title, year,
             f"{season:02d}" if season else "?",
             f"{episode:02d}" if episode else "?")

    # Extract LFE to a temp directory (avoids long path issues with the
    # legacy path-mirrored cache when filenames have many codec tags).
    import tempfile
    tmp_dir = Path(tempfile.mkdtemp(prefix="beq_lfe_"))
    safe_stem = "".join(c if c.isalnum() or c in " -_()" else "_" for c in media_path.stem)[:80]
    tmp_wav = tmp_dir / f"{safe_stem}.lfe-1000hz.wav"

    log.info("  extracting LFE...")
    import subprocess

    # Handle Blu-ray ISOs: use bluray: protocol for ffprobe and ffmpeg.
    is_iso = media_path.suffix.lower() == ".iso"
    ffmpeg_input = f"bluray:{media_path}" if is_iso else str(media_path)
    probe_input = ffmpeg_input

    probe_cmd = [
        "ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=codec_name,channels,channel_layout,sample_rate",
        "-of", "json", probe_input,
    ]
    probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
    if probe_result.returncode == 0:
        stream = _parse_ffprobe_streams(probe_result.stdout)
    else:
        stream = {}

    layout = stream.get("channel_layout", "")
    has_lfe = "LFE" in layout.upper() or any(
        layout.lower().startswith(p) for p in ("5.1", "6.1", "7.1")
    )
    af_filter = "pan=mono|c0=LFE" if has_lfe else "aresample"
    if not has_lfe:
        log.info("  no LFE channel — falling back to mono downmix")
    else:
        log.info("  LFE channel detected: %s", layout)

    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", ffmpeg_input,
        "-af", af_filter, "-ac", "1", "-ar", "1000",
        "-sample_fmt", "s16", "-f", "wav",
        str(tmp_wav),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        log.error("ffmpeg failed: %s", result.stderr.strip()[:200])
        raise RuntimeError(f"ffmpeg failed (exit {result.returncode})")

    wav_path = tmp_wav
    log.info("  WAV: %s (%d bytes)", wav_path.name, wav_path.stat().st_size)

    # Measure spectrum.
    from model.signal import Signal, read_wav_data
    samples, read_fs, _ = read_wav_data(str(wav_path))
    mono = samples[:, 0] if samples.ndim > 1 else samples
    sig = Signal(wav_path.stem, mono, fs=1000)
    measured_freqs, measured_db = sig.avg_spectrum()

    curve = np.interp(DEFAULT_GRID, measured_freqs, measured_db)
    anchor_idx = int(np.argmin(np.abs(DEFAULT_GRID - 80.0)))
    curve -= curve[anchor_idx]
    curve = smooth_fractional_octave(curve, DEFAULT_GRID, octaves=1.0 / 6.0)
    curve -= curve[anchor_idx]

    features = extract_curve_features(curve, DEFAULT_GRID)

    # Probe audio metadata.
    from spike._auto_beq_helpers import probe_audio_stream
    try:
        stream = probe_audio_stream(media_path)
        audio_codec = stream.get("codec_name", "unknown")
        channels = stream.get("channels", 2)
        layout = stream.get("channel_layout", "stereo")
    except Exception:
        audio_codec = "unknown"
        channels = 2
        layout = "stereo"

    # Build metadata.
    metadata = MediaMetadata(
        title=title,
        year=year,
        audio_codec=audio_codec,
        genres=(),
        source="Disc" if "bluray" in str(media_path).lower() else "Streaming",
        author=author if author != "auto" else None,
    )

    # Predict with the pre-loaded model (loaded once in main()).
    log.info("  predicting filters (model: %s)...", model_source)
    x = build_feature_vector(features, metadata)

    # E85 torch predictor has predict_filters(x) → filter dicts directly.
    # XGBoost models have predict(X) → raw label vector → labels_to_filters.
    if hasattr(model, "predict_filters"):
        filters = model.predict_filters(x)
    else:
        y_pred = model.predict(x.reshape(1, -1))[0]
        filters = labels_to_filters(y_pred)

    # Add biquad coefficients.
    filters_with_biquads = [_compute_biquads(f) for f in filters]

    log.info("  predicted %d filters", len(filters))
    for f in filters:
        log.info("    %s(%.0fHz, %+.1fdB, Q=%.1f)", f["type"], f["freq"], f["gain"], f["q"])

    # Generate spectrographs.
    img_dir = (output_dir or output_path.parent if output_path else Path("."))
    ep_label = f"S{season:02d}E{episode:02d}" if season and episode else ""
    author_label = f" [{author}]" if author != "auto" else ""
    full_title = f"{title} ({year}) {ep_label}{author_label}".strip()
    images = _generate_spectrographs(curve, DEFAULT_GRID, filters, img_dir, full_title)

    # Compute MV adjust and digest.
    mv = _compute_mv_adjust(filters, DEFAULT_GRID)
    digest = _compute_digest(filters_with_biquads)

    # Assemble profile.
    profile = {
        "title": title,
        "year": str(year),
        "audioTypes": [f"{audio_codec.upper()} {channels}.{'1' if 'lfe' in layout.lower() else '0'}"],
        "content_type": "TV" if season else "film",
        "author": "auto-beq-nn",
        "filters": filters_with_biquads,
        "images": images,
        "warning": "",
        "mv": str(mv),
        "sortTitle": title.lower(),
        "edition": "",
        "note": f"Auto-generated by NN model ({model_source}, style={author})",
        "language": "Japanese",
        "source": metadata.source or "",
        "overview": "",  # TODO: TMDb lookup
        "rating": "",
        "runtime": "",
        "genres": list(metadata.genres),
        "digest": digest,
        "created_at": int(time.time()),
        "updated_at": int(time.time()),
    }

    if season:
        profile["season"] = str(season)
    if episode:
        profile["episode"] = str(episode)

    # Save.
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(profile, indent=2) + "\n")
        log.info("profile saved: %s", output_path)

    return profile


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate complete BEQ profiles for uncatalogued media.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--media", type=Path, help="Single media file.")
    parser.add_argument("--media-dir", type=Path, help="Directory of media files (batch mode).")
    parser.add_argument("--output", "-o", type=Path, help="Output JSON file (single mode).")
    parser.add_argument("--output-dir", type=Path, help="Output directory (batch mode).")
    parser.add_argument("--author", default="aron7awol", help="Author style (default: aron7awol).")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.media and not args.media_dir:
        parser.error("specify --media or --media-dir")

    # Load the production model once; batch mode reuses it for every file.
    model, model_source = _load_or_train_model()

    if args.media:
        generate_profile(
            args.media,
            model=model,
            author=args.author,
            output_path=args.output or Path(f"{args.media.stem}_beq.json"),
            model_source=model_source,
        )
    else:
        output_dir = args.output_dir or Path("profiles")
        for mkv in sorted(args.media_dir.glob("*.mkv")):
            try:
                generate_profile(
                    mkv,
                    model=model,
                    author=args.author,
                    output_path=output_dir / f"{mkv.stem}_beq.json",
                    output_dir=output_dir,
                    model_source=model_source,
                )
            except Exception as exc:
                log.error("failed: %s — %s", mkv.name, exc)


if __name__ == "__main__":
    main()
