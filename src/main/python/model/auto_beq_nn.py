"""Trained-model Advisor for auto-BEQ — Experiment 18.

A regression model trained on BEQ catalogue entries learns the mapping from:
  audio features (Option A: 9-bin 90th-percentile curve, 20–80 Hz)
  + metadata (year, audio format, source, studio, mixer, genre, country,
             runtime, rating)
to corrective filter parameters (up to 4 biquad filters, padded to 16 dims).

This is Stage 1 of the architecture progression (XGBoost). The same feature
vector feeds Stage 2 (1D CNN) and Stage 3 (transformer) without changes;
only the model object changes.

See docs/design/auto_beq_ml_experiments.md for full design rationale.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from model.auto_beq_advisor import Advice, CurveFeatures, MediaMetadata

log = logging.getLogger("auto_beq_nn")

# ---------------------------------------------------------------------------
# Audio feature constants — Option A (9-bin percentile curve, plan spec)
# ---------------------------------------------------------------------------

OPTION_A_BINS_HZ = [20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 60.0, 70.0, 80.0]
N_AUDIO_FEATURES = len(OPTION_A_BINS_HZ)  # 9

# ---------------------------------------------------------------------------
# Metadata encoding constants
# ---------------------------------------------------------------------------

# Audio format buckets (Tier 1 — different encode chains → different headroom)
_AUDIO_FORMAT_BUCKETS = ["atmos", "truehd", "dts-hd ma", "dd+ atmos", "dd+", "other"]
N_AUDIO_FORMAT = len(_AUDIO_FORMAT_BUCKETS)  # 6

# Source / release type (Tier 1 — UHD remasters often re-filtered vs BD)
_SOURCE_BUCKETS = ["disc", "streaming", "unknown"]
N_SOURCE = len(_SOURCE_BUCKETS)  # 3

# Studio parent-company groups — subsidiaries share mixing stages.
# A film mixed at Disney's Buena Vista stages has the same bass rolloff
# tendencies whether it's branded Pixar or Marvel. Grouping gives 35%
# coverage in 11 dims vs 20% in 30 dims with individual studios.
_STUDIO_PARENT_GROUPS: dict[str, list[str]] = {
    "disney": [
        "walt disney pictures", "pixar", "marvel studios", "touchstone pictures",
        "lucasfilm", "hollywood pictures", "walt disney animation studios",
        "20th century fox", "20th century studios", "searchlight pictures",
        "dreamworks animation", "blue sky studios",
    ],
    "warner": [
        "warner bros. pictures", "warner bros. animation", "new line cinema",
        "castle rock entertainment", "dc films", "dc studios", "hbo films",
        "warner animation group",
    ],
    "universal": [
        "universal pictures", "focus features", "working title films",
        "illumination", "amblin entertainment", "gramercy pictures",
        "universal 1440 entertainment", "dreamworks pictures",
    ],
    "sony/columbia": [
        "columbia pictures", "tristar pictures", "screen gems",
        "sony pictures", "revolution studios", "sony pictures animation",
    ],
    "paramount": [
        "paramount pictures", "paramount animation", "republic pictures",
        "paramount players", "miramax",
    ],
    "lionsgate": [
        "lionsgate", "lionsgate films", "summit entertainment",
        "artisan entertainment", "studiocanal",
    ],
    "mgm": [
        "metro-goldwyn-mayer", "united artists", "orion pictures",
    ],
    "netflix": ["netflix"],
    "amazon": ["amazon studios", "amazon mgm studios"],
    "a24": ["a24"],
    "blumhouse": ["blumhouse productions"],
}


def _resolve_studio_parent(studio: str | None, all_studios: tuple[str, ...] = ()) -> str:
    """Resolve a studio name to its parent company.

    Checks primary studio first, then all production companies from TMDb.
    Returns the parent group name (lowercase) or the original studio name
    (lowercase) if no parent match.
    """
    candidates = []
    if studio:
        candidates.append(studio.lower().strip())
    candidates.extend(s.lower().strip() for s in all_studios)

    for candidate in candidates:
        for parent, subsidiaries in _STUDIO_PARENT_GROUPS.items():
            if candidate in subsidiaries or any(sub in candidate for sub in subsidiaries):
                return parent
    # No parent match — return primary studio name for individual vocab lookup.
    return (studio or "").lower().strip()


# Hybrid studio vocab: 11 parent groups + top ungrouped studios + "other".
# Parent groups carry the mixing-stage signal (35% coverage); individual
# ungrouped studios add indie/international coverage (~10% more).
_STUDIO_VOCAB = [
    # Parent groups (11)
    "disney", "warner", "universal", "sony/columbia", "paramount",
    "lionsgate", "mgm", "netflix", "amazon", "a24", "blumhouse",
    # Top ungrouped individual studios (≥15 entries in catalogue)
    "dimension films", "yash raj films", "cj entertainment", "studio dragon",
    "t-series", "media asia films", "legendary pictures", "jce movies",
    "xyz films", "europacorp", "anonymous content", "next entertainment world",
    "village roadshow pictures", "eon productions", "canal+",
    "relativity media", "regency enterprises", "imagine entertainment",
    "skydance media", "plan b entertainment",
    # Catch-all
    "other",
]
N_STUDIO = len(_STUDIO_VOCAB)  # 32

# Mixer vocabulary — top-20 sound re-recording mixers + "other" + "unknown".
# Covers ~17% of entries with known mixers. Sparse but each name is a
# meaningful signal (individual mixers have consistent rolloff styles).
_MIXER_VOCAB = [
    "andy nelson", "kevin o'connell", "mike prestwood smith",
    "paul massey", "ron bartlett", "anna behlmer", "michael minkler",
    "lora hirschberg", "michael semanick", "tom fleischman",
    "mark paterson", "chris burdon", "tom johnson", "christopher boyes",
    "doug hemphill", "gary rizzo", "skip lievsay", "scott millan",
    "frank a. montaño", "gary summers",
    "other", "unknown",
]
N_MIXER = len(_MIXER_VOCAB)  # 22

# Genre multi-hot (Tier 2)
_GENRE_BUCKETS = [
    "action", "adventure", "science fiction", "thriller", "drama",
    "horror", "comedy", "animation", "documentary", "other",
]
N_GENRE = len(_GENRE_BUCKETS)  # 10

# Country origin one-hot derived from language (Tier 2)
_COUNTRY_BUCKETS = ["english", "korean", "japanese", "european", "other"]
N_COUNTRY = len(_COUNTRY_BUCKETS)  # 5

# Rating label encoding (Tier 3)
_RATING_MAP = {"G": 0.0, "PG": 1.0, "PG-13": 2.0, "R": 3.0}

# BEQ profile author — different authors have different calibration styles.
# 100% coverage (every catalogue entry has an author), only 8 unique.
_AUTHOR_VOCAB = [
    "mobe1969", "aron7awol", "mikejl", "kaelaria", "remixmark",
    "t1g8rsfan", "halcyon888", "bombaycat007", "unknown",
]
N_AUTHOR = len(_AUTHOR_VOCAB)  # 9

# Era buckets (E39b) — gives XGBoost explicit split points for decade patterns.
# 1980s content is 5.1 dB mean vs 2.4 dB for 2020s.
_ERA_BUCKETS = ["pre_1990", "1990_2009", "2010_plus"]
N_ERA = len(_ERA_BUCKETS)  # 3

# Metadata vector size
N_METADATA_FEATURES = (
    1                # year normalised
    + N_AUDIO_FORMAT  # 6
    + N_SOURCE        # 3
    + N_STUDIO        # 32
    + N_MIXER         # 22
    + N_GENRE         # 10
    + N_COUNTRY       # 5
    + 1               # runtime normalised
    + 1               # rating normalised
    + N_AUTHOR        # 9
    + N_ERA           # 3
)  # = 93

N_FEATURES = N_AUDIO_FEATURES + N_METADATA_FEATURES  # 102


# ---------------------------------------------------------------------------
# Audio feature configuration (F-experiment support)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AugmentationConfig:
    """Training-time noise augmentation for synthetic audio features (F1).

    Adds controlled noise to audio feature dims during training to simulate
    the distribution mismatch between synthetic "perfect inverse" curves and
    real measured spectra.  Only applied during training, never at inference.

    In ML literature this is *input feature augmentation* — analogous to
    image augmentation (random crop, colour jitter) but applied to 1-D
    spectral features.  The rationale is that synthetic training features are
    deterministic inversions of catalogue filter chains, while real measured
    spectra contain content-dependent noise, Welch averaging artefacts, and
    chunk-level variation.  By injecting noise whose magnitude matches the
    observed synthetic-to-real gap (~1-3 dB per bin from E25-E34), we teach
    the model to be robust to these real-world perturbations.
    """

    gaussian_sigma_db: float = 1.5    # σ of additive N(0, σ) noise per bin
    per_bin_uniform_db: float = 2.0   # half-width of U(-u, +u) per-bin jitter
    smooth_prob: float = 0.3          # probability of 3-point running average
    n_copies: int = 3                 # augmented copies per original sample
    seed: int = 42

    @property
    def label(self) -> str:
        return (
            f"aug-s{self.gaussian_sigma_db:.1f}"
            f"-u{self.per_bin_uniform_db:.0f}"
            f"-n{self.n_copies}"
        )


@dataclass(frozen=True)
class AudioFeatureConfig:
    """Describes which audio features are active and their dimensionality.

    Defaults reproduce the original 9-dim Option A behaviour so all existing
    code paths work unchanged when constructed with no arguments.

    Flag fields enable F-experiment extensions:
      * ``use_option_b``      — F2: adds 18 chunk-statistics dims (stddev +
                                ceiling fraction per bin)
      * ``use_absolute_dbfs`` — F3: adds 9 absolute dBFS level dims
      * ``use_high_res``      — F11: 16-bin high-resolution frequency grid
      * ``use_rolloff_cluster`` — F7: k-means cluster ID as one-hot feature
    """

    n_audio: int = N_AUDIO_FEATURES   # base Option A bins (9 or 16)
    use_option_b: bool = False        # F2: +18 dims
    use_absolute_dbfs: bool = False   # F3: +9 dims
    use_high_res: bool = False        # F11: 16 bins instead of 9
    use_rolloff_cluster: bool = False # F7: cluster ID one-hot
    n_clusters: int = 6              # F7: number of clusters

    @property
    def n_total_audio(self) -> int:
        n = 16 if self.use_high_res else self.n_audio
        if self.use_option_b:
            n += 18  # 9 stddev + 9 ceiling_fraction (always at 9 bins)
        if self.use_absolute_dbfs:
            n += 9   # absolute dBFS at 9 bins
        if self.use_rolloff_cluster:
            n += self.n_clusters  # one-hot cluster ID
        return n

    @property
    def n_features(self) -> int:
        return self.n_total_audio + N_METADATA_FEATURES

    @property
    def label(self) -> str:
        parts: list[str] = []
        if self.use_high_res:
            parts.append("HR16")
        else:
            parts.append(f"A{self.n_audio}")
        if self.use_option_b:
            parts.append("optB")
        if self.use_absolute_dbfs:
            parts.append("dBFS")
        if self.use_rolloff_cluster:
            parts.append(f"clust{self.n_clusters}")
        return "+".join(parts) if parts else "A9"


DEFAULT_AUDIO_CONFIG = AudioFeatureConfig()


# ---------------------------------------------------------------------------
# Label constants (filter parameter output vector)
# ---------------------------------------------------------------------------

MAX_FILTER_SLOTS = 6  # E39a: was 4→8 (over-predicted), 6 covers 77% of catalogue
N_TYPE = 3  # LowShelf, HighShelf, PeakingEQ — one-hot encoded
# Per slot: [type_LS, type_HS, type_PEQ, freq_hz, gain_db, q] = 6 values
N_PER_SLOT = N_TYPE + 3
N_OUTPUT = MAX_FILTER_SLOTS * N_PER_SLOT  # 8 × 6 = 48

FILTER_TYPES = ["LowShelf", "HighShelf", "PeakingEQ"]
_TYPE_TO_IDX = {t: i for i, t in enumerate(FILTER_TYPES)}
_IDX_TO_TYPE = {i: t for i, t in enumerate(FILTER_TYPES)}

# Downstream evaluation band
_DOWNSTREAM_BAND_HZ = (20.0, 80.0)

# ---------------------------------------------------------------------------
# Audio feature extraction
# ---------------------------------------------------------------------------


def build_audio_features(features: CurveFeatures) -> np.ndarray:
    """Build Option A 9-bin audio feature vector from pre-computed CurveFeatures.

    Samples the 12-point ``curve_sample_points`` at the 9 Option A bins using
    nearest-bin lookup. The curve is already normalised to 0 dB at 80 Hz by
    ``extract_curve_features()``.

    Returns float32 array of shape (9,).
    """
    sample_pts = list(features.curve_sample_points)  # [(hz, db), ...]
    if not sample_pts:
        return np.zeros(N_AUDIO_FEATURES, dtype=np.float32)

    sample_hz = np.array([hz for hz, _ in sample_pts])
    sample_db = np.array([db for _, db in sample_pts])

    out = np.empty(N_AUDIO_FEATURES, dtype=np.float32)
    for i, target_hz in enumerate(OPTION_A_BINS_HZ):
        idx = int(np.argmin(np.abs(sample_hz - target_hz)))
        out[i] = float(sample_db[idx])
    return out


# ---------------------------------------------------------------------------
# Metadata feature encoding
# ---------------------------------------------------------------------------


def _audio_format_onehot(audio_types: tuple[str, ...]) -> np.ndarray:
    """One-hot encode the first recognised audio format from audioTypes."""
    v = np.zeros(N_AUDIO_FORMAT, dtype=np.float32)
    joined = " ".join(t.lower() for t in audio_types)
    for i, bucket in enumerate(_AUDIO_FORMAT_BUCKETS[:-1]):  # skip 'other'
        if bucket in joined:
            v[i] = 1.0
            return v
    v[-1] = 1.0  # 'other'
    return v


def _source_onehot(source: str | None) -> np.ndarray:
    v = np.zeros(N_SOURCE, dtype=np.float32)
    s = (source or "").lower()
    if "disc" in s:
        v[0] = 1.0
    elif "stream" in s:
        v[1] = 1.0
    else:
        v[2] = 1.0
    return v


def _genre_multihot(genres: tuple[str, ...]) -> np.ndarray:
    v = np.zeros(N_GENRE, dtype=np.float32)
    genre_lower = {g.lower() for g in genres}
    matched = False
    for i, bucket in enumerate(_GENRE_BUCKETS[:-1]):  # skip 'other'
        if bucket in genre_lower:
            v[i] = 1.0
            matched = True
    if not matched:
        v[-1] = 1.0
    return v


def _country_onehot(language: str | None) -> np.ndarray:
    v = np.zeros(N_COUNTRY, dtype=np.float32)
    lang = (language or "").lower()
    _european = {"french", "german", "spanish", "italian", "portuguese", "dutch", "swedish", "norwegian", "danish"}
    if "english" in lang:
        v[0] = 1.0
    elif "korean" in lang:
        v[1] = 1.0
    elif "japanese" in lang:
        v[2] = 1.0
    elif any(e in lang for e in _european):
        v[3] = 1.0
    else:
        v[4] = 1.0
    return v


def _vocab_onehot(name: str | None, vocab: list[str]) -> np.ndarray:
    """One-hot encode a string against a fixed vocabulary.

    Looks up ``name`` (lowercased) in ``vocab``. If not found, activates
    the last bucket (expected to be "other" or "unknown"). If ``name`` is
    None, activates the last bucket.
    """
    v = np.zeros(len(vocab), dtype=np.float32)
    if name:
        key = name.lower().strip()
        if key in vocab:
            v[vocab.index(key)] = 1.0
        else:
            v[-1] = 1.0  # "other" bucket
    else:
        v[-1] = 1.0
    return v


def build_metadata_features(metadata: MediaMetadata) -> np.ndarray:
    """Encode MediaMetadata into a 51-dim float32 vector.

    When ``metadata.studio`` or ``metadata.supervising_mixer`` are populated
    (via TMDb lookup), their slots are feature-hashed into their respective
    embedding dimensions. When absent (None), slots are all zeros.
    """
    parts: list[np.ndarray] = []

    # Year (Tier 1, normalised)
    year = float(metadata.year or 2000)
    parts.append(np.array([(year - 2000.0) / 20.0], dtype=np.float32))

    # Audio format one-hot (Tier 1)
    audio_types = getattr(metadata, "audio_types", ())
    parts.append(_audio_format_onehot(audio_types))

    # Source one-hot (Tier 1)
    parts.append(_source_onehot(getattr(metadata, "source", None)))

    # Studio — resolved to parent company, then one-hot against hybrid vocab.
    # Disney subsidiaries (Pixar, Marvel, etc.) share mixing stages → same group.
    studio = getattr(metadata, "studio", None)
    all_studios = getattr(metadata, "all_studios", ())
    resolved_studio = _resolve_studio_parent(studio, all_studios)
    parts.append(_vocab_onehot(resolved_studio, _STUDIO_VOCAB))

    # Mixer — one-hot against top-20 vocabulary (Tier 2).
    # "Sound Re-Recording Mixer" from TMDb credits.
    mixer = getattr(metadata, "supervising_mixer", None)
    parts.append(_vocab_onehot(mixer, _MIXER_VOCAB))

    # Genre multi-hot (Tier 2)
    genres = getattr(metadata, "genres", ())
    parts.append(_genre_multihot(genres))

    # Country of origin one-hot from language (Tier 2)
    parts.append(_country_onehot(getattr(metadata, "language", None)))

    # Runtime normalised (Tier 2)
    runtime = float(getattr(metadata, "runtime_min", None) or 0)
    parts.append(np.array([runtime / 180.0], dtype=np.float32))

    # Rating label-encoded (Tier 3)
    rating = getattr(metadata, "rating", None)
    rating_val = _RATING_MAP.get(rating or "", 2.0) / 4.0  # default PG-13
    parts.append(np.array([rating_val], dtype=np.float32))

    # BEQ profile author — one-hot (E24). 100% coverage, 8 unique authors.
    author = getattr(metadata, "author", None)
    parts.append(_vocab_onehot(author, _AUTHOR_VOCAB))

    # Era bucket — one-hot (E39b). Gives XGBoost explicit split points
    # for decade-specific rolloff patterns.
    year = float(metadata.year or 2000)
    era = np.zeros(N_ERA, dtype=np.float32)
    if year < 1990:
        era[0] = 1.0  # pre_1990
    elif year < 2010:
        era[1] = 1.0  # 1990_2009
    else:
        era[2] = 1.0  # 2010_plus
    parts.append(era)

    result = np.concatenate(parts)
    assert result.shape == (N_METADATA_FEATURES,), (
        f"metadata feature dim mismatch: {result.shape} != ({N_METADATA_FEATURES},)"
    )
    return result


def build_feature_vector(
    features: CurveFeatures,
    metadata: MediaMetadata,
    config: AudioFeatureConfig = DEFAULT_AUDIO_CONFIG,
) -> np.ndarray:
    """Build the full input vector: audio (variable) + metadata (93).

    The audio portion is configurable via *config*:
      - Default (9 dims): Option A 9-bin percentile curve
      - ``use_option_b``: +18 dims (stddev + ceiling fraction per bin)
      - ``use_absolute_dbfs``: +9 dims (absolute dBFS at 9 bins)
      - ``use_high_res``: 16 bins instead of 9
      - ``use_rolloff_cluster``: +n_clusters dims (one-hot cluster ID)

    Returns float32 array of shape ``(config.n_features,)``.
    """
    parts: list[np.ndarray] = [build_audio_features(features)]

    if config.use_option_b:
        if features.chunk_stddev is not None and features.chunk_ceiling_frac is not None:
            parts.append(np.array(features.chunk_stddev, dtype=np.float32))
            parts.append(np.array(features.chunk_ceiling_frac, dtype=np.float32))
        else:
            # Synthetic fallback: stddev=0 (perfectly consistent), ceiling_frac=1.0
            parts.append(np.zeros(9, dtype=np.float32))
            parts.append(np.ones(9, dtype=np.float32))

    if config.use_absolute_dbfs:
        if features.absolute_dbfs is not None:
            parts.append(np.array(
                [db for _, db in features.absolute_dbfs], dtype=np.float32,
            ))
        else:
            parts.append(np.zeros(9, dtype=np.float32))

    # F7 cluster ID is injected externally (requires a fitted KMeans model),
    # so it is NOT populated here — the caller appends it after this call.

    audio = np.concatenate(parts)
    meta = build_metadata_features(metadata)
    vec = np.concatenate([audio, meta]).astype(np.float32)
    expected = config.n_features
    # Allow cluster dims to be added later by caller
    if not config.use_rolloff_cluster:
        assert vec.shape == (expected,), (
            f"feature vector dim mismatch: {vec.shape} != ({expected},)"
        )
    return vec


# ---------------------------------------------------------------------------
# Label encoding / decoding
# ---------------------------------------------------------------------------


def catalogue_entry_to_labels(entry: dict) -> np.ndarray:
    """Encode a catalogue entry's filter chain as a fixed 24-dim Y vector.

    MAX_FILTER_SLOTS=4 slots, each [type_LS, type_HS, type_PEQ, freq, gain, q].
    Filter type is one-hot encoded (3 dims) so XGBoost treats it as
    categorical, not ordinal. Unused slots are all zeros.
    """
    y = np.zeros(N_OUTPUT, dtype=np.float32)
    filters = entry.get("filters", [])
    for i, f in enumerate(filters[:MAX_FILTER_SLOTS]):
        slot = i * N_PER_SLOT
        # One-hot type encoding.
        type_idx = _TYPE_TO_IDX.get(str(f.get("type", "LowShelf")), 0)
        y[slot + type_idx] = 1.0
        # Continuous params.
        y[slot + N_TYPE] = float(f.get("freq", 0.0))
        y[slot + N_TYPE + 1] = float(f.get("gain", 0.0))
        y[slot + N_TYPE + 2] = float(f.get("q", 0.9))
    return y


def labels_to_filters(y: np.ndarray, gain_threshold: float = 0.5) -> list[dict]:
    """Decode a 24-dim Y vector back into a list of filter dicts.

    Filter type decoded from one-hot (argmax of first 3 dims per slot).
    Skips slots whose |gain| is below ``gain_threshold`` (empty/noise slots).
    Clamps all parameters to valid ranges.
    """
    filters = []
    for i in range(MAX_FILTER_SLOTS):
        slot = i * N_PER_SLOT
        # Decode one-hot type: argmax of the 3 type dims.
        type_probs = y[slot:slot + N_TYPE]
        type_idx = int(np.argmax(type_probs))
        freq = float(y[slot + N_TYPE])
        gain = float(y[slot + N_TYPE + 1])
        q = float(y[slot + N_TYPE + 2])

        if abs(gain) < gain_threshold:
            continue

        ftype = _IDX_TO_TYPE.get(type_idx, "LowShelf")
        filters.append({
            "type": ftype,
            "freq": float(np.clip(freq, 5.0, 200.0)),
            "gain": float(np.clip(gain, -30.0, 30.0)),
            "q": float(np.clip(q, 0.1, 10.0)),
        })
    return filters


# ---------------------------------------------------------------------------
# Downstream evaluation loss (production metric — not parameter MSE)
# ---------------------------------------------------------------------------


def downstream_loss(
    predicted_filters: list[dict],
    target_filters: list[dict],
    freqs_hz: np.ndarray,
    band_hz: tuple[float, float] = _DOWNSTREAM_BAND_HZ,
) -> float:
    """Mean absolute error in dB between predicted and target filter responses.

    This is the production metric per the ML plan: apply the predicted filter
    to flat audio and measure RMS error vs what the target filter produces
    across 20–80 Hz. Divergence between this and parameter MSE signals
    overfitting on parameter space rather than acoustic output.
    """
    from model.auto_beq import evaluate_filter_chain

    band_mask = (freqs_hz >= band_hz[0]) & (freqs_hz <= band_hz[1])
    if not band_mask.any():
        return 0.0

    pred_resp = evaluate_filter_chain(predicted_filters, freqs_hz) if predicted_filters else np.zeros_like(freqs_hz)
    tgt_resp = evaluate_filter_chain(target_filters, freqs_hz) if target_filters else np.zeros_like(freqs_hz)
    return float(np.mean(np.abs(pred_resp[band_mask] - tgt_resp[band_mask])))


# ---------------------------------------------------------------------------
# Catalogue deduplication
# ---------------------------------------------------------------------------

# Audio format quality ranking for picking the best variant per title.
_FORMAT_RANK = {
    "atmos": 0, "truehd": 1, "dts-hd ma": 2, "dd+ atmos": 3, "dd+": 4,
}


def _format_rank(entry: dict) -> int:
    joined = " ".join(t.lower() for t in entry.get("audioTypes", []))
    for fmt, rank in _FORMAT_RANK.items():
        if fmt in joined:
            return rank
    return 99


def deduplicate_by_title(entries: list[dict]) -> list[dict]:
    """Remove BD/UHD/streaming duplicates, keeping one entry per unique title.

    Groups by normalised title (lowercased, stripped). Within each group picks
    the highest-quality audio format entry (Atmos > TrueHD > DTS-HD MA > etc.).
    Must be called before any train/val/test split to avoid data leakage.
    """
    groups: dict[str, list[dict]] = {}
    for e in entries:
        key = str(e.get("title", "")).lower().strip()
        groups.setdefault(key, []).append(e)

    result = []
    for group in groups.values():
        best = min(group, key=_format_rank)
        result.append(best)
    return result


# ---------------------------------------------------------------------------
# Train / validation / test split
# ---------------------------------------------------------------------------


def _rolloff_severity_bucket(entry: dict) -> str:
    """Classify an entry by total filter gain as a proxy for rolloff severity."""
    total_gain = sum(abs(float(f.get("gain", 0))) for f in entry.get("filters", []))
    if total_gain >= 20.0:
        return "heavy"
    if total_gain >= 10.0:
        return "moderate"
    return "gentle"


def split_dataset(
    X: np.ndarray,
    Y: np.ndarray,
    entries: list[dict],
    val: float = 0.15,
    test: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split into train/val/test with stratification by rolloff severity.

    Stratifies by rolloff severity bucket (heavy/moderate/gentle) so each
    split has a representative mix of easy and hard titles.

    Returns (X_train, X_val, X_test, Y_train, Y_val, Y_test).
    """
    from sklearn.model_selection import train_test_split

    severity = [_rolloff_severity_bucket(e) for e in entries]

    # First cut out the test set.
    idx = np.arange(len(X))
    idx_trainval, idx_test = train_test_split(
        idx, test_size=test, random_state=seed,
        stratify=severity if len(set(severity)) > 1 else None,
    )
    severity_trainval = [severity[i] for i in idx_trainval]
    val_frac = val / (1.0 - test)
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=val_frac, random_state=seed,
        stratify=severity_trainval if len(set(severity_trainval)) > 1 else None,
    )
    return (
        X[idx_train], X[idx_val], X[idx_test],
        Y[idx_train], Y[idx_val], Y[idx_test],
    )


# ---------------------------------------------------------------------------
# Training-time augmentation (F1 — E41)
# ---------------------------------------------------------------------------


def augment_audio_features(
    X: np.ndarray,
    Y: np.ndarray,
    config: AugmentationConfig,
    n_audio: int = N_AUDIO_FEATURES,
) -> tuple[np.ndarray, np.ndarray]:
    """Create noisy copies of each training sample's audio features.

    For each original sample, ``config.n_copies`` augmented copies are
    generated by perturbing only the audio feature columns (``0:n_audio``).
    Metadata columns are preserved exactly.

    Augmentation pipeline per copy:
      1. Additive Gaussian noise ``N(0, σ)`` across all audio bins
      2. Independent per-bin uniform jitter ``U(-u, +u)``
      3. With probability ``smooth_prob``, a 3-point running average
         (simulates Welch smoothing artefact on real measurements)

    The original (clean) samples are prepended to the output so the model
    still sees the exact synthetic features alongside the noisy variants.

    Returns ``(X_augmented, Y_augmented)`` with shape ``(n_orig * (1 + n_copies), ...)``.
    """
    rng = np.random.default_rng(config.seed)
    n_orig = X.shape[0]
    augmented_X: list[np.ndarray] = [X]  # originals first

    for _ in range(config.n_copies):
        X_copy = X.copy()
        audio = X_copy[:, :n_audio].copy()

        # 1. Global Gaussian noise
        audio += rng.normal(0.0, config.gaussian_sigma_db, size=audio.shape)

        # 2. Per-bin uniform jitter
        audio += rng.uniform(
            -config.per_bin_uniform_db,
            config.per_bin_uniform_db,
            size=audio.shape,
        )

        # 3. Optional 3-point running average (simulates smoothing)
        mask = rng.random(n_orig) < config.smooth_prob
        if mask.any():
            kernel = np.array([0.25, 0.5, 0.25])
            for i in np.where(mask)[0]:
                audio[i, :] = np.convolve(audio[i, :], kernel, mode="same")

        X_copy[:, :n_audio] = audio
        augmented_X.append(X_copy)

    X_aug = np.vstack(augmented_X)
    Y_aug = np.tile(Y, (1 + config.n_copies, 1))

    log.info(
        "augmented training data: %d → %d samples (n_copies=%d, σ=%.1f, u=%.1f)",
        n_orig, X_aug.shape[0], config.n_copies,
        config.gaussian_sigma_db, config.per_bin_uniform_db,
    )
    return X_aug, Y_aug


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_xgboost(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    Y_val: np.ndarray | None = None,
    sample_weight: np.ndarray | None = None,
    augmentation: AugmentationConfig | None = None,
    n_audio: int = N_AUDIO_FEATURES,
) -> object:
    """Train an XGBoost multi-output regressor.

    Uses native XGBoost multi-output tree strategy which processes all output
    columns simultaneously.

    *augmentation* (F1/E41): if provided, generates noisy copies of audio
    features in the training set before fitting.  Only affects training data;
    validation and inference use clean features.

    *sample_weight* upweights specific training samples (used by reweighted
    training in E38 and confidence-weighted training in F6/E46).

    Returns the fitted model.
    """
    import xgboost as xgb

    if augmentation is not None:
        X_train, Y_train = augment_audio_features(
            X_train, Y_train, augmentation, n_audio=n_audio,
        )
        # Don't carry over sample_weight after augmentation — augmented
        # copies get uniform weight.  The original samples' weights would
        # need to be tiled, but for simplicity we drop custom weights when
        # augmenting (they serve different purposes).
        if sample_weight is not None:
            log.warning(
                "sample_weight dropped during augmentation — "
                "augmented copies receive uniform weight",
            )
            sample_weight = None

    model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        multi_strategy="multi_output_tree",
        random_state=42,
        verbosity=1,
    )
    fit_kwargs: dict = {}
    if X_val is not None and Y_val is not None:
        fit_kwargs["eval_set"] = [(X_val, Y_val)]
        fit_kwargs["verbose"] = False
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight

    model.fit(X_train, Y_train, **fit_kwargs)
    log.info("XGBoost training complete. n_features_in=%d", model.n_features_in_)
    return model


def train_xgboost_reweighted(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    freqs_hz: np.ndarray,
    n_rounds: int = 2,
) -> object:
    """Two-stage reweighted training (E38).

    Stage 1: standard MSE training. Stage 2: compute downstream dB loss
    per training sample, upweight high-loss samples, retrain. Focuses the
    model on samples where parameter-MSE doesn't produce good acoustic results.
    """
    model = train_xgboost(X_train, Y_train)

    for round_idx in range(1, n_rounds):
        Y_pred = model.predict(X_train)
        weights = np.ones(len(X_train), dtype=np.float32)
        for i in range(len(X_train)):
            pred_filters = labels_to_filters(Y_pred[i])
            target_filters = labels_to_filters(Y_train[i])
            loss = downstream_loss(pred_filters, target_filters, freqs_hz)
            weights[i] = max(1.0, loss)  # upweight high-loss samples

        log.info("reweighted round %d: mean_weight=%.2f max_weight=%.2f",
                 round_idx + 1, weights.mean(), weights.max())
        model = train_xgboost(X_train, Y_train, sample_weight=weights)

    return model


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Late fusion (E22) — separate audio + metadata models, blended predictions
# ---------------------------------------------------------------------------


class LateFusionModel:
    """Wraps two independent XGBoost sub-models (audio-only + metadata-only).

    Predictions are blended: ``alpha * Y_audio + (1 - alpha) * Y_meta``.
    Prevents cross-feature overfitting that occurs with early fusion on
    synthetic training data (see E18e ablation).

    Compatible with ``TrainedModelAdvisor`` — has ``.predict(X)``.
    """

    def __init__(
        self,
        model_audio: object,
        model_meta: object,
        alpha: float = 0.5,
        n_audio: int = N_AUDIO_FEATURES,
    ) -> None:
        self.model_audio = model_audio
        self.model_meta = model_meta
        self.alpha = alpha
        self._n_audio = n_audio

    def predict(self, X: np.ndarray) -> np.ndarray:
        n_audio = self._n_audio
        X_audio = np.zeros_like(X)
        X_audio[:, :n_audio] = X[:, :n_audio]
        X_meta = np.zeros_like(X)
        X_meta[:, n_audio:] = X[:, n_audio:]

        Y_audio = self.model_audio.predict(X_audio)
        Y_meta = self.model_meta.predict(X_meta)
        return self.alpha * Y_audio + (1.0 - self.alpha) * Y_meta


def train_late_fusion(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    alpha: float = 0.5,
    n_audio: int | None = None,
    augmentation: AugmentationConfig | None = None,
) -> LateFusionModel:
    """Train a late-fusion model: separate audio and metadata XGBoost models.

    Each sub-model sees only its own feature subset (audio dims zeroed for
    the metadata model and vice versa). Their predictions are blended with
    weight ``alpha`` (audio) vs ``1 - alpha`` (metadata).

    *n_audio* specifies how many leading columns are audio features.
    Defaults to ``N_AUDIO_FEATURES`` (9) for backward compatibility;
    pass ``config.n_total_audio`` when using extended audio features
    (Option B, absolute dBFS, high-res bins, etc.).

    *augmentation* (F1/E41): if provided, augments the audio sub-model's
    training data.  The metadata sub-model always trains on clean data
    (metadata features are deterministic and don't need augmentation).
    """
    if n_audio is None:
        n_audio = N_AUDIO_FEATURES

    X_audio = np.zeros_like(X_train)
    X_audio[:, :n_audio] = X_train[:, :n_audio]
    X_meta = np.zeros_like(X_train)
    X_meta[:, n_audio:] = X_train[:, n_audio:]

    log.info("training audio-only sub-model (%d features active)...", n_audio)
    model_audio = train_xgboost(
        X_audio, Y_train, augmentation=augmentation, n_audio=n_audio,
    )

    log.info("training metadata-only sub-model (%d features active)...",
             X_train.shape[1] - n_audio)
    model_meta = train_xgboost(X_meta, Y_train)

    log.info("late fusion complete, alpha=%.2f", alpha)
    return LateFusionModel(model_audio, model_meta, alpha=alpha, n_audio=n_audio)


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------


def save_model(model: object, path: str) -> None:
    """Serialise a trained model to disk via joblib."""
    import joblib

    joblib.dump(model, path)
    log.info("Model saved to %s", path)


def load_model(path: str) -> object:
    """Load a model serialised by ``save_model``."""
    import joblib

    model = joblib.load(path)
    log.info("Model loaded from %s", path)
    return model


# ---------------------------------------------------------------------------
# TrainedModelAdvisor — Advisor protocol implementation
# ---------------------------------------------------------------------------


class TrainedModelAdvisor:
    """Advisor backed by a trained regression model (XGBoost → CNN → transformer).

    Implements the ``Advisor`` protocol from ``auto_beq_advisor``. Given a
    film's ``MediaMetadata`` and measured ``CurveFeatures``, builds the 60-dim
    input vector, runs inference, decodes the predicted 16-dim label vector
    into filter dicts, and returns an ``Advice`` with the filters set.

    Load with ``TrainedModelAdvisor.load(path)`` for production use.
    """

    name = "trained_model"

    def __init__(self, model: object) -> None:
        self._model = model

    @classmethod
    def load(cls, path: str) -> "TrainedModelAdvisor":
        """Load from a path written by ``save_model``."""
        return cls(load_model(path))

    def advise(self, metadata: MediaMetadata, features: CurveFeatures) -> Advice:
        from model.auto_beq_advisor import Advice, _clamp_advice

        x = build_feature_vector(features, metadata)
        y_pred = self._model.predict(x.reshape(1, -1))[0]
        filters = labels_to_filters(y_pred)

        if not filters:
            log.debug("trained_model: no filters predicted above gain threshold")
            return _clamp_advice(
                Advice(
                    max_gain_db=10.0,
                    reasoning="trained_model: no filters predicted",
                    confidence=0.2,
                    source="trained_model",
                ),
                source="trained_model",
            )

        total_gain = sum(abs(f["gain"]) for f in filters)
        primary_knee = filters[0]["freq"]
        log.debug(
            "trained_model: %d filter(s), total_gain=%.1f dB, primary_knee=%.1f Hz",
            len(filters), total_gain, primary_knee,
        )
        return _clamp_advice(
            Advice(
                max_gain_db=total_gain,
                knee_hz=primary_knee,
                filters=tuple(filters),
                reasoning=f"trained_model: {len(filters)} filter(s) predicted",
                confidence=0.5,
                source="trained_model",
            ),
            source="trained_model",
        )


class LateFusionAdvisor:
    """Advisor backed by a late-fusion model (E22).

    Two independent XGBoost sub-models (audio-only + metadata-only) with
    blended predictions. Prevents the cross-feature overfitting observed
    in E18e when early-fusing audio and metadata on synthetic data.
    """

    name = "late_fusion"

    def __init__(self, model: LateFusionModel) -> None:
        self._model = model

    @classmethod
    def load(cls, path: str) -> "LateFusionAdvisor":
        return cls(load_model(path))

    def advise(self, metadata: MediaMetadata, features: CurveFeatures) -> Advice:
        from model.auto_beq_advisor import Advice, _clamp_advice

        x = build_feature_vector(features, metadata)
        y_pred = self._model.predict(x.reshape(1, -1))[0]
        filters = labels_to_filters(y_pred)

        if not filters:
            return _clamp_advice(
                Advice(max_gain_db=10.0, reasoning="late_fusion: no filters predicted",
                       confidence=0.2, source="late_fusion"),
                source="late_fusion",
            )

        total_gain = sum(abs(f["gain"]) for f in filters)
        primary_knee = filters[0]["freq"]
        return _clamp_advice(
            Advice(
                max_gain_db=total_gain, knee_hz=primary_knee,
                filters=tuple(filters),
                reasoning=f"late_fusion(α={self._model.alpha:.2f}): {len(filters)} filter(s)",
                confidence=0.5, source="late_fusion",
            ),
            source="late_fusion",
        )
