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
    # G6: per-bin sigma (measured noise profile).  When set, overrides
    # gaussian_sigma_db with a different σ for each audio bin.
    per_bin_sigma: tuple[float, ...] | None = None

    @property
    def label(self) -> str:
        sigma_part = "targeted" if self.per_bin_sigma else f"s{self.gaussian_sigma_db:.1f}"
        return (
            f"aug-{sigma_part}"
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
    # T1.1/E83: audio foundation model embedding as additional features.
    # ``foundation_model`` is a key into ``FOUNDATION_MODEL_DIMS`` (e.g.
    # "whisper-tiny"). When set, ``build_feature_vector`` concatenates a
    # pooled embedding vector of the model's native dim. Synthetic samples
    # get a zero-vector fallback of the same dim.
    foundation_model: str | None = None

    @property
    def foundation_dim(self) -> int:
        if self.foundation_model is None:
            return 0
        # Local import to avoid circular dependency at module load time.
        from model.auto_beq_advisor import foundation_embedding_dim
        return foundation_embedding_dim(self.foundation_model)

    @property
    def n_total_audio(self) -> int:
        n = 16 if self.use_high_res else self.n_audio
        if self.use_option_b:
            n += 18  # 9 stddev + 9 ceiling_fraction (always at 9 bins)
        if self.use_absolute_dbfs:
            n += 9   # absolute dBFS at 9 bins
        if self.use_rolloff_cluster:
            n += self.n_clusters  # one-hot cluster ID
        n += self.foundation_dim  # T1.1/E83
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
        if self.foundation_model is not None:
            parts.append(f"fnd-{self.foundation_model}")
        return "+".join(parts) if parts else "A9"


DEFAULT_AUDIO_CONFIG = AudioFeatureConfig()


# ---------------------------------------------------------------------------
# Label constants (filter parameter output vector)
# ---------------------------------------------------------------------------

MAX_FILTER_SLOTS = 6  # E39a: was 4→8 (over-predicted), 6 covers 77% of catalogue
N_TYPE = 3  # LowShelf, HighShelf, PeakingEQ — one-hot encoded
# Per slot: [type_LS, type_HS, type_PEQ, freq_hz, gain_db, q] = 6 values
N_PER_SLOT = N_TYPE + 3
N_OUTPUT = MAX_FILTER_SLOTS * N_PER_SLOT  # 6 × 6 = 36

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
    if config.use_high_res:
        parts: list[np.ndarray] = [build_audio_features_high_res(features)]
    else:
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

    # T1.1/E83: foundation model embedding. Pulled from features when
    # available (populated for real WAVs upstream), zeros otherwise
    # (synthetic samples — no raw audio to embed).
    if config.foundation_model is not None:
        dim = config.foundation_dim
        if features.foundation_embedding is not None:
            emb = np.asarray(features.foundation_embedding, dtype=np.float32)
            if emb.shape != (dim,):
                raise ValueError(
                    f"foundation_embedding shape mismatch: got {emb.shape}, "
                    f"expected ({dim},) for model {config.foundation_model!r}",
                )
            parts.append(emb)
        else:
            parts.append(np.zeros(dim, dtype=np.float32))

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


# ---------------------------------------------------------------------------
# H4 — Response curve label encoding (alternative to filter parameter labels)
# ---------------------------------------------------------------------------


def catalogue_entry_to_response_labels(
    entry: dict,
    freqs_hz: np.ndarray,
    bins_hz: list[float] = OPTION_A_BINS_HZ,
    fs: int = 1000,
) -> np.ndarray:
    """Encode a catalogue entry's filter chain as a 9-bin response curve (H4).

    Predicting the response curve directly (instead of 24-dim filter
    parameters) eliminates parameter-space ambiguity: two different chains
    can produce identical responses, but their response curves are unique.
    Aligns the training objective with the evaluation metric (downstream
    dB error in response space).

    Returns a 9-dim float32 array of dB values at the Option A frequency
    bins, normalised to 0 dB at 80 Hz.
    """
    from model.auto_beq import evaluate_filter_chain

    response = evaluate_filter_chain(entry.get("filters", []), freqs_hz, fs=fs)
    out = np.array([float(np.interp(b, freqs_hz, response)) for b in bins_hz])
    out -= out[-1]  # normalise to 0 dB at 80 Hz (last bin)
    return out.astype(np.float32)


def response_labels_to_filters(
    y: np.ndarray,
    freqs_hz: np.ndarray,
    bins_hz: list[float] = OPTION_A_BINS_HZ,
    fs: int = 1000,
    max_filters: int = 6,
) -> list[dict]:
    """Decode a 9-bin predicted response into a filter chain (H4 inference).

    Interpolates the predicted curve onto the full frequency grid, then
    fits a chain via ``propose_filters()``.  Note: the fitter expects
    "cancel this curve", so we flip the sign.
    """
    from model.auto_beq import propose_filters

    # Interpolate 9 predicted bins to the full grid.
    full_curve = np.interp(freqs_hz, bins_hz, y).astype(np.float64)
    return propose_filters(
        -full_curve, freqs_hz, fs=fs,
        band=(5.0, 80.0), max_filters=max_filters,
    )


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


# Trusted authors for H1d response-averaging (excludes remixmark — worst
# in validation — and bombaycat007 — too few samples to assess).
TRUSTED_AUTHORS = frozenset({
    "aron7awol", "t1g8rsfan", "kaelaria", "mobe1969", "mikejl", "halcyon888",
})


def _process_multi_author_group(args: tuple) -> dict | None:
    """Worker for parallel response-avg dedup.  Module-level for picklability."""
    per_author, strategy, freqs_hz, fs, band, max_filters = args
    from model.auto_beq import evaluate_filter_chain, propose_filters

    curves = []
    for e in per_author:
        try:
            response = evaluate_filter_chain(e["filters"], freqs_hz, fs=fs)
            curves.append(response)
        except Exception:
            pass
    if not curves:
        return None

    stacked = np.stack(curves, axis=0)
    if strategy == "median":
        consensus = np.median(stacked, axis=0)
    else:
        consensus = np.mean(stacked, axis=0)

    try:
        new_filters = propose_filters(
            -consensus, freqs_hz, fs=fs, band=band, max_filters=max_filters,
        )
    except Exception:
        return None

    best_meta = min(per_author, key=_format_rank)
    consensus_entry = dict(best_meta)
    consensus_entry["filters"] = new_filters
    consensus_entry["_consensus_n_authors"] = len(per_author)
    return consensus_entry


def deduplicate_by_title_response_avg(
    entries: list[dict],
    freqs_hz: np.ndarray,
    fs: int = 1000,
    strategy: str = "mean",
    band: tuple[float, float] = (5.0, 80.0),
    max_filters: int = 6,
    n_workers: int | None = None,
) -> list[dict]:
    """Group by title; average response curves across authors and refit.

    For multi-author titles, computes the response curve of each author's
    chain, averages in dB space across authors, then fits a new chain to
    the consensus curve via ``propose_filters()``.  This eliminates
    parameter-space ambiguity (two different chains can produce identical
    responses) by averaging in the only space where it makes sense.

    Strategies:
      * ``"mean"`` — equal-weight mean across all authors
      * ``"median"`` — robust to outlier authors (e.g. remixmark)
      * ``"trusted"`` — mean across only TRUSTED_AUTHORS
      * ``"format"`` — original behaviour (pick highest-format entry)

    Multi-author groups are refit in parallel via ``ProcessPoolExecutor``
    (each ``propose_filters()`` call is an independent scipy optimisation).
    """
    if strategy == "format":
        return deduplicate_by_title(entries)

    groups: dict[str, list[dict]] = {}
    for e in entries:
        key = str(e.get("title", "")).lower().strip()
        groups.setdefault(key, []).append(e)

    # Split groups into single-author (fast path) and multi-author (parallel).
    single_author_results: list[dict] = []
    multi_author_jobs: list[tuple] = []

    for group in groups.values():
        by_author: dict[str, list[dict]] = {}
        for e in group:
            a = str(e.get("author", "")).strip().lower()
            by_author.setdefault(a, []).append(e)

        if len(by_author) <= 1:
            single_author_results.append(min(group, key=_format_rank))
            continue

        per_author = [min(eps, key=_format_rank) for eps in by_author.values()]

        if strategy == "trusted":
            per_author = [
                e for e in per_author
                if str(e.get("author", "")).strip().lower() in TRUSTED_AUTHORS
            ]
            if not per_author:
                single_author_results.append(min(group, key=_format_rank))
                continue

        multi_author_jobs.append(
            (per_author, strategy, freqs_hz, fs, band, max_filters),
        )

    log.info(
        "response-avg dedup (%s): %d single-author + %d multi-author titles to refit",
        strategy, len(single_author_results), len(multi_author_jobs),
    )

    # Refit multi-author groups in parallel.
    consensus_results: list[dict] = []
    n_refit_failed = 0
    if multi_author_jobs:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import os as _os

        if n_workers is None:
            n_workers = max(2, _os.cpu_count() // 2)

        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = [
                pool.submit(_process_multi_author_group, job)
                for job in multi_author_jobs
            ]
            done = 0
            for fut in as_completed(futures):
                done += 1
                if done % 100 == 0:
                    log.info("refit progress: %d / %d", done, len(multi_author_jobs))
                res = fut.result()
                if res is None:
                    n_refit_failed += 1
                    continue
                consensus_results.append(res)

    result = single_author_results + consensus_results
    log.info(
        "response-avg dedup (%s): %d titles total, %d multi-author averaged, %d refit failures",
        strategy, len(result), len(consensus_results), n_refit_failed,
    )
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

        # 1. Gaussian noise (global or per-bin)
        if config.per_bin_sigma is not None:
            for b in range(min(n_audio, len(config.per_bin_sigma))):
                audio[:, b] += rng.normal(0.0, config.per_bin_sigma[b], size=n_orig)
        else:
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
    n_estimators: int = 400,
    max_depth: int = 6,
    learning_rate: float = 0.05,
) -> object:
    """Train an XGBoost multi-output regressor.

    Uses native XGBoost multi-output tree strategy which processes all output
    columns simultaneously.

    *augmentation* (F1/E41): if provided, generates noisy copies of audio
    features in the training set before fitting.  Only affects training data;
    validation and inference use clean features.

    *sample_weight* upweights specific training samples (used by reweighted
    training in E38 and confidence-weighted training in F6/E46).

    *n_estimators*, *max_depth*, *learning_rate* (G5): XGBoost hyperparams,
    configurable for tuning with augmented (larger) training data.

    Returns the fitted model.
    """
    import xgboost as xgb

    if augmentation is not None:
        X_train, Y_train = augment_audio_features(
            X_train, Y_train, augmentation, n_audio=n_audio,
        )
        if sample_weight is not None:
            log.warning(
                "sample_weight dropped during augmentation — "
                "augmented copies receive uniform weight",
            )
            sample_weight = None

    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        multi_strategy="multi_output_tree",
        random_state=42,
        # Pin to single-threaded (E75 determinism fix).  When the
        # experiment harness runs batches via ThreadPoolExecutor, each
        # concurrent XGBoost call contending for the same cores produces
        # non-deterministic histogram schedules — the same Baseline
        # config produced 2.78/2.82/2.78/3.00 dB across four 932-WAV
        # runs (±0.22 dB noise).  n_jobs=1 makes each training fully
        # deterministic; parallelism is still handled at the batch level
        # by the harness's ThreadPoolExecutor.
        n_jobs=1,
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

    def predict_with_alphas(
        self,
        X: np.ndarray,
        alphas: np.ndarray,
    ) -> np.ndarray:
        """Predict with per-sample alphas (G8 — per-author alpha selection).

        ``alphas`` must have shape ``(n_samples,)``.  For each row, the
        sub-model predictions are blended with that row's alpha.  This
        enables per-author optimal alpha lookup at inference time.
        """
        n_audio = self._n_audio
        X_audio = np.zeros_like(X)
        X_audio[:, :n_audio] = X[:, :n_audio]
        X_meta = np.zeros_like(X)
        X_meta[:, n_audio:] = X[:, n_audio:]

        Y_audio = self.model_audio.predict(X_audio)
        Y_meta = self.model_meta.predict(X_meta)
        a = alphas.reshape(-1, 1)
        return a * Y_audio + (1.0 - a) * Y_meta

    def predict_marginalized(
        self,
        X: np.ndarray,
        author_col_start: int,
        n_author: int = N_AUTHOR,
        weights: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict with author marginalization (H2/E54).

        For each sample, query the model with each author one-hot identity
        in turn, then average the predictions.  The user gets a consensus
        prediction without specifying an author — equivalent to a Bayesian
        model average over the author categorical variable.

        ``weights`` (optional, shape ``(n_author,)``): per-author averaging
        weights.  Default is uniform.  Use frequency weights for
        catalogue-wide consensus or quality weights to down-weight noisy
        authors.
        """
        if weights is None:
            weights = np.ones(n_author, dtype=np.float32) / n_author
        else:
            weights = np.asarray(weights, dtype=np.float32)
            weights = weights / weights.sum()

        # Pre-compute audio prediction once (independent of author).
        n_audio = self._n_audio
        X_audio = np.zeros_like(X)
        X_audio[:, :n_audio] = X[:, :n_audio]
        Y_audio = self.model_audio.predict(X_audio)

        # Sum metadata predictions across all author identities.
        Y_meta_sum = None
        for author_idx in range(n_author):
            X_meta = np.zeros_like(X)
            X_meta[:, n_audio:] = X[:, n_audio:]
            # Zero out existing author one-hot, set the chosen one.
            X_meta[:, author_col_start : author_col_start + n_author] = 0
            X_meta[:, author_col_start + author_idx] = 1.0
            Y_meta_a = self.model_meta.predict(X_meta) * weights[author_idx]
            if Y_meta_sum is None:
                Y_meta_sum = Y_meta_a
            else:
                Y_meta_sum = Y_meta_sum + Y_meta_a

        return self.alpha * Y_audio + (1.0 - self.alpha) * Y_meta_sum


# Author column offset in the full feature vector (after audio + metadata
# preceding fields): 9 + 1 (year) + 6 (format) + 3 (source) + 32 (studio)
# + 22 (mixer) + 10 (genre) + 5 (country) + 1 (runtime) + 1 (rating) = 90
# in metadata, 99 in full vector.
AUTHOR_COL_OFFSET_IN_METADATA = (
    1 + N_AUDIO_FORMAT + N_SOURCE + N_STUDIO + N_MIXER
    + N_GENRE + N_COUNTRY + 1 + 1
)  # = 81

def author_col_start(n_audio: int = N_AUDIO_FEATURES) -> int:
    """Return the column index where the author one-hot starts in the
    full feature vector.  Depends on n_audio (which varies for F2/F3/F11)."""
    return n_audio + AUTHOR_COL_OFFSET_IN_METADATA


# Catalogue-wide author frequency weights (from full ~8k catalogue).
# Used by H2b (frequency-weighted marginalization).
AUTHOR_FREQUENCY_WEIGHTS: dict[str, float] = {
    "mobe1969": 0.57,
    "aron7awol": 0.13,
    "kaelaria": 0.08,
    "mikejl": 0.06,
    "remixmark": 0.06,
    "t1g8rsfan": 0.04,
    "halcyon888": 0.03,
    "bombaycat007": 0.02,
    "unknown": 0.01,
}

# Quality weights from G2 per-author validation results.
# Inverse of mean loss — better authors get more weight.
# Mean losses (G2a): t1g8rsfan 0.89, aron7awol 1.64, mobe1969 1.93,
# kaelaria 2.01, halcyon888 2.27, remixmark 2.85.  Other authors absent
# from validation get 1/2.0 ≈ 0.5 default.
AUTHOR_QUALITY_WEIGHTS: dict[str, float] = {
    "mobe1969": 1.0 / 1.93,
    "aron7awol": 1.0 / 1.64,
    "kaelaria": 1.0 / 2.01,
    "mikejl": 1.0 / 2.5,        # not in validation, assume average
    "remixmark": 1.0 / 2.85,
    "t1g8rsfan": 1.0 / 0.89,
    "halcyon888": 1.0 / 2.27,
    "bombaycat007": 1.0 / 2.5,  # not in validation, assume average
    "unknown": 1.0 / 2.5,
}


def author_weights_array(
    weights_dict: dict[str, float] | None = None,
    vocab: list[str] | None = None,
) -> np.ndarray:
    """Convert per-author weight dict into array aligned with _AUTHOR_VOCAB."""
    if vocab is None:
        vocab = _AUTHOR_VOCAB
    if weights_dict is None:
        return np.ones(len(vocab), dtype=np.float32) / len(vocab)
    return np.array(
        [weights_dict.get(a, 1.0 / len(vocab)) for a in vocab],
        dtype=np.float32,
    )


# Per-author optimal alphas, derived from G2 sweep results.
# kaelaria and t1g8rsfan want audio-heavy (their styles correlate with
# measured rolloff); aron7awol/halcyon888/remixmark want metadata-balanced
# (their styles correlate with film context more than measured curve).
PER_AUTHOR_ALPHA: dict[str, float] = {
    "aron7awol": 0.5,
    "halcyon888": 0.5,
    "kaelaria": 0.9,
    "mobe1969": 0.7,
    "remixmark": 0.5,
    "t1g8rsfan": 0.7,
}
DEFAULT_PER_AUTHOR_ALPHA = 0.6  # for unknown authors


# ---------------------------------------------------------------------------
# I-series: automated author selection from metadata (E68+)
# ---------------------------------------------------------------------------


def strip_author_columns(X_full: np.ndarray, n_audio: int) -> np.ndarray:
    """Drop the 9 author one-hot columns from a feature matrix.

    Used for the author classifier — we want metadata WITHOUT the author
    one-hot (since that's the target we're predicting).

    The author block starts at column ``n_audio + AUTHOR_COL_OFFSET_IN_METADATA``
    and is ``N_AUTHOR`` columns wide.
    """
    start = n_audio + AUTHOR_COL_OFFSET_IN_METADATA
    end = start + N_AUTHOR
    return np.concatenate([X_full[:, :start], X_full[:, end:]], axis=1)


def train_author_classifier(
    X_train: np.ndarray,
    entries: list[dict],
    n_audio: int = N_AUDIO_FEATURES,
) -> object:
    """Train an XGBoost classifier: metadata → author label (I1/E69).

    Trains on the metadata portion of the full feature vector (with the
    author one-hot columns dropped — that's the target).  The classifier
    learns implicit author specialisation patterns from the catalogue
    (e.g. mobe1969 dominates pre-2000 titles, t1g8rsfan does modern Atmos).

    At inference, given an uncatalogued film's metadata, the classifier
    predicts which author would most likely have scored it.  That author's
    optimal alpha (from PER_AUTHOR_ALPHA) is then used for the final BEQ
    prediction.

    Returns a fitted XGBClassifier with an attached
    ``_author_index_map`` attribute mapping the contiguous training
    labels back to ``_AUTHOR_VOCAB`` indices (used by
    ``predict_alpha_from_metadata``).
    """
    from xgboost import XGBClassifier

    X_no_author = strip_author_columns(X_train, n_audio)
    raw_y = np.array([
        _AUTHOR_VOCAB.index(
            str(e.get("author", "unknown")).strip().lower()
            if str(e.get("author", "unknown")).strip().lower() in _AUTHOR_VOCAB
            else "unknown"
        )
        for e in entries
    ])

    # XGBClassifier requires contiguous class labels [0, k-1].  If the
    # training set is missing some authors (small folds, low-volume
    # authors), the raw author indices may be non-contiguous (e.g.
    # [0, 1, 2, 5]).  Remap to contiguous and store the inverse mapping
    # so predict_alpha_from_metadata can recover the original indices.
    unique_authors = sorted(set(int(v) for v in raw_y))
    author_idx_to_contig = {a: i for i, a in enumerate(unique_authors)}
    y = np.array([author_idx_to_contig[int(v)] for v in raw_y])

    clf = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        objective="multi:softprob",
        num_class=len(unique_authors),
        random_state=42,
        n_jobs=1,  # E75: determinism under ThreadPoolExecutor concurrency
        verbosity=0,
    )
    clf.fit(X_no_author, y)
    # Inverse mapping: contiguous label index → _AUTHOR_VOCAB index.
    clf._author_index_map = np.array(unique_authors, dtype=int)
    log.info(
        "author classifier trained: %d samples, %d distinct authors, "
        "train accuracy=%.3f",
        len(y), len(unique_authors),
        float((clf.predict(X_no_author) == y).mean()),
    )
    return clf


def predict_alpha_from_metadata(
    classifier: object,
    X_full: np.ndarray,
    n_audio: int = N_AUDIO_FEATURES,
    method: str = "hard",
) -> tuple[np.ndarray, np.ndarray]:
    """Predict per-sample alphas from a fitted author classifier.

    Methods:
      * ``"hard"`` — argmax author, look up its alpha (I1a)
      * ``"soft_blend"`` — sum_i(prob_i * alpha_i), single blended alpha (I1b/I3)
      * ``"top3"`` — top-3 authors, mean of their alphas weighted by prob (I1c)

    Returns ``(alphas, predicted_author_indices)`` — the predicted author
    indices are returned for diagnostic / per-sample logging.
    """
    X_no_author = strip_author_columns(X_full, n_audio)
    raw_probs = classifier.predict_proba(X_no_author)  # shape (n, n_classes)

    # The classifier was trained on contiguous labels [0..k-1] but those
    # labels correspond to a subset of _AUTHOR_VOCAB indices.  Use the
    # ``_author_index_map`` attached at training time to expand the probs
    # back to a (n, N_AUTHOR) matrix with zeros for absent authors.
    n_samples = raw_probs.shape[0]
    probs = np.zeros((n_samples, N_AUTHOR), dtype=np.float64)
    author_index_map = getattr(classifier, "_author_index_map", None)
    if author_index_map is not None:
        for contig_col, vocab_idx in enumerate(author_index_map):
            if 0 <= int(vocab_idx) < N_AUTHOR:
                probs[:, int(vocab_idx)] = raw_probs[:, contig_col]
    else:
        # Fall back to classifier.classes_ for older models.
        classes = list(getattr(classifier, "classes_", range(raw_probs.shape[1])))
        for col_idx, class_label in enumerate(classes):
            if 0 <= int(class_label) < N_AUTHOR:
                probs[:, int(class_label)] = raw_probs[:, col_idx]

    author_alphas = np.array(
        [PER_AUTHOR_ALPHA.get(a, DEFAULT_PER_AUTHOR_ALPHA) for a in _AUTHOR_VOCAB],
        dtype=np.float32,
    )

    if method == "hard":
        author_idx = probs.argmax(axis=1)
        alphas = author_alphas[author_idx]
        return alphas, author_idx

    if method == "soft_blend":
        alphas = (probs * author_alphas[None, :]).sum(axis=1)
        return alphas.astype(np.float32), probs.argmax(axis=1)

    if method == "top3":
        # For each sample, take the top-3 most probable authors and weight
        # their alphas by their (renormalised) probabilities.
        top3_idx = np.argsort(-probs, axis=1)[:, :3]
        top3_probs = np.take_along_axis(probs, top3_idx, axis=1)
        top3_probs = top3_probs / top3_probs.sum(axis=1, keepdims=True)
        top3_alphas = author_alphas[top3_idx]
        alphas = (top3_probs * top3_alphas).sum(axis=1)
        return alphas.astype(np.float32), top3_idx[:, 0]

    raise ValueError(f"unknown method: {method}")


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
# F6/E46 — Confidence-weighted training (inter-author agreement)
# ---------------------------------------------------------------------------


def compute_agreement_weights(
    entries: list[dict],
    freqs_hz: np.ndarray,
) -> np.ndarray:
    """Compute per-sample training weights based on inter-author agreement.

    BEQ catalogue entries vary in reliability.  When multiple authors profile
    the same title and their filter chains produce similar frequency responses,
    that's high-confidence ground truth.  When they disagree, the entry is
    noisy.

    Weighting by agreement is analogous to *label smoothing* in
    classification — it down-weights unreliable labels so the model
    focuses on clean signal.

    For titles with 2+ entries from different authors: compute mean pairwise
    ``downstream_loss`` between their filter chains.  Weight =
    ``1 / (1 + mean_pairwise_loss)``.

    For single-author or single-entry titles: weight = 1.0 (neutral).

    Returns array of shape ``(len(entries),)`` with weights ≥ 0.
    """
    from model.auto_beq import evaluate_filter_chain

    # Group by normalised title.
    groups: dict[str, list[int]] = {}
    for i, e in enumerate(entries):
        key = str(e.get("title", "")).lower().strip()
        groups.setdefault(key, []).append(i)

    weights = np.ones(len(entries), dtype=np.float32)

    for indices in groups.values():
        if len(indices) < 2:
            continue
        # Pairwise downstream loss across authors.
        losses: list[float] = []
        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                fa = entries[indices[a]].get("filters", [])
                fb = entries[indices[b]].get("filters", [])
                if fa and fb:
                    losses.append(downstream_loss(fa, fb, freqs_hz))
        if losses:
            mean_loss = float(np.mean(losses))
            w = 1.0 / (1.0 + mean_loss)
            for idx in indices:
                weights[idx] = w

    log.info(
        "agreement weights: mean=%.2f min=%.2f max=%.2f",
        weights.mean(), weights.min(), weights.max(),
    )
    return weights


# ---------------------------------------------------------------------------
# F7/E47 — Rolloff shape clustering
# ---------------------------------------------------------------------------


def compute_rolloff_clusters(
    X_audio: np.ndarray,
    n_clusters: int = 6,
) -> tuple[object, np.ndarray]:
    """K-means clustering on audio feature vectors (F7/E47).

    Discovers natural rolloff shape families (e.g. "Disney 2010s Atmos
    rolloff", "1990s action cliff") via unsupervised clustering.  The
    cluster ID becomes a categorical feature, compressing complex rolloff
    patterns into a single high-information split point for XGBoost.

    This is a form of *feature engineering via unsupervised learning* —
    the same technique used in NLP (word2vec clusters) and image recognition
    (visual bag-of-words).

    Returns ``(kmeans_model, cluster_ids)`` where ``cluster_ids`` is a
    1-D array of shape ``(n_samples,)`` with values in ``[0, n_clusters)``.
    The kmeans model should be applied to validation data via
    ``kmeans_model.predict(X_audio_val)``.
    """
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(X_audio)
    log.info(
        "rolloff clusters (%d): sizes=%s",
        n_clusters,
        np.bincount(cluster_ids).tolist(),
    )
    return kmeans, cluster_ids


def cluster_ids_to_onehot(ids: np.ndarray, n_clusters: int) -> np.ndarray:
    """Convert cluster IDs to one-hot encoded matrix."""
    onehot = np.zeros((len(ids), n_clusters), dtype=np.float32)
    onehot[np.arange(len(ids)), ids] = 1.0
    return onehot


# ---------------------------------------------------------------------------
# F9/E48 — Downstream loss as training objective (2-phase)
# ---------------------------------------------------------------------------


def train_xgboost_downstream(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    freqs_hz: np.ndarray,
    n_rounds: int = 2,
) -> object:
    """Two-phase training with downstream dB loss weighting (F9/E48).

    Currently XGBoost trains on MSE of filter parameters, but two
    different filter parameter sets can produce nearly identical acoustic
    results (*parameter-space equivalence*).  This is analogous to
    *perceptual loss* in image generation — pixel MSE penalises visually
    identical images differently depending on pixel arrangement.

    Phase 1: standard MSE training (400 trees).
    Phase 2+: evaluate each training sample's *acoustic* error (downstream
    dB loss, not parameter MSE), then upweight samples where parameter-MSE
    succeeded but acoustic output diverged.  This focuses the model on the
    acoustically meaningful errors.

    Differs from E38 ``train_xgboost_reweighted`` in weighting strategy:
    E38 uses ``max(1, loss)`` (floor at 1), this uses ``1 + loss²``
    (quadratic emphasis on high-loss samples).
    """
    model = train_xgboost(X_train, Y_train)

    for round_idx in range(1, n_rounds):
        Y_pred = model.predict(X_train)
        weights = np.ones(len(X_train), dtype=np.float32)
        for i in range(len(X_train)):
            pred_filters = labels_to_filters(Y_pred[i])
            target_filters = labels_to_filters(Y_train[i])
            loss = downstream_loss(pred_filters, target_filters, freqs_hz)
            weights[i] = 1.0 + loss * loss  # quadratic emphasis

        log.info(
            "downstream-loss round %d: mean_weight=%.2f max_weight=%.2f",
            round_idx + 1, weights.mean(), weights.max(),
        )
        model = train_xgboost(X_train, Y_train, sample_weight=weights)

    return model


# ---------------------------------------------------------------------------
# F11/E49 — Multi-resolution audio features (16 bins)
# ---------------------------------------------------------------------------

# 16 bins with concentration in the 10-40 Hz critical range where most
# BEQ correction happens.  The standard 9 bins (20-80 Hz) have only 2
# below 30 Hz.  This is analogous to mel-scale binning in speech
# recognition — allocate resolution where human (or in this case,
# subwoofer) perception is most sensitive.
OPTION_A_BINS_HR = [
    10.0, 12.0, 15.0, 18.0, 20.0, 22.0, 25.0, 28.0,
    30.0, 33.0, 35.0, 40.0, 50.0, 60.0, 70.0, 80.0,
]


def build_audio_features_high_res(features: CurveFeatures) -> np.ndarray:
    """Build 16-bin high-resolution audio feature vector (F11/E49).

    Uses OPTION_A_BINS_HR with denser sampling in 10-40 Hz.  Interpolates
    from the full normalised curve rather than nearest-bin from the 12-point
    sample (which doesn't cover 10-18 Hz well).
    """
    sample_pts = list(features.curve_sample_points)
    if not sample_pts:
        return np.zeros(len(OPTION_A_BINS_HR), dtype=np.float32)

    sample_hz = np.array([hz for hz, _ in sample_pts])
    sample_db = np.array([db for _, db in sample_pts])

    out = np.empty(len(OPTION_A_BINS_HR), dtype=np.float32)
    for i, target_hz in enumerate(OPTION_A_BINS_HR):
        # Linear interpolation for bins between sample points.
        out[i] = float(np.interp(target_hz, sample_hz, sample_db))
    return out


# ---------------------------------------------------------------------------
# F12/E50 — Per-author ensemble with router
# ---------------------------------------------------------------------------


class AuthorEnsembleModel:
    """Routes predictions to author-specific models when available (F12/E50).

    E40 showed 3 of 5 testable authors benefit from isolated models
    (aron7awol -0.09 dB, t1g8rsfan -0.44, kaelaria -0.42).  This is a
    *mixture-of-experts* approach where the gating function is the author
    identity — trivial routing but effective because author style is the
    single strongest feature (28% of importance in E29a).

    Dedicated models are trained on author-filtered subsets with the author
    column zeroed (the model IS that author, so the feature is redundant).
    The fallback model handles all other authors using the full catalogue.
    """

    # Authors that benefit from isolation (E40 results).
    DEDICATED_AUTHORS = {"aron7awol", "kaelaria", "t1g8rsfan"}

    def __init__(
        self,
        dedicated: dict[str, object],
        fallback: object,
        author_col_start: int = 0,
        n_author: int = N_AUTHOR,
    ) -> None:
        self.dedicated = dedicated
        self.fallback = fallback
        self._author_col_start = author_col_start
        self._n_author = n_author

    def predict(self, X: np.ndarray, authors: list[str] | None = None) -> np.ndarray:
        """Predict with routing.  *authors* must match rows of X."""
        if authors is None or not self.dedicated:
            return self.fallback.predict(X)

        Y = np.empty((len(X), self.fallback.predict(X[:1]).shape[1]))
        for i in range(len(X)):
            author = authors[i] if i < len(authors) else "unknown"
            if author in self.dedicated:
                Y[i] = self.dedicated[author].predict(X[i : i + 1])[0]
            else:
                Y[i] = self.fallback.predict(X[i : i + 1])[0]
        return Y


def train_author_ensemble(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    entries: list[dict],
    author_col_start: int | None = None,
) -> AuthorEnsembleModel:
    """Train an author-routed ensemble (F12/E50).

    Trains dedicated XGBoost models for each author in
    ``AuthorEnsembleModel.DEDICATED_AUTHORS``, plus a fallback model
    on the full catalogue.

    The author columns are zeroed in dedicated models — they don't need
    the author feature since they ARE that author.
    """
    if author_col_start is None:
        # Author one-hot starts after: audio(9) + year(1) + format(6) +
        # source(3) + studio(32) + mixer(22) + genre(10) + country(5) +
        # runtime(1) + rating(1) = 90
        author_col_start = N_AUDIO_FEATURES + (
            1 + N_AUDIO_FORMAT + N_SOURCE + N_STUDIO + N_MIXER
            + N_GENRE + N_COUNTRY + 1 + 1
        )

    # Full fallback model.
    log.info("training fallback (full catalogue) model...")
    fallback = train_xgboost(X_train, Y_train)

    dedicated: dict[str, object] = {}
    for author in AuthorEnsembleModel.DEDICATED_AUTHORS:
        mask = np.array([
            e.get("author", "").lower() == author for e in entries
        ])
        n_author_samples = int(mask.sum())
        if n_author_samples < 50:
            log.info("skipping %s: only %d samples", author, n_author_samples)
            continue

        X_author = X_train[mask].copy()
        Y_author = Y_train[mask]
        # Zero out author columns — the model IS this author.
        X_author[:, author_col_start : author_col_start + N_AUTHOR] = 0.0

        log.info("training dedicated model for %s (%d samples)...", author, n_author_samples)
        dedicated[author] = train_xgboost(X_author, Y_author)

    log.info("author ensemble: %d dedicated + 1 fallback", len(dedicated))
    return AuthorEnsembleModel(
        dedicated, fallback,
        author_col_start=author_col_start,
        n_author=N_AUTHOR,
    )


# ---------------------------------------------------------------------------
# I4 (E76) — Per-author dedicated late-fusion models with classifier routing
# ---------------------------------------------------------------------------


class AuthorEnsembleV2Model:
    """Late-fusion per-author ensemble routed by the I1b metadata classifier.

    Addresses the failure mode of F12/E50: that experiment trained
    per-author models but routed by the ground-truth author at
    inference.  At production time the author is unknown, so the
    approach didn't generalise.

    I4 fixes this by using the metadata classifier (same infrastructure
    as I1b) to *predict* which author's style the film belongs to, then
    routing to that author's dedicated late-fusion model.  For authors
    without enough training data (<MIN_SAMPLES), the fallback is the
    shared multi-author model.

    The dedicated models are full late-fusion (audio + metadata
    sub-models), trained on their author's subset with augmentation
    applied to the audio branch.  Author columns are zeroed in the
    training metadata since the model IS that author.

    Supports per-author alpha (like G8) — each dedicated model can use
    its own optimal alpha from PER_AUTHOR_ALPHA.
    """

    # Minimum training samples required to train a dedicated model.
    # Below this, the fallback handles the author.
    MIN_SAMPLES = 100

    def __init__(
        self,
        dedicated: dict[str, "LateFusionModel"],
        fallback: "LateFusionModel",
        classifier: object,
        n_audio: int = N_AUDIO_FEATURES,
    ) -> None:
        self.dedicated = dedicated
        self.fallback = fallback
        self.classifier = classifier
        self._n_audio = n_audio

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict by routing each row to the classifier-chosen author.

        For each row:
          1. Strip the author columns from the feature vector.
          2. Ask the classifier which author this film most resembles.
          3. If that author has a dedicated model, use it; else fallback.
        """
        X_no_author = strip_author_columns(X, self._n_audio)
        raw_probs = self.classifier.predict_proba(X_no_author)
        author_index_map = getattr(self.classifier, "_author_index_map", None)

        # Determine predicted author per row (as _AUTHOR_VOCAB index).
        if author_index_map is not None:
            contig_argmax = raw_probs.argmax(axis=1)
            predicted = np.array(
                [author_index_map[i] for i in contig_argmax], dtype=int,
            )
        else:
            predicted = raw_probs.argmax(axis=1)

        # First dry run to discover Y shape (use fallback on row 0).
        y_shape = self.fallback.predict(X[:1]).shape[1]
        Y = np.empty((len(X), y_shape), dtype=np.float32)

        for i in range(len(X)):
            author_idx = int(predicted[i])
            author = _AUTHOR_VOCAB[author_idx] if 0 <= author_idx < N_AUTHOR else "unknown"
            model = self.dedicated.get(author, self.fallback)
            Y[i] = model.predict(X[i : i + 1])[0]
        return Y


def train_author_ensemble_v2(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    entries: list[dict],
    alpha: float = 0.5,
    n_audio: int = N_AUDIO_FEATURES,
    augmentation: "AugmentationConfig | None" = None,
    dedicated_authors: list[str] | None = None,
) -> AuthorEnsembleV2Model:
    """Train an author-routed late-fusion ensemble (I4 / E76).

    Trains dedicated late-fusion models for each author in
    *dedicated_authors* that has at least ``MIN_SAMPLES`` training
    examples.  Uses the same augmentation and alpha settings as the
    shared fallback model.

    Also trains an author classifier (reusing the I1 infrastructure)
    which routes inference to the appropriate dedicated model.

    *dedicated_authors* defaults to the three authors with the largest
    WAV counts on the current 932-WAV cache: mobe1969 (305),
    aron7awol (304), kaelaria (181).
    """
    if dedicated_authors is None:
        dedicated_authors = ["mobe1969", "aron7awol", "kaelaria"]

    author_col_start_idx = n_audio + AUTHOR_COL_OFFSET_IN_METADATA

    # 1. Train the shared fallback model (handles authors without a
    #    dedicated model, e.g. remixmark, halcyon888, t1g8rsfan).
    log.info("I4: training shared fallback (late fusion α=%.1f)...", alpha)
    fallback = train_late_fusion(
        X_train, Y_train, alpha=alpha,
        n_audio=n_audio, augmentation=augmentation,
    )

    # 2. Train the author classifier on the full training set.
    log.info("I4: training author classifier for routing...")
    classifier = train_author_classifier(X_train, entries, n_audio=n_audio)

    # 3. Train dedicated late-fusion models per author.
    dedicated: dict[str, LateFusionModel] = {}
    for author in dedicated_authors:
        mask = np.array([
            str(e.get("author", "")).strip().lower() == author for e in entries
        ])
        n_samples = int(mask.sum())
        if n_samples < AuthorEnsembleV2Model.MIN_SAMPLES:
            log.info(
                "I4: skipping %s dedicated model: only %d samples "
                "(need %d)",
                author, n_samples, AuthorEnsembleV2Model.MIN_SAMPLES,
            )
            continue

        X_author = X_train[mask].copy()
        Y_author = Y_train[mask]
        # Zero the author one-hot: the dedicated model IS this author.
        X_author[:, author_col_start_idx : author_col_start_idx + N_AUTHOR] = 0.0

        log.info(
            "I4: training dedicated late-fusion for %s (%d samples, α=%.1f)...",
            author, n_samples, alpha,
        )
        dedicated[author] = train_late_fusion(
            X_author, Y_author, alpha=alpha,
            n_audio=n_audio, augmentation=augmentation,
        )

    log.info(
        "I4: ensemble ready — %d dedicated + 1 fallback + classifier router",
        len(dedicated),
    )
    return AuthorEnsembleV2Model(
        dedicated=dedicated,
        fallback=fallback,
        classifier=classifier,
        n_audio=n_audio,
    )


# ---------------------------------------------------------------------------
# G7 — Augmented model ensemble (average of multiple random-seed models)
# ---------------------------------------------------------------------------


class AugmentedEnsembleModel:
    """Ensemble of models trained with different augmentation random seeds (G7).

    Averaging predictions from models that saw different random noise creates
    a more robust estimator — a standard technique in Kaggle competitions
    and deep learning (dropout ensemble approximation).
    """

    def __init__(self, models: list[object]) -> None:
        self.models = models

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = [m.predict(X) for m in self.models]
        return np.mean(preds, axis=0)


def train_augmented_ensemble(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    augmentation: AugmentationConfig,
    n_seeds: int = 3,
    n_audio: int = N_AUDIO_FEATURES,
    late_fusion: bool = True,
    alpha: float = 0.7,
) -> AugmentedEnsembleModel:
    """Train an ensemble of augmented models with different random seeds."""
    models = []
    base_seed = augmentation.seed
    for i in range(n_seeds):
        aug_i = AugmentationConfig(
            gaussian_sigma_db=augmentation.gaussian_sigma_db,
            per_bin_uniform_db=augmentation.per_bin_uniform_db,
            smooth_prob=augmentation.smooth_prob,
            n_copies=augmentation.n_copies,
            seed=base_seed + i,
            per_bin_sigma=augmentation.per_bin_sigma,
        )
        if late_fusion:
            m = train_late_fusion(
                X_train, Y_train, alpha=alpha,
                n_audio=n_audio, augmentation=aug_i,
            )
        else:
            m = train_xgboost(X_train, Y_train, augmentation=aug_i, n_audio=n_audio)
        log.info("ensemble member %d/%d trained (seed=%d)", i + 1, n_seeds, aug_i.seed)
        models.append(m)
    return AugmentedEnsembleModel(models)


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
# E82 — Production training: 50:1 weighted hybrid plain XGBoost
# ---------------------------------------------------------------------------


def train_production_weighted_hybrid(
    real_samples: "list[tuple[dict, object]]",
    synth_entries: "list[dict]",
    tmdb_cache: dict,
    freqs_hz: np.ndarray,
    fs: int = 1000,
    real_weight: float = 50.0,
    config: AudioFeatureConfig = DEFAULT_AUDIO_CONFIG,
) -> "tuple[object, dict]":
    """Train the production 50:1 weighted hybrid plain XGBoost model (E82).

    Builds a combined training set from real-audio samples plus synthetic
    samples (for catalogue entries without real WAVs), constructs a
    ``sample_weight`` array that gives real samples ``real_weight`` weight
    and synthetic samples 1.0, and trains plain XGBoost — no late fusion,
    no augmentation, no classifier routing.

    This is the E82 champion configuration.  On the 219-title test split
    (1091-WAV cache) it matches real-only plain XGB at 1.99 dB mean while
    retaining synthetic coverage for the ~7k catalogue titles without
    real WAVs.

    Parameters
    ----------
    real_samples
        List of ``(catalogue_entry, curve_features)`` tuples.  The caller
        is responsible for WAV extraction (via the spike helpers). For
        E83 (T1.1), the ``CurveFeatures`` should have
        ``foundation_embedding`` populated; synthetic samples below
        automatically get a zero-vector fallback.
    synth_entries
        List of catalogue entries to synthesise.  Typically every
        trainable catalogue entry whose tmdb_id does NOT appear in
        ``real_samples``, so there's no duplication.
    tmdb_cache
        TMDb metadata cache (from ``load_cache()``) for metadata enrichment.
    freqs_hz
        Frequency grid used for synthetic feature generation (e.g.
        ``DEFAULT_GRID``).
    fs
        Sample rate for the filter chain evaluator (default 1000 Hz).
    real_weight
        Per-sample weight for real samples vs. 1.0 for synthetic samples.
        E82 found 50:1 ties real-only on in-distribution test titles.
    config
        Audio feature configuration. Default (E82): 102-dim plain
        features. Pass ``AudioFeatureConfig(foundation_model=
        "whisper-tiny")`` for E83 foundation-model features.

    Returns
    -------
    (model, metadata)
        ``model``: a fitted ``XGBRegressor`` (plain XGBoost, no wrappers)
        suitable for ``save_model()`` and ``TrainedModelAdvisor.load()``.
        ``metadata``: a dict with provenance fields — ``n_real``,
        ``n_synth``, ``real_weight``, ``trained_at`` (unix timestamp),
        ``wav_cache_mtime`` (max WAV mtime at train time or 0.0 if
        unavailable), ``xgb_params`` (hyperparameter snapshot),
        ``feature_config`` (the AudioFeatureConfig.label).
    """
    import time as _time
    from model.auto_beq import evaluate_filter_chain
    from model.auto_beq_advisor import extract_curve_features
    from model.auto_beq_metadata import enrich_media_metadata

    # --- Build the real-sample training rows (X_real, Y_real). ---
    X_real_list: list[np.ndarray] = []
    Y_real_list: list[np.ndarray] = []
    real_tmdb_ids: set[str] = set()
    for entry, features in real_samples:
        if not entry.get("filters"):
            continue
        metadata = enrich_media_metadata(entry, tmdb_cache)
        X_real_list.append(build_feature_vector(features, metadata, config=config))
        Y_real_list.append(catalogue_entry_to_labels(entry))
        tid = str(entry.get("theMovieDB", "")).strip()
        if tid:
            real_tmdb_ids.add(tid)

    if not X_real_list:
        raise ValueError(
            "train_production_weighted_hybrid: no usable real samples "
            "(every provided entry had empty filters or invalid features)",
        )

    X_real = np.array(X_real_list, dtype=np.float32)
    Y_real = np.array(Y_real_list, dtype=np.float32)

    # --- Build synthetic rows for catalogue entries without real WAVs. ---
    X_synth_list: list[np.ndarray] = []
    Y_synth_list: list[np.ndarray] = []
    anchor_idx = int(np.argmin(np.abs(freqs_hz - 80.0)))
    for entry in synth_entries:
        if not entry.get("filters"):
            continue
        tid = str(entry.get("theMovieDB", "")).strip()
        if tid and tid in real_tmdb_ids:
            continue  # already covered by a real sample
        try:
            correction = evaluate_filter_chain(entry["filters"], freqs_hz, fs=fs)
            rolloff = -correction
            rolloff = rolloff - rolloff[anchor_idx]
            features = extract_curve_features(rolloff, freqs_hz)
        except Exception as exc:
            log.debug("skipping synthetic entry %s: %s", entry.get("title"), exc)
            continue
        metadata = enrich_media_metadata(entry, tmdb_cache)
        X_synth_list.append(build_feature_vector(features, metadata, config=config))
        Y_synth_list.append(catalogue_entry_to_labels(entry))

    if X_synth_list:
        X_synth = np.array(X_synth_list, dtype=np.float32)
        Y_synth = np.array(Y_synth_list, dtype=np.float32)
        X_combined = np.vstack([X_real, X_synth])
        Y_combined = np.vstack([Y_real, Y_synth])
    else:
        X_synth = np.empty((0, X_real.shape[1]), dtype=np.float32)
        Y_synth = np.empty((0, Y_real.shape[1]), dtype=np.float32)
        X_combined = X_real
        Y_combined = Y_real

    n_real = len(X_real)
    n_synth = len(X_synth)

    # --- Build the sample_weight array. ---
    sample_weight = np.concatenate([
        np.full(n_real, float(real_weight), dtype=np.float32),
        np.ones(n_synth, dtype=np.float32),
    ])

    log.info(
        "E82 production training: %d real + %d synth (ratio %.0f:1)",
        n_real, n_synth, real_weight,
    )

    # --- Train plain XGBoost (no late fusion, no augmentation). ---
    model = train_xgboost(X_combined, Y_combined, sample_weight=sample_weight)

    # --- Provenance metadata for the sidecar file. ---
    metadata = {
        "n_real": n_real,
        "n_synth": n_synth,
        "real_weight": float(real_weight),
        "trained_at": int(_time.time()),
        "xgb_params": {
            "n_estimators": 400,
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_jobs": 1,
        },
        "feature_config": config.label,
        "n_features": int(config.n_features),
        "foundation_model": config.foundation_model,
    }
    return model, metadata


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
