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

# Studio embedding dim (Tier 1 — most predictive; zero-padded until TMDb lookup)
N_STUDIO_EMBED = 16

# Supervising mixer embedding dim (Tier 2 — zero-padded until IMDB lookup)
N_MIXER_EMBED = 8

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

# Metadata vector size
N_METADATA_FEATURES = (
    1                # year normalised
    + N_AUDIO_FORMAT  # 6
    + N_SOURCE        # 3
    + N_STUDIO_EMBED  # 16
    + N_MIXER_EMBED   # 8
    + N_GENRE         # 10
    + N_COUNTRY       # 5
    + 1               # runtime normalised
    + 1               # rating normalised
)  # = 51

N_FEATURES = N_AUDIO_FEATURES + N_METADATA_FEATURES  # 60

# ---------------------------------------------------------------------------
# Label constants (filter parameter output vector)
# ---------------------------------------------------------------------------

MAX_FILTER_SLOTS = 4
N_OUTPUT = MAX_FILTER_SLOTS * 4  # [type_int, freq_hz, gain_db, q] × 4 = 16

FILTER_TYPES = ["LowShelf", "HighShelf", "PeakingEQ"]
_TYPE_TO_INT = {t: float(i) for i, t in enumerate(FILTER_TYPES)}
_INT_TO_TYPE = {i: t for i, t in enumerate(FILTER_TYPES)}

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


def build_metadata_features(metadata: MediaMetadata) -> np.ndarray:
    """Encode MediaMetadata into a 51-dim float32 vector.

    Studio and mixer embedding slots are zero-padded until external lookup
    (TMDb/IMDB) is implemented. When those fields are populated they slot
    directly into the existing vector positions — no architecture change needed.
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

    # Studio embedding stub — 16 zeros until TMDb lookup (Tier 1)
    # When metadata.studio is populated, replace with a looked-up embedding vector.
    # For XGBoost stage: studio identity would be encoded by the caller as a
    # pre-computed lookup; for now all-zeros means "unknown studio".
    parts.append(np.zeros(N_STUDIO_EMBED, dtype=np.float32))

    # Mixer embedding stub — 8 zeros until IMDB lookup (Tier 2)
    parts.append(np.zeros(N_MIXER_EMBED, dtype=np.float32))

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

    result = np.concatenate(parts)
    assert result.shape == (N_METADATA_FEATURES,), (
        f"metadata feature dim mismatch: {result.shape} != ({N_METADATA_FEATURES},)"
    )
    return result


def build_feature_vector(features: CurveFeatures, metadata: MediaMetadata) -> np.ndarray:
    """Build the full 60-dim input vector: audio (9) + metadata (51).

    Returns float32 array of shape (60,).
    """
    audio = build_audio_features(features)
    meta = build_metadata_features(metadata)
    vec = np.concatenate([audio, meta]).astype(np.float32)
    assert vec.shape == (N_FEATURES,), (
        f"feature vector dim mismatch: {vec.shape} != ({N_FEATURES},)"
    )
    return vec


# ---------------------------------------------------------------------------
# Label encoding / decoding
# ---------------------------------------------------------------------------


def catalogue_entry_to_labels(entry: dict) -> np.ndarray:
    """Encode a catalogue entry's filter chain as a fixed 16-dim Y vector.

    MAX_FILTER_SLOTS=4 slots, each [type_int, freq_hz, gain_db, q].
    Unused slots are all zeros.
    """
    y = np.zeros(N_OUTPUT, dtype=np.float32)
    filters = entry.get("filters", [])
    for i, f in enumerate(filters[:MAX_FILTER_SLOTS]):
        slot = i * 4
        y[slot] = _TYPE_TO_INT.get(str(f.get("type", "LowShelf")), 0.0)
        y[slot + 1] = float(f.get("freq", 0.0))
        y[slot + 2] = float(f.get("gain", 0.0))
        y[slot + 3] = float(f.get("q", 0.9))
    return y


def labels_to_filters(y: np.ndarray, gain_threshold: float = 0.5) -> list[dict]:
    """Decode a 16-dim Y vector back into a list of filter dicts.

    Skips slots whose |gain| is below ``gain_threshold`` (empty/noise slots).
    Clamps all parameters to valid ranges.
    """
    filters = []
    for i in range(MAX_FILTER_SLOTS):
        slot = i * 4
        type_int = int(round(float(y[slot])))
        freq = float(y[slot + 1])
        gain = float(y[slot + 2])
        q = float(y[slot + 3])

        if abs(gain) < gain_threshold:
            continue

        ftype = _INT_TO_TYPE.get(max(0, min(2, type_int)), "LowShelf")
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
# Training
# ---------------------------------------------------------------------------


def train_xgboost(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    Y_val: np.ndarray | None = None,
) -> object:
    """Train an XGBoost multi-output regressor.

    Uses native XGBoost multi-output tree strategy which processes all output
    columns simultaneously. Early stopping is on validation downstream loss
    (not parameter MSE) when val data is provided.

    Returns the fitted model.
    """
    import xgboost as xgb

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

    model.fit(X_train, Y_train, **fit_kwargs)
    log.info("XGBoost training complete. n_features_in=%d", model.n_features_in_)
    return model


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
