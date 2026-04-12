"""E85 / T1.2 — Differentiable DSP for acoustic-loss training.

This module implements the paradigm shift from "MSE on filter params"
to "acoustic match between predicted and target frequency response":

- ``BiquadResponseLayer``: a fully differentiable PyTorch layer that
  evaluates a filter chain's log-magnitude response on a fixed frequency
  grid. Takes `(freq, gain, q, enabled)` per slot and returns the dB
  response — autograd flows back through the biquad coefficient trig.

- ``FilterChainPredictor``: a small MLP that consumes the same 102-dim
  (or 486-dim with foundation-model features) input the XGBoost
  pipeline uses and outputs `(6, 4)` slot params. Range-clamped via
  softplus/sigmoid/tanh so gradient descent can't escape the valid
  filter parameter space.

- ``train_e85_differentiable_dsp``: two-stage training. Stage 1 is
  MSE warm-start from an XGBoost teacher's predictions (cloning, to
  avoid cold-start bad local minima). Stage 2 is acoustic-loss
  fine-tuning: minimise ``||BiquadResponseLayer(predicted) -
  target_response||_2`` in dB space, weighted by the 5–80 Hz band.

Why fixed LowShelf topology (not 3-way soft type selection)? The
paradigm-shifts doc ranks this as Tier 1 with the caveat "start with
fixed 5-slot all-LowShelf topology (matches most of E82's
predictions) and only relax type selection if the fixed-topology
version works". LowShelves cover the vast majority of catalogue
filter chains for sub-bass extension; adding HighShelf + PeakingEQ
slots via softmax complicates the optimisation without a clear win
until the fixed-topology baseline is proven.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

log = logging.getLogger("auto_beq_torch")

# Default frequency grid for training and inference (1 kHz sample rate,
# log-spaced 5-80 Hz in fine steps). Imported from auto_beq to stay in
# sync with the rest of the pipeline.
_DEFAULT_FS = 1000


# ---------------------------------------------------------------------------
# Parameter range definitions (must match the production clamping in
# labels_to_filters so our predictions round-trip through the existing
# advisor protocol without surprises).
# ---------------------------------------------------------------------------

FREQ_MIN_HZ = 5.0
FREQ_MAX_HZ = 80.0
GAIN_MIN_DB = -15.0
GAIN_MAX_DB = 15.0
Q_MIN = 0.3
Q_MAX = 4.0

# Matches model.auto_beq_nn.MAX_FILTER_SLOTS; kept as a local constant so
# this module has no import-time dependency on auto_beq_nn.
NUM_SLOTS = 6

# Per-slot parameter count for this module's fixed topology:
#   (freq_raw, gain_raw, q_raw, enabled_raw)
# 4 raw scalars per slot → projected into bounded physical params via
# sigmoid/tanh. NO type-selection logits in v1 — fixed LowShelf.
PARAMS_PER_SLOT = 4


# ---------------------------------------------------------------------------
# BiquadResponseLayer — differentiable LowShelf chain log-mag response
# ---------------------------------------------------------------------------
#
# For a single LowShelf biquad with fs, f0, Q, gain:
#
#   A     = 10 ** (gain/40)
#   w0    = 2*pi*f0/fs
#   cos_w = cos(w0)
#   alpha = sin(w0) / (2*Q)
#
#   b0 =   A*( (A+1) - (A-1)*cos_w + 2*sqrt(A)*alpha )
#   b1 = 2*A*( (A-1) - (A+1)*cos_w                 )
#   b2 =   A*( (A+1) - (A-1)*cos_w - 2*sqrt(A)*alpha )
#   a0 =        (A+1) + (A-1)*cos_w + 2*sqrt(A)*alpha
#   a1 =   -2*( (A-1) + (A+1)*cos_w                 )
#   a2 =        (A+1) + (A-1)*cos_w - 2*sqrt(A)*alpha
#
# (Normalise b,a by a0 so a0 becomes 1.)
#
# Log-magnitude at eval freq f (rad/s w = 2*pi*f/fs):
#
#   |H(e^jw)|^2 = (b0^2 + b1^2 + b2^2
#                  + 2*(b0*b1 + b1*b2)*cos(w)
#                  + 2*b0*b2*cos(2w))
#               / (a0^2 + a1^2 + a2^2
#                  + 2*(a0*a1 + a1*a2)*cos(w)
#                  + 2*a0*a2*cos(2w))
#
# This uses only real arithmetic, so PyTorch autograd works without
# needing complex tensor support. Chain response = sum in dB = product
# in linear magnitude.


def _low_shelf_log_mag_db(
    freq_hz,
    gain_db,
    q,
    enabled,
    eval_freqs_hz,
    fs: int,
):
    """Differentiable log-magnitude of a LowShelf biquad evaluated on a grid.

    All tensor inputs are broadcastable; the last dim of the output
    matches ``eval_freqs_hz``. ``enabled`` is a sigmoid-smoothed
    contribution weight (0=disabled, 1=enabled) — disabled filters
    contribute 0 dB to the chain (unity magnitude).
    """
    import torch

    # Shape guard: broadcast everything to (..., 1) for the freq dim.
    freq_hz = freq_hz.unsqueeze(-1)
    gain_db = gain_db.unsqueeze(-1)
    q = q.unsqueeze(-1)
    enabled = enabled.unsqueeze(-1)

    eps = 1e-8

    A = torch.pow(torch.tensor(10.0, dtype=freq_hz.dtype, device=freq_hz.device),
                  gain_db / 40.0)
    sqrtA = torch.sqrt(A + eps)

    w0 = 2.0 * math.pi * freq_hz / fs
    cos_w0 = torch.cos(w0)
    sin_w0 = torch.sin(w0)
    alpha = sin_w0 / (2.0 * (q + eps))

    two_sqrtA_alpha = 2.0 * sqrtA * alpha
    a_plus_1 = A + 1.0
    a_minus_1 = A - 1.0

    # Unnormalised coefficients.
    b0 = A * (a_plus_1 - a_minus_1 * cos_w0 + two_sqrtA_alpha)
    b1 = 2.0 * A * (a_minus_1 - a_plus_1 * cos_w0)
    b2 = A * (a_plus_1 - a_minus_1 * cos_w0 - two_sqrtA_alpha)
    a0 = a_plus_1 + a_minus_1 * cos_w0 + two_sqrtA_alpha
    a1 = -2.0 * (a_minus_1 + a_plus_1 * cos_w0)
    a2 = a_plus_1 + a_minus_1 * cos_w0 - two_sqrtA_alpha

    # Normalise so a0_norm = 1.
    inv_a0 = 1.0 / (a0 + eps)
    b0n = b0 * inv_a0
    b1n = b1 * inv_a0
    b2n = b2 * inv_a0
    a0n = torch.ones_like(a0)
    a1n = a1 * inv_a0
    a2n = a2 * inv_a0

    # Evaluate |H(e^jw)|^2 at eval frequencies using the real-arithmetic
    # form. w and 2w cosines are cheap.
    w = 2.0 * math.pi * torch.as_tensor(
        eval_freqs_hz, dtype=freq_hz.dtype, device=freq_hz.device,
    ) / fs
    cos_w = torch.cos(w)   # shape: (len(eval_freqs),)
    cos_2w = torch.cos(2.0 * w)

    num = (
        b0n * b0n + b1n * b1n + b2n * b2n
        + 2.0 * (b0n * b1n + b1n * b2n) * cos_w
        + 2.0 * b0n * b2n * cos_2w
    )
    den = (
        a0n * a0n + a1n * a1n + a2n * a2n
        + 2.0 * (a0n * a1n + a1n * a2n) * cos_w
        + 2.0 * a0n * a2n * cos_2w
    )

    mag_sq = num / (den + eps)
    mag_sq = torch.clamp(mag_sq, min=eps)
    log_mag_db = 10.0 * torch.log10(mag_sq)

    # enabled gate: 0 dB when disabled, full dB when enabled.
    return log_mag_db * enabled


class BiquadResponseLayer:
    """Differentiable log-magnitude response of a LowShelf filter chain.

    Not actually an ``nn.Module`` because it has no learnable state —
    it's a pure function over (predicted_params, eval_freqs_hz). Kept as
    a class for symmetry with the rest of the model-loading code and so
    future extensions (HighShelf / PeakingEQ soft mixture) can add
    state cleanly.

    The forward pass sums the per-slot log-mag responses in dB, which
    is equivalent to multiplying magnitudes in linear — the standard
    "chain = product of transfer functions" rule for cascaded biquads.
    """

    def __init__(self, eval_freqs_hz: np.ndarray, fs: int = _DEFAULT_FS):
        self.eval_freqs_hz = np.asarray(eval_freqs_hz, dtype=np.float64)
        self.fs = fs

    def __call__(self, slot_params):
        """Evaluate the chain at every eval_freq.

        Parameters
        ----------
        slot_params
            Tensor of shape ``(batch, NUM_SLOTS, 4)`` with columns
            ``(freq_hz, gain_db, q, enabled)``. All entries must
            already be in valid physical ranges (the
            ``FilterChainPredictor`` handles that via its output
            activations).

        Returns
        -------
        Tensor of shape ``(batch, len(eval_freqs_hz))`` containing the
        cascaded chain's log-magnitude response in dB.
        """
        import torch

        if slot_params.shape[-2:] != (NUM_SLOTS, 4):
            raise ValueError(
                f"expected (..., {NUM_SLOTS}, 4) slot_params, got {tuple(slot_params.shape)}",
            )

        freq = slot_params[..., 0]
        gain = slot_params[..., 1]
        q = slot_params[..., 2]
        enabled = slot_params[..., 3]

        per_slot_db = _low_shelf_log_mag_db(
            freq_hz=freq,
            gain_db=gain,
            q=q,
            enabled=enabled,
            eval_freqs_hz=self.eval_freqs_hz,
            fs=self.fs,
        )
        # per_slot_db: (batch, NUM_SLOTS, n_freqs) → sum across slot dim.
        return per_slot_db.sum(dim=-2)


# ---------------------------------------------------------------------------
# FilterChainPredictor — MLP that outputs well-formed slot params
# ---------------------------------------------------------------------------


def _maybe_import_nn():
    """Lazy import of torch.nn so this module can be imported without torch."""
    try:
        import torch
        import torch.nn as nn
        return torch, nn
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for E85 differentiable DSP. Install via "
            "`poetry run pip install torch`.",
        ) from exc


class FilterChainPredictor:
    """Feature vector → ``(batch, NUM_SLOTS, 4)`` filter chain params.

    Shared MLP trunk, then a per-slot head. Output activations map raw
    linear outputs to valid physical ranges:

    - ``freq``: ``sigmoid`` → scaled to ``[FREQ_MIN_HZ, FREQ_MAX_HZ]``
    - ``gain``: ``tanh`` → scaled to ``[GAIN_MIN_DB, GAIN_MAX_DB]``
    - ``q``:    ``sigmoid`` → scaled to ``[Q_MIN, Q_MAX]``
    - ``enabled``: ``sigmoid`` → ``[0, 1]`` (soft at train time, hard
      threshold at 0.5 for inference filter extraction)

    Factored as a free-standing class rather than ``nn.Module`` so we
    can construct it even in environments without torch (e.g. CI) —
    torch is imported lazily on first use. The underlying trunk lives
    on ``self._trunk`` which IS an ``nn.Module``.
    """

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 256,
        n_hidden_layers: int = 3,
        device: str = "cpu",
    ):
        torch, nn = _maybe_import_nn()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers
        self.device = device

        layers: list = []
        in_dim = n_features
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            in_dim = hidden_dim
        # Output: NUM_SLOTS * PARAMS_PER_SLOT raw values, reshaped downstream.
        layers.append(nn.Linear(in_dim, NUM_SLOTS * PARAMS_PER_SLOT))
        self._trunk = nn.Sequential(*layers).to(device)

    def parameters(self):
        return self._trunk.parameters()

    def train(self, mode: bool = True):
        self._trunk.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def to(self, device):
        self.device = device
        self._trunk.to(device)
        return self

    def state_dict(self):
        return self._trunk.state_dict()

    def load_state_dict(self, state):
        self._trunk.load_state_dict(state)

    def __call__(self, x):
        """Forward pass.

        ``x`` shape ``(batch, n_features)`` → output shape
        ``(batch, NUM_SLOTS, 4)`` with each column already in valid
        physical ranges.
        """
        import torch

        raw = self._trunk(x)  # (batch, NUM_SLOTS * PARAMS_PER_SLOT)
        raw = raw.view(-1, NUM_SLOTS, PARAMS_PER_SLOT)

        freq_raw = raw[..., 0]
        gain_raw = raw[..., 1]
        q_raw = raw[..., 2]
        enabled_raw = raw[..., 3]

        freq = FREQ_MIN_HZ + (FREQ_MAX_HZ - FREQ_MIN_HZ) * torch.sigmoid(freq_raw)
        gain = GAIN_MAX_DB * torch.tanh(gain_raw)  # symmetric ±15 dB
        q = Q_MIN + (Q_MAX - Q_MIN) * torch.sigmoid(q_raw)
        enabled = torch.sigmoid(enabled_raw)

        return torch.stack([freq, gain, q, enabled], dim=-1)

    def predict_filters(self, feature_vector: np.ndarray) -> list[dict]:
        """Convenience: 1-D feature vector → list of filter dicts (catalogue schema).

        Used by the advisor wrapper at inference time. Applies a hard
        threshold (enabled > 0.5) and returns LowShelf dicts matching
        the format ``catalogue_entry_to_labels`` expects.
        """
        import torch

        self._trunk.eval()
        with torch.no_grad():
            x = torch.as_tensor(
                feature_vector, dtype=torch.float32, device=self.device,
            ).unsqueeze(0)
            slot_params = self(x)  # (1, NUM_SLOTS, 4)
        slot_params = slot_params[0].cpu().numpy()

        filters: list[dict] = []
        for i in range(NUM_SLOTS):
            freq, gain, q, en = slot_params[i]
            if en < 0.5 or abs(gain) < 0.5:
                continue
            filters.append({
                "type": "LowShelf",
                "freq": float(np.clip(freq, FREQ_MIN_HZ, FREQ_MAX_HZ)),
                "gain": float(np.clip(gain, GAIN_MIN_DB, GAIN_MAX_DB)),
                "q": float(np.clip(q, Q_MIN, Q_MAX)),
            })
        return filters


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


@dataclass
class E85TrainingConfig:
    """Tunable knobs for ``train_e85_differentiable_dsp``."""
    n_epochs_warm: int = 10
    n_epochs_acoustic: int = 20
    batch_size: int = 64
    learning_rate_warm: float = 1e-3
    learning_rate_acoustic: float = 5e-4
    band_lo_hz: float = 5.0
    band_hi_hz: float = 80.0
    hidden_dim: int = 256
    n_hidden_layers: int = 3
    device: str = "cpu"


def _target_response_from_entry(entry: dict, eval_freqs_hz: np.ndarray, fs: int) -> np.ndarray:
    """The acoustic target for a catalogue entry = its filter chain's response."""
    from model.auto_beq import evaluate_filter_chain

    return evaluate_filter_chain(entry["filters"], eval_freqs_hz, fs=fs)


def _teacher_predicted_params(
    teacher_model, X: np.ndarray,
) -> np.ndarray:
    """Run the XGBoost teacher and decode its 36-dim output into per-slot params.

    Returns shape ``(n, NUM_SLOTS, 4)`` — (freq, gain, q, enabled) per slot.
    Used for the Stage 1 MSE warm-start target.
    """
    from model.auto_beq_nn import MAX_FILTER_SLOTS, N_PER_SLOT, labels_to_filters

    raw = teacher_model.predict(X)  # (n, 36)
    out = np.zeros((len(X), NUM_SLOTS, 4), dtype=np.float32)
    for i, y in enumerate(raw):
        filters = labels_to_filters(y)
        for slot_idx in range(min(NUM_SLOTS, len(filters))):
            f = filters[slot_idx]
            out[i, slot_idx, 0] = float(f.get("freq", 20.0))
            out[i, slot_idx, 1] = float(f.get("gain", 0.0))
            out[i, slot_idx, 2] = float(f.get("q", 0.9))
            out[i, slot_idx, 3] = 1.0  # enabled
        # Remaining slots stay zeroed with enabled=0
    return out


def train_e85_differentiable_dsp(
    X_train: np.ndarray,
    entries_train: list[dict],
    teacher_model,
    eval_freqs_hz: np.ndarray,
    fs: int = _DEFAULT_FS,
    config: "E85TrainingConfig | None" = None,
):
    """E85: two-stage training with MSE warm-start + acoustic fine-tune.

    Stage 1 (warm-start): MSE on filter-param space against the XGBoost
    teacher's predictions. This gets the predictor into a reasonable
    region of parameter space before we switch to the non-convex
    acoustic loss.

    Stage 2 (acoustic loss): band-masked MSE between the biquad chain
    response predicted by the model and the target response
    ``evaluate_filter_chain(entry.filters)``. Direct acoustic-match
    objective — this is the whole point of E85.

    Returns ``(predictor, training_stats_dict)``.
    """
    torch, nn = _maybe_import_nn()

    if config is None:
        config = E85TrainingConfig()

    predictor = FilterChainPredictor(
        n_features=X_train.shape[1],
        hidden_dim=config.hidden_dim,
        n_hidden_layers=config.n_hidden_layers,
        device=config.device,
    )

    # Band mask for Stage 2 acoustic loss — BEQ only cares about 5-80 Hz.
    band_mask_np = (eval_freqs_hz >= config.band_lo_hz) & (
        eval_freqs_hz <= config.band_hi_hz
    )
    band_mask = torch.tensor(band_mask_np, dtype=torch.float32, device=config.device)

    # Pre-compute target responses for the acoustic stage — one row per train entry.
    log.info(
        "E85 prep: building target responses for %d training entries…",
        len(entries_train),
    )
    target_resp_np = np.stack([
        _target_response_from_entry(e, eval_freqs_hz, fs)
        for e in entries_train
    ], axis=0).astype(np.float32)
    target_resp = torch.tensor(target_resp_np, device=config.device)

    X_t = torch.tensor(X_train.astype(np.float32), device=config.device)
    n = len(X_train)

    # Stage 1 warm-start target from the XGBoost teacher.
    log.info("E85 prep: cloning teacher's filter-param predictions (MSE warm-start target)…")
    teacher_params_np = _teacher_predicted_params(teacher_model, X_train)
    teacher_params = torch.tensor(teacher_params_np, device=config.device)

    biquad_layer = BiquadResponseLayer(eval_freqs_hz, fs=fs)

    import time as _time
    t0 = _time.time()

    # --- Stage 1: MSE warm-start ---
    log.info(
        "=== E85 stage 1: MSE warm-start (%d epochs, lr=%.0e) ===",
        config.n_epochs_warm, config.learning_rate_warm,
    )
    optim_warm = torch.optim.Adam(predictor.parameters(), lr=config.learning_rate_warm)
    predictor.train()
    for epoch in range(config.n_epochs_warm):
        perm = torch.randperm(n, device=config.device)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, config.batch_size):
            idx = perm[start:start + config.batch_size]
            xb = X_t[idx]
            yb = teacher_params[idx]
            pred = predictor(xb)  # (b, NUM_SLOTS, 4)
            loss = nn.functional.mse_loss(pred, yb)
            optim_warm.zero_grad()
            loss.backward()
            optim_warm.step()
            epoch_loss += float(loss.detach())
            n_batches += 1
        log.info(
            "  warm epoch %d/%d: mse_loss=%.4f",
            epoch + 1, config.n_epochs_warm, epoch_loss / max(1, n_batches),
        )

    # --- Stage 2: acoustic loss fine-tune ---
    log.info(
        "=== E85 stage 2: acoustic-loss fine-tune (%d epochs, lr=%.0e, band=%.0f-%.0f Hz) ===",
        config.n_epochs_acoustic, config.learning_rate_acoustic,
        config.band_lo_hz, config.band_hi_hz,
    )
    optim_ac = torch.optim.Adam(predictor.parameters(), lr=config.learning_rate_acoustic)
    predictor.train()
    for epoch in range(config.n_epochs_acoustic):
        perm = torch.randperm(n, device=config.device)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, config.batch_size):
            idx = perm[start:start + config.batch_size]
            xb = X_t[idx]
            target_b = target_resp[idx]
            pred_params = predictor(xb)
            pred_resp = biquad_layer(pred_params)
            diff = (pred_resp - target_b) * band_mask
            loss = (diff ** 2).sum(dim=-1).mean() / band_mask.sum()
            optim_ac.zero_grad()
            loss.backward()
            optim_ac.step()
            epoch_loss += float(loss.detach())
            n_batches += 1
        log.info(
            "  acoustic epoch %d/%d: band-masked mse_db2=%.4f",
            epoch + 1, config.n_epochs_acoustic, epoch_loss / max(1, n_batches),
        )

    stats = {
        "train_time_s": round(_time.time() - t0, 1),
        "n_epochs_warm": config.n_epochs_warm,
        "n_epochs_acoustic": config.n_epochs_acoustic,
        "n_train": n,
        "n_features": X_train.shape[1],
        "hidden_dim": config.hidden_dim,
        "n_hidden_layers": config.n_hidden_layers,
    }
    log.info(
        "=== E85 training done in %.1fs — predictor ready ===",
        stats["train_time_s"],
    )
    return predictor, stats


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def save_torch_predictor(predictor: FilterChainPredictor, path: str) -> None:
    """Save a FilterChainPredictor as a .pt file with its config.

    The saved file is a dict with ``state_dict`` + ``config`` so a
    matching predictor can be reconstructed on load.
    """
    torch, _nn = _maybe_import_nn()
    torch.save({
        "state_dict": predictor.state_dict(),
        "n_features": predictor.n_features,
        "hidden_dim": predictor.hidden_dim,
        "n_hidden_layers": predictor.n_hidden_layers,
    }, path)
    log.info("E85 predictor saved to %s", path)


def load_torch_predictor(path: str, device: str = "cpu") -> FilterChainPredictor:
    """Load a FilterChainPredictor previously saved by ``save_torch_predictor``."""
    torch, _nn = _maybe_import_nn()
    blob = torch.load(path, map_location=device, weights_only=False)
    predictor = FilterChainPredictor(
        n_features=int(blob["n_features"]),
        hidden_dim=int(blob["hidden_dim"]),
        n_hidden_layers=int(blob["n_hidden_layers"]),
        device=device,
    )
    predictor.load_state_dict(blob["state_dict"])
    predictor.eval()
    log.info("E85 predictor loaded from %s", path)
    return predictor
