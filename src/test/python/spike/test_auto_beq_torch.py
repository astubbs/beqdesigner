"""Unit tests for E85 / T1.2 — differentiable DSP components.

Focus:
- ``BiquadResponseLayer`` must produce the SAME log-magnitude response
  as the existing numpy/scipy ``evaluate_filter_chain`` to within
  floating-point slack. This is the critical correctness gate — if the
  differentiable layer drifts from the production path, any training
  would optimise against the wrong target.
- ``FilterChainPredictor`` output shapes + parameter range clamping.
- Save / load round-trip of a predictor.

All tests are pure-function (no training, no file I/O beyond a tmp_path
in one test), so they run in the fast unit suite. Import-guarded on
``torch`` availability: tests skip cleanly in a torch-free environment.
"""
from __future__ import annotations

import pytest
import numpy as np


# ---------------------------------------------------------------------------
# Module-level torch gate
# ---------------------------------------------------------------------------


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _torch_available(),
    reason="torch not installed — install via `poetry run pip install torch`",
)
# NOTE: These tests must run in a SEPARATE pytest invocation from Qt tests.
# torch + PyQt6 segfault when loaded in the same process on macOS (MPS conflict).
# bin/beq-designer dev test handles this automatically with two passes.


# ---------------------------------------------------------------------------
# BiquadResponseLayer ↔ evaluate_filter_chain parity
# ---------------------------------------------------------------------------


def _make_grid() -> np.ndarray:
    """Log-spaced 5-80 Hz grid matching DEFAULT_GRID's band of interest."""
    from model.auto_beq import DEFAULT_GRID
    return DEFAULT_GRID


def test_biquad_response_matches_numpy_single_shelf():
    """A single LowShelf biquad: torch layer vs scipy freqz must agree within 0.01 dB."""
    import torch
    from model.auto_beq import evaluate_filter_chain
    from model.auto_beq_torch import BiquadResponseLayer, NUM_SLOTS

    freqs = _make_grid()
    filters = [{"type": "LowShelf", "freq": 20.0, "gain": 5.0, "q": 0.9}]
    numpy_response = evaluate_filter_chain(filters, freqs, fs=1000)

    slot_params = torch.zeros(1, NUM_SLOTS, 4, dtype=torch.float64)
    slot_params[0, 0, 0] = 20.0
    slot_params[0, 0, 1] = 5.0
    slot_params[0, 0, 2] = 0.9
    slot_params[0, 0, 3] = 1.0  # enabled

    layer = BiquadResponseLayer(eval_freqs_hz=freqs, fs=1000)
    torch_response = layer(slot_params)[0].numpy()

    # Band-limit comparison to 5-80 Hz — outside this band the scipy
    # reference is 0 (no filter above 80 Hz) but the differentiable
    # layer still evaluates the biquad there. Both paths match inside
    # the band, which is what matters for training.
    band_mask = (freqs >= 5.0) & (freqs <= 80.0)
    max_diff = float(np.max(np.abs(torch_response[band_mask] - numpy_response[band_mask])))
    assert max_diff < 0.01, (
        f"biquad response drift: max |torch - numpy| = {max_diff:.4f} dB "
        f"at {len(freqs[band_mask])} freq bins in 5-80 Hz"
    )


def test_biquad_response_matches_numpy_three_slot_chain():
    """A 3-slot LowShelf chain: sum in dB equals product-of-magnitudes in numpy."""
    import torch
    from model.auto_beq import evaluate_filter_chain
    from model.auto_beq_torch import BiquadResponseLayer, NUM_SLOTS

    freqs = _make_grid()
    filters = [
        {"type": "LowShelf", "freq": 10.0, "gain": 3.5, "q": 0.8},
        {"type": "LowShelf", "freq": 20.0, "gain": 4.0, "q": 1.0},
        {"type": "LowShelf", "freq": 30.0, "gain": 2.0, "q": 1.2},
    ]
    numpy_response = evaluate_filter_chain(filters, freqs, fs=1000)

    slot_params = torch.zeros(1, NUM_SLOTS, 4, dtype=torch.float64)
    for i, f in enumerate(filters):
        slot_params[0, i, 0] = f["freq"]
        slot_params[0, i, 1] = f["gain"]
        slot_params[0, i, 2] = f["q"]
        slot_params[0, i, 3] = 1.0

    layer = BiquadResponseLayer(eval_freqs_hz=freqs, fs=1000)
    torch_response = layer(slot_params)[0].numpy()

    band_mask = (freqs >= 5.0) & (freqs <= 80.0)
    max_diff = float(np.max(np.abs(torch_response[band_mask] - numpy_response[band_mask])))
    assert max_diff < 0.01, (
        f"3-slot chain response drift: max |torch - numpy| = {max_diff:.4f} dB"
    )


def test_biquad_response_disabled_slots_contribute_zero_db():
    """Disabled slots must contribute 0 dB to the chain (unity magnitude)."""
    import torch
    from model.auto_beq_torch import BiquadResponseLayer, NUM_SLOTS

    freqs = _make_grid()

    # A chain with ONE active slot.
    active = torch.zeros(1, NUM_SLOTS, 4, dtype=torch.float64)
    active[0, 0, 0] = 20.0
    active[0, 0, 1] = 5.0
    active[0, 0, 2] = 0.9
    active[0, 0, 3] = 1.0

    # Same chain + 2 disabled garbage slots. Should match.
    with_disabled = active.clone()
    with_disabled[0, 1, 0] = 15.0   # garbage freq
    with_disabled[0, 1, 1] = 99.0   # out-of-range gain
    with_disabled[0, 1, 2] = 0.5
    with_disabled[0, 1, 3] = 0.0    # but disabled
    with_disabled[0, 2, 0] = 40.0
    with_disabled[0, 2, 1] = -8.0
    with_disabled[0, 2, 2] = 2.0
    with_disabled[0, 2, 3] = 0.0    # disabled

    layer = BiquadResponseLayer(eval_freqs_hz=freqs, fs=1000)
    r_active = layer(active).numpy()
    r_with_disabled = layer(with_disabled).numpy()
    max_diff = float(np.max(np.abs(r_active - r_with_disabled)))
    assert max_diff < 1e-6, (
        f"disabled slot leaked {max_diff} dB into the chain response"
    )


def test_biquad_response_gradients_flow():
    """Autograd through the whole layer — regression check for the real gate."""
    import torch
    from model.auto_beq_torch import BiquadResponseLayer, NUM_SLOTS

    freqs = _make_grid()
    layer = BiquadResponseLayer(eval_freqs_hz=freqs, fs=1000)

    # Requires-grad params mimicking the predictor output.
    slot_params = torch.zeros(2, NUM_SLOTS, 4, dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        slot_params[:, 0, 0] = 20.0
        slot_params[:, 0, 1] = 5.0
        slot_params[:, 0, 2] = 0.9
        slot_params[:, 0, 3] = 1.0

    response = layer(slot_params)
    loss = (response ** 2).sum()
    loss.backward()

    assert slot_params.grad is not None, "no grad produced"
    # At least the active slot's params should have non-zero grad.
    nonzero = float((slot_params.grad[:, 0, :].abs().sum()))
    assert nonzero > 0, (
        f"active slot grads are all zero: {slot_params.grad[:, 0, :]}"
    )


# ---------------------------------------------------------------------------
# FilterChainPredictor: shape + range clamping
# ---------------------------------------------------------------------------


def test_filter_chain_predictor_output_shape_and_ranges():
    """Predictor outputs (batch, 6, 4) with every column in its valid physical range."""
    import torch
    from model.auto_beq_torch import (
        FREQ_MAX_HZ, FREQ_MIN_HZ, FilterChainPredictor,
        GAIN_MAX_DB, GAIN_MIN_DB, NUM_SLOTS, Q_MAX, Q_MIN,
    )

    predictor = FilterChainPredictor(n_features=102)
    predictor.eval()
    # Batch of 4 zero vectors + random vectors — full range of raw inputs.
    x = torch.randn(8, 102) * 3.0
    with torch.no_grad():
        out = predictor(x)
    assert out.shape == (8, NUM_SLOTS, 4)

    freq = out[..., 0]
    gain = out[..., 1]
    q = out[..., 2]
    enabled = out[..., 3]

    assert float(freq.min()) >= FREQ_MIN_HZ - 1e-4
    assert float(freq.max()) <= FREQ_MAX_HZ + 1e-4
    assert float(gain.min()) >= GAIN_MIN_DB - 1e-4
    assert float(gain.max()) <= GAIN_MAX_DB + 1e-4
    assert float(q.min()) >= Q_MIN - 1e-4
    assert float(q.max()) <= Q_MAX + 1e-4
    assert float(enabled.min()) >= 0.0 - 1e-4
    assert float(enabled.max()) <= 1.0 + 1e-4


def test_filter_chain_predictor_predict_filters_roundtrip():
    """predict_filters() returns catalogue-schema dicts with clamped values."""
    from model.auto_beq_torch import FilterChainPredictor

    predictor = FilterChainPredictor(n_features=102)
    predictor.eval()
    x = np.random.default_rng(42).standard_normal(102).astype(np.float32)
    filters = predictor.predict_filters(x)
    # May be zero filters if everything's disabled — but any returned
    # dict must be a valid LowShelf with params in range.
    for f in filters:
        assert f["type"] == "LowShelf"
        assert 5.0 <= f["freq"] <= 80.0
        assert -15.0 <= f["gain"] <= 15.0
        assert 0.3 <= f["q"] <= 4.0


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# E86 — Multi-type topology (HighShelf + PeakingEQ parity)
# ---------------------------------------------------------------------------


def test_high_shelf_matches_numpy():
    """HighShelf biquad: torch layer vs scipy freqz within 0.01 dB."""
    import torch
    from model.auto_beq import evaluate_filter_chain
    from model.auto_beq_torch import BiquadResponseLayer, NUM_SLOTS, PARAMS_PER_SLOT_V2

    freqs = _make_grid()
    filters = [{"type": "HighShelf", "freq": 40.0, "gain": 4.0, "q": 1.0}]
    numpy_response = evaluate_filter_chain(filters, freqs, fs=1000)

    slot_params = torch.zeros(1, NUM_SLOTS, PARAMS_PER_SLOT_V2, dtype=torch.float64)
    slot_params[0, 0, 0] = 40.0    # freq
    slot_params[0, 0, 1] = 4.0     # gain
    slot_params[0, 0, 2] = 1.0     # q
    slot_params[0, 0, 3] = 1.0     # enabled
    slot_params[0, 0, 4] = -10.0   # LS logit (low)
    slot_params[0, 0, 5] = 10.0    # HS logit (high → picks HighShelf)
    slot_params[0, 0, 6] = -10.0   # PEQ logit (low)

    layer = BiquadResponseLayer(eval_freqs_hz=freqs, fs=1000)
    torch_response = layer(slot_params)[0].detach().numpy()

    band_mask = (freqs >= 5.0) & (freqs <= 80.0)
    max_diff = float(np.max(np.abs(torch_response[band_mask] - numpy_response[band_mask])))
    assert max_diff < 0.01, f"HighShelf drift: {max_diff:.4f} dB"


def test_peaking_eq_matches_numpy():
    """PeakingEQ biquad: torch layer vs scipy freqz within 0.01 dB."""
    import torch
    from model.auto_beq import evaluate_filter_chain
    from model.auto_beq_torch import BiquadResponseLayer, NUM_SLOTS, PARAMS_PER_SLOT_V2

    freqs = _make_grid()
    filters = [{"type": "PeakingEQ", "freq": 25.0, "gain": 3.0, "q": 1.5}]
    numpy_response = evaluate_filter_chain(filters, freqs, fs=1000)

    slot_params = torch.zeros(1, NUM_SLOTS, PARAMS_PER_SLOT_V2, dtype=torch.float64)
    slot_params[0, 0, 0] = 25.0
    slot_params[0, 0, 1] = 3.0
    slot_params[0, 0, 2] = 1.5
    slot_params[0, 0, 3] = 1.0
    slot_params[0, 0, 4] = -10.0   # LS low
    slot_params[0, 0, 5] = -10.0   # HS low
    slot_params[0, 0, 6] = 10.0    # PEQ high

    layer = BiquadResponseLayer(eval_freqs_hz=freqs, fs=1000)
    torch_response = layer(slot_params)[0].detach().numpy()

    band_mask = (freqs >= 5.0) & (freqs <= 80.0)
    max_diff = float(np.max(np.abs(torch_response[band_mask] - numpy_response[band_mask])))
    assert max_diff < 0.01, f"PeakingEQ drift: {max_diff:.4f} dB"


def test_multi_type_predictor_shape_and_types():
    """Multi-type predictor outputs 7-dim slot params + type selection works."""
    import torch
    from model.auto_beq_torch import FilterChainPredictor, NUM_SLOTS

    predictor = FilterChainPredictor(n_features=102, multi_type=True)
    predictor.eval()
    x = torch.randn(4, 102)
    with torch.no_grad():
        params, type_logits = predictor(x)
    assert params.shape == (4, NUM_SLOTS, 4)
    assert type_logits.shape == (4, NUM_SLOTS, 3)

    # predict_filters should return typed filter dicts.
    filters = predictor.predict_filters(x[0].numpy())
    for f in filters:
        assert f["type"] in ("LowShelf", "HighShelf", "PeakingEQ")


def test_multi_type_mixed_chain_matches_numpy():
    """A chain with one LS + one HS + one PEQ matches numpy evaluate_filter_chain."""
    import torch
    from model.auto_beq import evaluate_filter_chain
    from model.auto_beq_torch import BiquadResponseLayer, NUM_SLOTS, PARAMS_PER_SLOT_V2

    freqs = _make_grid()
    filters = [
        {"type": "LowShelf", "freq": 15.0, "gain": 5.0, "q": 0.8},
        {"type": "HighShelf", "freq": 50.0, "gain": 3.0, "q": 1.0},
        {"type": "PeakingEQ", "freq": 30.0, "gain": 2.0, "q": 2.0},
    ]
    numpy_response = evaluate_filter_chain(filters, freqs, fs=1000)

    slot_params = torch.zeros(1, NUM_SLOTS, PARAMS_PER_SLOT_V2, dtype=torch.float64)
    type_configs = [
        (15.0, 5.0, 0.8, [10, -10, -10]),   # LS
        (50.0, 3.0, 1.0, [-10, 10, -10]),    # HS
        (30.0, 2.0, 2.0, [-10, -10, 10]),    # PEQ
    ]
    for i, (f, g, q, tl) in enumerate(type_configs):
        slot_params[0, i, 0] = f
        slot_params[0, i, 1] = g
        slot_params[0, i, 2] = q
        slot_params[0, i, 3] = 1.0
        slot_params[0, i, 4] = tl[0]
        slot_params[0, i, 5] = tl[1]
        slot_params[0, i, 6] = tl[2]

    layer = BiquadResponseLayer(eval_freqs_hz=freqs, fs=1000)
    torch_response = layer(slot_params)[0].detach().numpy()

    band_mask = (freqs >= 5.0) & (freqs <= 80.0)
    max_diff = float(np.max(np.abs(torch_response[band_mask] - numpy_response[band_mask])))
    assert max_diff < 0.05, (
        f"mixed LS+HS+PEQ chain drift: {max_diff:.4f} dB "
        f"(soft blending introduces slight interpolation — 0.05 dB tolerance)"
    )


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------


def test_save_load_torch_predictor_roundtrip(tmp_path):
    """A saved predictor loads back with identical output for the same input."""
    import torch
    from model.auto_beq_torch import (
        FilterChainPredictor, load_torch_predictor, save_torch_predictor,
    )

    predictor = FilterChainPredictor(n_features=102, hidden_dim=64, n_hidden_layers=2)
    predictor.eval()
    x = torch.randn(3, 102)
    with torch.no_grad():
        out_before = predictor(x).numpy()

    path = str(tmp_path / "e85_predictor.pt")
    save_torch_predictor(predictor, path)
    loaded = load_torch_predictor(path)
    loaded.eval()
    with torch.no_grad():
        out_after = loaded(x).numpy()

    np.testing.assert_allclose(out_before, out_after, rtol=1e-5, atol=1e-6)
