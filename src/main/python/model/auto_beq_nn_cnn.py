"""CNN dual-branch Advisor for auto-BEQ — Experiment 23.

Two-branch architecture that naturally prevents the cross-feature
overfitting found in E18e:

    audio features (9 bins)
        → 1D Conv layers → audio embedding (64 dims)
    metadata features (81 dims)
        → Dense layer → metadata embedding (32 dims)
    audio_embed ⊕ meta_embed
        → 2 dense layers (96 → 48 → 16)
        → filter parameter output (16 dims)

Audio and metadata are processed through separate branches before merging,
so the model can't learn spurious cross-correlations between synthetic
audio features and metadata during training.

See docs/design/auto_beq.md for architecture rationale.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from model.auto_beq_advisor import Advice, CurveFeatures, MediaMetadata

log = logging.getLogger("auto_beq_nn_cnn")

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------


def _check_torch():
    if not _HAS_TORCH:
        raise ImportError("PyTorch is required for E23 CNN dual-branch. Install with: poetry add --group dev torch")


class DualBranchCNN(nn.Module):
    """Two-branch CNN: audio 1D conv + metadata dense, merged before output.

    Audio branch: 1D convolutions learn frequency-domain patterns from the
    9-bin percentile curve. Metadata branch: dense layers process the 81-dim
    encoded metadata. Both branches produce embeddings that are concatenated
    and passed through shared output layers.
    """

    def __init__(
        self,
        n_audio: int = 9,
        n_meta: int = 81,
        n_output: int = 16,
        audio_embed_dim: int = 64,
        meta_embed_dim: int = 32,
    ):
        _check_torch()
        super().__init__()
        self.n_audio = n_audio
        self.n_meta = n_meta

        # Audio branch: 1D conv over frequency bins.
        # Input: (batch, 1, n_audio) — single channel, 9 frequency bins.
        self.audio_branch = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # → (batch, 64, 1)
            nn.Flatten(),             # → (batch, 64)
            nn.Linear(64, audio_embed_dim),
            nn.ReLU(),
        )

        # Metadata branch: dense layers.
        self.meta_branch = nn.Sequential(
            nn.Linear(n_meta, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, meta_embed_dim),
            nn.ReLU(),
        )

        # Merged output layers.
        merged_dim = audio_embed_dim + meta_embed_dim
        self.output_head = nn.Sequential(
            nn.Linear(merged_dim, 48),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(48, n_output),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_audio = x[:, :self.n_audio].unsqueeze(1)  # (batch, 1, 9)
        x_meta = x[:, self.n_audio:]                  # (batch, 81)

        audio_embed = self.audio_branch(x_audio)
        meta_embed = self.meta_branch(x_meta)

        merged = torch.cat([audio_embed, meta_embed], dim=1)
        return self.output_head(merged)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_cnn(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    Y_val: np.ndarray | None = None,
    n_audio: int = 9,
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 20,
) -> DualBranchCNN:
    """Train the dual-branch CNN with early stopping on validation loss.

    Returns the trained model (on CPU).
    """
    _check_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("training CNN on %s, %d samples, %d epochs max", device, len(X_train), epochs)

    n_meta = X_train.shape[1] - n_audio
    n_output = Y_train.shape[1]
    model = DualBranchCNN(
        n_audio=n_audio, n_meta=n_meta, n_output=n_output,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    Y_t = torch.tensor(Y_train, dtype=torch.float32, device=device)

    has_val = X_val is not None and Y_val is not None
    if has_val:
        X_v = torch.tensor(X_val, dtype=torch.float32, device=device)
        Y_v = torch.tensor(Y_val, dtype=torch.float32, device=device)

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        # Mini-batch training.
        perm = torch.randperm(len(X_t), device=device)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, len(X_t), batch_size):
            idx = perm[i:i + batch_size]
            y_pred = model(X_t[idx])
            loss = loss_fn(y_pred, Y_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / n_batches

        # Validation.
        if has_val:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_v)
                val_loss = loss_fn(val_pred, Y_v).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if (epoch + 1) % 20 == 0:
                log.info("epoch %d: train=%.4f val=%.4f best=%.4f",
                         epoch + 1, avg_train_loss, val_loss, best_val_loss)

            if no_improve >= patience:
                log.info("early stopping at epoch %d (patience=%d)", epoch + 1, patience)
                break
        else:
            if (epoch + 1) % 20 == 0:
                log.info("epoch %d: train=%.4f", epoch + 1, avg_train_loss)

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.cpu().eval()
    log.info("CNN training complete. %d epochs, best_val=%.4f",
             epoch + 1, best_val_loss if has_val else avg_train_loss)
    return model


# ---------------------------------------------------------------------------
# Prediction wrapper (numpy interface for compatibility)
# ---------------------------------------------------------------------------


class CNNPredictor:
    """Wraps a trained DualBranchCNN with a numpy ``.predict(X)`` interface.

    Compatible with ``TrainedModelAdvisor`` and ``save_model/load_model``.
    """

    def __init__(self, model: DualBranchCNN) -> None:
        self.model = model

    def predict(self, X: np.ndarray) -> np.ndarray:
        _check_torch()
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            Y_t = self.model(X_t)
        return Y_t.numpy()


# ---------------------------------------------------------------------------
# CNNAdvisor — Advisor protocol implementation
# ---------------------------------------------------------------------------


class CNNAdvisor:
    """Advisor backed by a CNN dual-branch model (E23).

    Audio features are processed through 1D convolutions; metadata through
    dense layers. The two branches merge at the penultimate layer, naturally
    preventing the cross-feature overfitting seen with early fusion.
    """

    name = "cnn_dual_branch"

    def __init__(self, predictor: CNNPredictor) -> None:
        self._predictor = predictor

    @classmethod
    def load(cls, path: str) -> "CNNAdvisor":
        from model.auto_beq_nn import load_model
        return cls(load_model(path))

    def advise(self, metadata: MediaMetadata, features: CurveFeatures) -> Advice:
        from model.auto_beq_advisor import Advice, _clamp_advice
        from model.auto_beq_nn import build_feature_vector, labels_to_filters

        x = build_feature_vector(features, metadata)
        y_pred = self._predictor.predict(x.reshape(1, -1))[0]
        filters = labels_to_filters(y_pred)

        if not filters:
            return _clamp_advice(
                Advice(max_gain_db=10.0, reasoning="cnn_dual_branch: no filters predicted",
                       confidence=0.2, source="cnn_dual_branch"),
                source="cnn_dual_branch",
            )

        total_gain = sum(abs(f["gain"]) for f in filters)
        primary_knee = filters[0]["freq"]
        return _clamp_advice(
            Advice(
                max_gain_db=total_gain, knee_hz=primary_knee,
                filters=tuple(filters),
                reasoning=f"cnn_dual_branch: {len(filters)} filter(s) predicted",
                confidence=0.5, source="cnn_dual_branch",
            ),
            source="cnn_dual_branch",
        )
