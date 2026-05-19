"""Loss functions for HUPA-UCM glucose forecasting (Phase C).

Phase C.1 uses ``MultiHorizonMSE`` (vanilla mean squared error averaged over
batch and horizons). Phase C.2 will add asymmetric and zone-weighted
variants in this same module so the training loop stays loss-agnostic.

All losses operate in mg/dL: targets in the NPZ are stored unscaled and
model outputs are produced in mg/dL directly.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MultiHorizonMSE(nn.Module):
    """Vanilla MSE averaged across ``(batch, horizons)``.

    Parameters
    ----------
    reduction : {'mean', 'sum', 'per_horizon'}
        * 'mean' (default) — scalar mean over all elements; used for
          ``backward()``. Matches ``torch.nn.MSELoss(reduction='mean')``.
        * 'sum' — scalar sum, occasionally useful for batch-size weighting.
        * 'per_horizon' — ``(n_horizons,)`` tensor of per-horizon means for
          diagnostic logging; not intended for backprop.
    """

    VALID_REDUCTIONS = ("mean", "sum", "per_horizon")

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction not in self.VALID_REDUCTIONS:
            raise ValueError(
                f"reduction must be one of {self.VALID_REDUCTIONS}; got {reduction!r}"
            )
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if y_pred.shape != y_true.shape:
            raise ValueError(
                f"shape mismatch: y_pred={tuple(y_pred.shape)}, y_true={tuple(y_true.shape)}"
            )
        sq_err = (y_pred - y_true) ** 2
        if self.reduction == "mean":
            return sq_err.mean()
        if self.reduction == "sum":
            return sq_err.sum()
        return sq_err.mean(dim=0)


def per_horizon_mae(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Diagnostic helper: ``(n_horizons,)`` MAE for in-training logging."""
    return (y_pred - y_true).abs().mean(dim=0)
