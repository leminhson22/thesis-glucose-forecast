"""Loss functions for the six-model HUPA-UCM thesis pipeline."""
from __future__ import annotations

import torch
import torch.nn as nn

try:
    from . import config as C
except ImportError:
    import config as C  # type: ignore[no-redef]


_HYPO_THR = float(C.GLUCOSE_HYPO_THRESHOLD)
_HYPER_THR = float(C.GLUCOSE_HYPER_THRESHOLD)


class MultiHorizonMSE(nn.Module):
    """Vanilla MSE averaged across batch and forecast horizons."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction not in {"mean", "none"}:
            raise ValueError("reduction must be 'mean' or 'none'")
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        loss = (y_pred - y_true).pow(2)
        if self.reduction == "mean":
            return loss.mean()
        return loss


class ZoneWeightedMSE(nn.Module):
    """MSE with glycaemic-zone, horizon, and hypo-under-detection weights.

    This is used by the proposed Hybrid CNN-GRU model. Targets and predictions
    are in mg/dL. Zones are derived from the true target value:
    hypo ``<70``, time-in-range ``70-180``, and hyper ``>180``.
    """

    def __init__(
        self,
        w_hypo: float = 2.0,
        w_tir: float = 1.0,
        w_hyper: float = 1.5,
        horizon_weights: tuple[float, ...] = (1.0, 1.0, 1.0),
        hypo_under_detect_penalty: float = 1.0,
        reduction: str = "mean",
        hypo_thr: float = _HYPO_THR,
        hyper_thr: float = _HYPER_THR,
    ):
        super().__init__()
        if reduction not in {"mean", "none"}:
            raise ValueError("reduction must be 'mean' or 'none'")
        if any(w <= 0 for w in (w_hypo, w_tir, w_hyper, hypo_under_detect_penalty)):
            raise ValueError("all weights must be positive")
        self.w_hypo = float(w_hypo)
        self.w_tir = float(w_tir)
        self.w_hyper = float(w_hyper)
        self.hypo_under_detect_penalty = float(hypo_under_detect_penalty)
        self.hypo_thr = float(hypo_thr)
        self.hyper_thr = float(hyper_thr)
        self.reduction = reduction
        self.register_buffer(
            "horizon_weights",
            torch.tensor(tuple(float(w) for w in horizon_weights), dtype=torch.float32),
        )

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if y_pred.shape != y_true.shape:
            raise ValueError(
                f"shape mismatch: y_pred={tuple(y_pred.shape)}, y_true={tuple(y_true.shape)}"
            )

        weights = torch.full_like(y_true, self.w_tir)
        weights = torch.where(
            y_true < self.hypo_thr,
            torch.as_tensor(self.w_hypo, device=y_true.device, dtype=y_true.dtype),
            weights,
        )
        weights = torch.where(
            y_true > self.hyper_thr,
            torch.as_tensor(self.w_hyper, device=y_true.device, dtype=y_true.dtype),
            weights,
        )

        horizon_w = self.horizon_weights.to(device=y_true.device, dtype=y_true.dtype)
        if horizon_w.numel() != y_true.shape[1]:
            raise ValueError(
                f"horizon_weights length {horizon_w.numel()} does not match y_true.shape[1]={y_true.shape[1]}"
            )
        weights = weights * horizon_w.unsqueeze(0)

        if self.hypo_under_detect_penalty != 1.0:
            under_detect = (y_true < self.hypo_thr) & (y_pred > y_true)
            weights = torch.where(under_detect, weights * self.hypo_under_detect_penalty, weights)

        loss = weights * (y_pred - y_true).pow(2)
        if self.reduction == "mean":
            return loss.mean()
        return loss

    def extra_repr(self) -> str:
        hw = tuple(float(x) for x in self.horizon_weights.detach().cpu().tolist())
        return (
            f"w_hypo={self.w_hypo}, w_tir={self.w_tir}, w_hyper={self.w_hyper}, "
            f"horizon_weights={hw}, "
            f"hypo_under_detect_penalty={self.hypo_under_detect_penalty}, "
            f"thr=({self.hypo_thr}, {self.hyper_thr})"
        )


def per_horizon_mae(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Diagnostic helper: ``(n_horizons,)`` MAE for in-training logging."""
    return (y_pred - y_true).abs().mean(dim=0)
