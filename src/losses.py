"""Loss functions for HUPA-UCM glucose forecasting (Phase C).

Phase C.1 uses ``MultiHorizonMSE`` (vanilla mean squared error averaged over
batch and horizons). Phase C.2 adds ``ZoneWeightedMSE``, a single composite
loss that supports per-zone weights (hypo/tir/hyper), per-horizon weights,
and an asymmetric penalty for hypoglycaemia under-detection — the
clinically worst error mode per SKILL.md §5.3.

All losses operate in mg/dL: targets in the NPZ are stored unscaled and
model outputs are produced in mg/dL directly.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# Default zone thresholds (mg/dL) per CLAUDE.md and SKILL.md
_HYPO_THR = 70.0
_HYPER_THR = 180.0


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


class ZoneWeightedMSE(nn.Module):
    """Composite MSE with per-zone, per-horizon, and asymmetric weighting.

    Per-element weight is::

        weight = zone_weight(y_true) * horizon_weight(h) * asym_factor(y_true, y_pred)

    where:

    * ``zone_weight`` is one of ``w_hypo`` / ``w_tir`` / ``w_hyper`` selected
      by the zone of ``y_true`` (CLAUDE.md mg/dL thresholds).
    * ``horizon_weight`` is the per-horizon entry from ``horizon_weights``;
      defaults to all-ones. Use this knob to up-weight the 30 min horizon
      where Phase C.1 found the largest gap to Persistence in hypo zone.
    * ``asym_factor`` is ``hypo_under_detect_penalty`` when ``y_true`` is in
      hypo zone AND ``y_pred > y_true`` (the model failed to predict a hypo
      event — the most dangerous clinical failure mode). Default 1.0
      (symmetric).

    Loss = ``mean(weight * (y_pred - y_true)**2)`` with the standard
    elementwise mean across batch and horizons. The mean uses the weight
    sum as the denominator only if ``normalise_by_weight=True``; otherwise
    it divides by the element count (preferred default — keeps the loss
    magnitude comparable to ``MultiHorizonMSE`` for early-stopping
    diagnostics).

    Parameters
    ----------
    w_hypo, w_tir, w_hyper
        Per-zone base weights. Default (2.0, 1.0, 1.5) follows SKILL.md
        §5.3 ("asymmetric penalty term that increases loss for errors in
        the hypoglycemic range") and prioritises hypo over hyper.
    horizon_weights
        Optional sequence of length ``n_horizons``. ``None`` means uniform.
    hypo_under_detect_penalty
        Multiplier applied to the weight when ``y_true < hypo_thr`` AND
        ``y_pred > y_true``. Default 1.0 (off).
    hypo_thr, hyper_thr
        Zone thresholds in mg/dL. Defaults match CLAUDE.md.
    normalise_by_weight
        If True, divides by ``weight.sum()`` instead of element count. Keep
        False so the loss is comparable to ``MultiHorizonMSE``.
    """

    def __init__(
        self,
        w_hypo: float = 2.0,
        w_tir: float = 1.0,
        w_hyper: float = 1.5,
        horizon_weights: tuple[float, ...] | None = None,
        hypo_under_detect_penalty: float = 1.0,
        hypo_thr: float = _HYPO_THR,
        hyper_thr: float = _HYPER_THR,
        normalise_by_weight: bool = False,
    ):
        super().__init__()
        if w_hypo <= 0 or w_tir <= 0 or w_hyper <= 0:
            raise ValueError("zone weights must be positive")
        if hypo_under_detect_penalty <= 0:
            raise ValueError("hypo_under_detect_penalty must be positive")
        self.w_hypo = float(w_hypo)
        self.w_tir = float(w_tir)
        self.w_hyper = float(w_hyper)
        self.hypo_under_detect_penalty = float(hypo_under_detect_penalty)
        self.hypo_thr = float(hypo_thr)
        self.hyper_thr = float(hyper_thr)
        self.normalise_by_weight = bool(normalise_by_weight)
        if horizon_weights is None:
            self.register_buffer("horizon_weights", None, persistent=False)
        else:
            hw = torch.tensor(list(horizon_weights), dtype=torch.float32)
            if (hw <= 0).any():
                raise ValueError("horizon_weights must all be positive")
            self.register_buffer("horizon_weights", hw, persistent=False)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if y_pred.shape != y_true.shape:
            raise ValueError(
                f"shape mismatch: y_pred={tuple(y_pred.shape)}, y_true={tuple(y_true.shape)}"
            )
        zone_w = torch.full_like(y_true, self.w_tir)
        zone_w = torch.where(y_true < self.hypo_thr, torch.tensor(self.w_hypo, device=y_true.device, dtype=y_true.dtype), zone_w)
        zone_w = torch.where(y_true > self.hyper_thr, torch.tensor(self.w_hyper, device=y_true.device, dtype=y_true.dtype), zone_w)

        weight = zone_w
        if self.horizon_weights is not None:
            hw = self.horizon_weights.to(device=y_true.device, dtype=y_true.dtype)
            if hw.numel() != y_true.shape[-1]:
                raise ValueError(
                    f"horizon_weights has {hw.numel()} entries but y_true has "
                    f"{y_true.shape[-1]} horizons"
                )
            weight = weight * hw.unsqueeze(0)
        if self.hypo_under_detect_penalty != 1.0:
            under_detect = (y_true < self.hypo_thr) & (y_pred > y_true)
            weight = torch.where(
                under_detect,
                weight * self.hypo_under_detect_penalty,
                weight,
            )

        sq_err = (y_pred - y_true) ** 2
        weighted = weight * sq_err
        if self.normalise_by_weight:
            return weighted.sum() / weight.sum().clamp_min(1e-8)
        return weighted.mean()

    def extra_repr(self) -> str:
        hw = None if self.horizon_weights is None else self.horizon_weights.tolist()
        return (
            f"w_hypo={self.w_hypo}, w_tir={self.w_tir}, w_hyper={self.w_hyper}, "
            f"horizon_weights={hw}, "
            f"hypo_under_detect_penalty={self.hypo_under_detect_penalty}, "
            f"thr=({self.hypo_thr}, {self.hyper_thr}), "
            f"normalise_by_weight={self.normalise_by_weight}"
        )


def per_horizon_mae(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Diagnostic helper: ``(n_horizons,)`` MAE for in-training logging."""
    return (y_pred - y_true).abs().mean(dim=0)


class TrajectoryLoss(nn.Module):
    """Seq2seq trajectory loss combining zone-weighted MSE and rate-of-change MSE.

    Used by the Step 6 v3 ``seq2seq`` variant. The model outputs a full
    18-step trajectory ``y_pred: (B, 18)`` and the target is
    ``y_true: (B, 18)`` covering t+5 through t+90 in 5-minute increments.

    Loss = L_point + lambda_rate * L_rate, where

    * ``L_point`` is :class:`ZoneWeightedMSE` applied to all 18 trajectory
      points (with per-zone, per-step, and asymmetric hypoglycaemia
      weighting derived from the same logic as Phase C.2). Glycaemic
      zones are computed from ``y_true``.
    * ``L_rate`` is the MSE between the predicted and reference
      first-difference along the time axis: ``Δy_pred[k] = y_pred[k+1] -
      y_pred[k]``, similarly for ``y_true``. This is the direct
      supervisory signal for CG-EGA Rate-EGA agreement and is the
      structural fix for the §8.6 / §8.8 long-horizon EP regression.

    The per-step weighting follows the §8.5 design intent of up-weighting
    the 30-minute horizon, applied here as a smooth weight profile that
    is 1.5 at indices 0..5 (t+5..t+30) and 1.0 elsewhere.
    """

    def __init__(
        self,
        w_hypo: float = 2.0,
        w_tir: float = 1.0,
        w_hyper: float = 1.5,
        hypo_under_detect_penalty: float = 2.0,
        early_horizon_steps: int = 6,
        early_horizon_weight: float = 1.5,
        lambda_rate: float = 0.5,
        hypo_thr: float = _HYPO_THR,
        hyper_thr: float = _HYPER_THR,
    ):
        super().__init__()
        if w_hypo <= 0 or w_tir <= 0 or w_hyper <= 0:
            raise ValueError("zone weights must be positive")
        if hypo_under_detect_penalty <= 0:
            raise ValueError("hypo_under_detect_penalty must be positive")
        if lambda_rate < 0:
            raise ValueError("lambda_rate must be non-negative")
        if early_horizon_weight <= 0:
            raise ValueError("early_horizon_weight must be positive")
        self.w_hypo = float(w_hypo)
        self.w_tir = float(w_tir)
        self.w_hyper = float(w_hyper)
        self.hypo_under_detect_penalty = float(hypo_under_detect_penalty)
        self.early_horizon_steps = int(early_horizon_steps)
        self.early_horizon_weight = float(early_horizon_weight)
        self.lambda_rate = float(lambda_rate)
        self.hypo_thr = float(hypo_thr)
        self.hyper_thr = float(hyper_thr)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if y_pred.shape != y_true.shape:
            raise ValueError(
                f"shape mismatch: y_pred={tuple(y_pred.shape)}, y_true={tuple(y_true.shape)}"
            )
        # Mask out any NaNs in y_true (boundary samples) — pers_resid v3 must
        # tolerate samples whose trajectory is partially out of the patient
        # timeline. ``build_trajectory_targets`` produced zero NaNs on the
        # current NPZ but the loss should be robust to future regenerations.
        valid_mask = ~torch.isnan(y_true)
        y_true_safe = torch.where(valid_mask, y_true, torch.zeros_like(y_true))

        # Zone weights from y_true_safe
        zone_w = torch.full_like(y_true_safe, self.w_tir)
        zone_w = torch.where(y_true_safe < self.hypo_thr,
                             torch.tensor(self.w_hypo, device=y_true_safe.device, dtype=y_true_safe.dtype),
                             zone_w)
        zone_w = torch.where(y_true_safe > self.hyper_thr,
                             torch.tensor(self.w_hyper, device=y_true_safe.device, dtype=y_true_safe.dtype),
                             zone_w)

        # Per-step horizon weight (up-weight the first ``early_horizon_steps``)
        T = y_true.shape[-1]
        horizon_w = torch.ones(T, device=y_true.device, dtype=y_true.dtype)
        if self.early_horizon_steps > 0 and self.early_horizon_steps <= T:
            horizon_w[: self.early_horizon_steps] = self.early_horizon_weight
        weight = zone_w * horizon_w.unsqueeze(0)

        # Asymmetric hypo under-detect penalty
        if self.hypo_under_detect_penalty != 1.0:
            under_detect = (y_true_safe < self.hypo_thr) & (y_pred > y_true_safe)
            weight = torch.where(under_detect, weight * self.hypo_under_detect_penalty, weight)

        # Point loss
        sq_err = (y_pred - y_true_safe) ** 2 * valid_mask.float()
        weighted = weight * sq_err
        denom = valid_mask.float().sum().clamp_min(1.0)
        l_point = weighted.sum() / denom

        # Rate-of-change loss
        l_rate = torch.tensor(0.0, device=y_true.device, dtype=y_true.dtype)
        if self.lambda_rate > 0 and T > 1:
            ref_rate = y_true_safe[:, 1:] - y_true_safe[:, :-1]
            pred_rate = y_pred[:, 1:] - y_pred[:, :-1]
            rate_mask = valid_mask[:, 1:] & valid_mask[:, :-1]
            rate_sq = (pred_rate - ref_rate) ** 2 * rate_mask.float()
            denom_r = rate_mask.float().sum().clamp_min(1.0)
            l_rate = rate_sq.sum() / denom_r

        return l_point + self.lambda_rate * l_rate

    def extra_repr(self) -> str:
        return (
            f"w_hypo={self.w_hypo}, w_tir={self.w_tir}, w_hyper={self.w_hyper}, "
            f"hypo_under_detect_penalty={self.hypo_under_detect_penalty}, "
            f"early_horizon_steps={self.early_horizon_steps}, "
            f"early_horizon_weight={self.early_horizon_weight}, "
            f"lambda_rate={self.lambda_rate}, "
            f"thr=({self.hypo_thr}, {self.hyper_thr})"
        )
