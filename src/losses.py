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


class ClinicalZoneRateLoss(ZoneWeightedMSE):
    """Zone-weighted loss with threshold-crossing and direction penalties.

    The base term is :class:`ZoneWeightedMSE`. Two clinical terms are added:

    * missed hypo/hyper penalties when the target is outside the safe range
      but the prediction remains inside it;
    * a direction penalty when the true future glucose changes materially
      from the current glucose but the predicted delta points the other way
      or stays too flat.

    The direction term needs the current glucose in mg/dL. For the
    persistence-residual model this is reconstructed from the last dynamic
    glucose z-score and the pid-index column appended to ``x_static``.
    """

    def __init__(
        self,
        *args,
        missed_hypo_penalty: float = 0.8,
        missed_hyper_penalty: float = 0.25,
        threshold_margin: float = 5.0,
        direction_penalty: float = 0.08,
        direction_delta_threshold: float = 10.0,
        direction_margin: float = 2.0,
        pid_glucose_mean: torch.Tensor | None = None,
        pid_glucose_std: torch.Tensor | None = None,
        glucose_dyn_idx: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if missed_hypo_penalty < 0 or missed_hyper_penalty < 0:
            raise ValueError("missed-zone penalties must be non-negative")
        if threshold_margin < 0:
            raise ValueError("threshold_margin must be non-negative")
        if direction_penalty < 0:
            raise ValueError("direction_penalty must be non-negative")
        if direction_delta_threshold < 0:
            raise ValueError("direction_delta_threshold must be non-negative")
        self.missed_hypo_penalty = float(missed_hypo_penalty)
        self.missed_hyper_penalty = float(missed_hyper_penalty)
        self.threshold_margin = float(threshold_margin)
        self.direction_penalty = float(direction_penalty)
        self.direction_delta_threshold = float(direction_delta_threshold)
        self.direction_margin = float(direction_margin)
        self.glucose_dyn_idx = int(glucose_dyn_idx)
        mean = None if pid_glucose_mean is None else torch.as_tensor(pid_glucose_mean, dtype=torch.float32)
        std = None if pid_glucose_std is None else torch.as_tensor(pid_glucose_std, dtype=torch.float32)
        self.register_buffer("pid_glucose_mean", mean, persistent=False)
        self.register_buffer("pid_glucose_std", std, persistent=False)

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.to(dtype=values.dtype)
        return (values * mask_f).sum() / mask_f.sum().clamp_min(1.0)

    def _last_glucose_mgdl(self, x_dyn: torch.Tensor, x_stat: torch.Tensor) -> torch.Tensor | None:
        if self.pid_glucose_mean is None or self.pid_glucose_std is None:
            return None
        pid_idx = x_stat[:, -1].long().clamp(min=0, max=self.pid_glucose_mean.numel() - 1)
        mean = self.pid_glucose_mean.to(device=x_dyn.device, dtype=x_dyn.dtype)[pid_idx]
        std = self.pid_glucose_std.to(device=x_dyn.device, dtype=x_dyn.dtype)[pid_idx]
        last_z = x_dyn[:, -1, self.glucose_dyn_idx]
        return last_z * std + mean

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        x_dyn: torch.Tensor | None = None,
        x_stat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        base = super().forward(y_pred, y_true)
        aux = torch.zeros((), device=y_true.device, dtype=y_true.dtype)

        if self.missed_hypo_penalty > 0:
            missed_hypo = (y_true < self.hypo_thr) & (y_pred >= self.hypo_thr)
            hypo_cost = torch.relu(y_pred - self.hypo_thr + self.threshold_margin) ** 2
            aux = aux + self.missed_hypo_penalty * self._masked_mean(hypo_cost, missed_hypo)

        if self.missed_hyper_penalty > 0:
            missed_hyper = (y_true > self.hyper_thr) & (y_pred <= self.hyper_thr)
            hyper_cost = torch.relu(self.hyper_thr - y_pred + self.threshold_margin) ** 2
            aux = aux + self.missed_hyper_penalty * self._masked_mean(hyper_cost, missed_hyper)

        if self.direction_penalty > 0 and x_dyn is not None and x_stat is not None:
            last_glucose = self._last_glucose_mgdl(x_dyn, x_stat)
            if last_glucose is not None:
                true_delta = y_true - last_glucose.unsqueeze(1)
                pred_delta = y_pred - last_glucose.unsqueeze(1)
                meaningful = true_delta.abs() >= self.direction_delta_threshold
                signed_pred = pred_delta * true_delta.sign()
                direction_cost = torch.relu(self.direction_margin - signed_pred) ** 2
                zone_boost = torch.ones_like(y_true)
                zone_boost = torch.where(y_true < self.hypo_thr, zone_boost * 2.0, zone_boost)
                zone_boost = torch.where(y_true > self.hyper_thr, zone_boost * 1.25, zone_boost)
                aux = aux + self.direction_penalty * self._masked_mean(direction_cost * zone_boost, meaningful)

        return base + aux

    def extra_repr(self) -> str:
        return (
            super().extra_repr()
            + f", missed_hypo_penalty={self.missed_hypo_penalty}, "
            + f"missed_hyper_penalty={self.missed_hyper_penalty}, "
            + f"direction_penalty={self.direction_penalty}"
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
