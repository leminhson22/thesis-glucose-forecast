"""Integrated Gradients explainability for the proposed PersResid model.

Sundararajan, Taly & Yan 2017 — *Axiomatic Attribution for Deep Networks*.
The integrated gradient at input ``x`` with baseline ``x'`` for a scalar
output ``F`` is the path integral

    IG_i(x) = (x_i - x'_i) * integral_{alpha=0..1} dF(x' + alpha (x - x')) / dx_i d_alpha

approximated by a discrete Riemann sum over ``m`` interpolation steps.

This module implements IG for the multi-horizon
:class:`HybridCNNGRUPersResid` model. Because the model outputs a 3-vector
``(y_30, y_60, y_90)``, we run IG once per horizon (selecting the scalar
output for that horizon as the target) and produce a 3-channel attribution
tensor of shape ``(B, H_lookback, F_dynamic + F_static)``.

Baseline choice
---------------
The model consumes z-score-scaled inputs (per §5.7 of report.md). The
neutral baseline in z-space is the all-zeros tensor, which corresponds to
each feature being at its training-set mean. We use this baseline.

Output convention
-----------------
Attributions are returned in the *scaled* (z-score) space. For absolute
attribution magnitudes in original units, multiply each feature column by
its training std. For the relative-importance heat-maps in §10 of the
report, we report the squared L2-norm of each feature attribution within
each timestep, which is invariant under per-feature scaling and produces
a single (B, H_lookback, F_dynamic) heat-map.

Reference
---------
Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic Attribution for
Deep Networks. ICML 2017.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Core Integrated Gradients
# ---------------------------------------------------------------------------

def _make_interpolated_inputs(
    x: torch.Tensor,
    x_baseline: torch.Tensor,
    m: int,
) -> torch.Tensor:
    """Build the Riemann grid of interpolated inputs.

    Returns a tensor of shape ``(m, *x.shape)`` with rows
    ``x_baseline + (k / m) * (x - x_baseline)`` for ``k = 1, ..., m``.
    The midpoint Riemann rule with shifted ``k = 1..m`` is equivalent to
    the right-endpoint rule and is the standard choice in IG implementations.
    """
    if x.shape != x_baseline.shape:
        raise ValueError(f"shape mismatch x {x.shape} vs baseline {x_baseline.shape}")
    alphas = torch.linspace(1.0 / m, 1.0, m, device=x.device, dtype=x.dtype)
    delta = (x - x_baseline).unsqueeze(0)  # (1, *x.shape)
    base = x_baseline.unsqueeze(0)
    return base + alphas.view(-1, *([1] * x.ndim)) * delta


def integrated_gradients_dyn(
    model: nn.Module,
    x_dyn: torch.Tensor,
    x_stat: torch.Tensor,
    *,
    horizon_idx: int,
    m: int = 50,
    x_dyn_baseline: torch.Tensor | None = None,
    chunk_size: int = 16,
) -> torch.Tensor:
    """Compute IG attributions for the dynamic input at one horizon.

    Parameters
    ----------
    model
        Torch module returning a ``(B, n_horizons)`` tensor when called as
        ``model(x_dyn, x_stat)``. Must be in ``eval()`` mode.
    x_dyn
        Dynamic input tensor of shape ``(B, T, F_dyn)``. Will be detached;
        gradient tracking is added internally.
    x_stat
        Static input tensor of shape ``(B, F_stat)``. Held constant across
        the Riemann grid.
    horizon_idx
        Integer in ``{0, 1, 2}`` selecting the horizon (30 / 60 / 90 min).
    m
        Number of Riemann interpolation steps. 50 is a standard default
        that achieves the IG completeness axiom within ~1 %% error for
        most networks.
    x_dyn_baseline
        Baseline input. If ``None``, uses zeros (z-score baseline ==
        feature mean per §5.7 scaling).
    chunk_size
        Mini-batch size along the ``m`` axis for memory efficiency.

    Returns
    -------
    attributions
        Tensor of shape ``(B, T, F_dyn)`` in the same scaled (z-score)
        space as ``x_dyn``.
    """
    model.eval()
    device = next(model.parameters()).device
    x_dyn = x_dyn.detach().to(device)
    x_stat = x_stat.detach().to(device)
    if x_dyn_baseline is None:
        x_dyn_baseline = torch.zeros_like(x_dyn)
    x_dyn_baseline = x_dyn_baseline.to(device)

    grads_accum = torch.zeros_like(x_dyn)

    for start in range(0, m, chunk_size):
        end = min(m, start + chunk_size)
        alphas = torch.linspace(
            (start + 1) / m, end / m, end - start, device=device, dtype=x_dyn.dtype
        )  # (k,)
        # Build (k * B, T, F_dyn) interpolated batch
        delta = (x_dyn - x_dyn_baseline).unsqueeze(0)  # (1, B, T, F)
        base = x_dyn_baseline.unsqueeze(0)
        interp = base + alphas.view(-1, 1, 1, 1) * delta  # (k, B, T, F)
        interp = interp.reshape(-1, *x_dyn.shape[1:])      # (k*B, T, F)
        interp.requires_grad_(True)

        # Repeat static for each alpha
        x_stat_rep = x_stat.unsqueeze(0).expand(end - start, -1, -1).reshape(
            -1, x_stat.shape[-1]
        )

        out = model(interp, x_stat_rep)             # (k*B, n_horizons)
        target = out[:, horizon_idx].sum()
        target.backward()

        g = interp.grad.detach().reshape(end - start, *x_dyn.shape)  # (k, B, T, F)
        grads_accum = grads_accum + g.mean(dim=0) * (end - start) / m

    # IG = (x - x_baseline) * average gradient over the grid.
    attributions = (x_dyn - x_dyn_baseline) * grads_accum
    return attributions.detach().cpu()


# ---------------------------------------------------------------------------
# Aggregation helpers for the report.md narrative
# ---------------------------------------------------------------------------

def temporal_feature_heatmap(
    attributions: torch.Tensor,
    feature_names: list[str],
    *,
    aggregate: str = "abs_mean",
) -> pd.DataFrame:
    """Aggregate per-sample IG attributions into a (timestep × feature) heatmap.

    Parameters
    ----------
    attributions
        Tensor of shape ``(B, T, F_dyn)``.
    feature_names
        Length-``F_dyn`` list of dynamic feature names.
    aggregate
        One of ``"abs_mean"`` (recommended; mean of |IG| across batch),
        ``"mean"`` (signed mean), or ``"l2"`` (sample-wise L2 norm then mean).

    Returns
    -------
    df
        DataFrame indexed by ``timestep`` (0..T-1, oldest..most recent), with
        one column per dynamic feature, values >= 0 for ``abs_mean`` / ``l2``.
    """
    a = attributions.detach().cpu().numpy()
    if aggregate == "abs_mean":
        grid = np.mean(np.abs(a), axis=0)
    elif aggregate == "mean":
        grid = np.mean(a, axis=0)
    elif aggregate == "l2":
        grid = np.sqrt(np.mean(a ** 2, axis=0))
    else:
        raise ValueError(f"unknown aggregate {aggregate!r}")
    df = pd.DataFrame(grid, columns=feature_names)
    df.index.name = "timestep"
    return df


def global_feature_importance(
    attributions: torch.Tensor,
    feature_names: list[str],
) -> pd.DataFrame:
    """Sum the absolute attribution across timestep and batch for each feature.

    Returns a sorted DataFrame with columns ``feature`` and ``importance``.
    """
    a = attributions.detach().cpu().numpy()
    total = np.sum(np.abs(a), axis=(0, 1))  # (F,)
    df = pd.DataFrame({"feature": feature_names, "importance": total})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df["importance_pct"] = 100.0 * df["importance"] / df["importance"].sum()
    return df


def temporal_focus(attributions: torch.Tensor) -> pd.DataFrame:
    """How much of the per-sample attribution is concentrated at each lookback step.

    Returns a DataFrame indexed by ``timestep`` with two columns:
    ``mean_abs_importance`` and ``share_pct`` (across the full lookback window).
    The most recent timestep is ``T - 1``.
    """
    a = attributions.detach().cpu().numpy()
    per_t = np.mean(np.abs(a).sum(axis=2), axis=0)  # (T,)
    df = pd.DataFrame({"mean_abs_importance": per_t})
    df.index.name = "timestep"
    df["share_pct"] = 100.0 * df["mean_abs_importance"] / df["mean_abs_importance"].sum()
    return df


@dataclass
class IGCase:
    """A single instance for IG case-study visualisation in §10."""

    sample_idx: int
    participant_id: str
    horizon_min: int
    zone: str
    y_true: float
    y_pred: float
    abs_err: float
    attributions: torch.Tensor   # (T, F_dyn)
    last_glucose: float
