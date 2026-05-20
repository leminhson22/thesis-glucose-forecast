"""Uncertainty quantification via Split Conformal and Mondrian Conformal Prediction.

Conformal Prediction (CP) wraps a fitted point-estimator with a distribution-free
prediction interval that has finite-sample marginal coverage guarantees under
exchangeability. This module implements two variants:

  * **Split CP**: one quantile per horizon, fitted on the calibration set.
    Guarantees marginal coverage `>= 1 - alpha - 1/(n_cal+1)`.
  * **Mondrian CP per glycaemic zone**: one quantile per (horizon, zone) cell,
    where zone is the glycaemic zone of the *reference glucose* (hypo / TIR /
    hyper). Guarantees zone-conditional coverage `>= 1 - alpha` separately for
    each zone (modulo finite-sample correction); produces wider intervals in
    the tails where residuals are larger.

Both variants are *split-conformal*: they require the point-estimator to be
fitted on data disjoint from the calibration set. The calibration data is the
chronological validation split of the HUPA-UCM per-patient 70/15/15 partition
(§5.6 of the thesis report); the point-estimator is the proposed
`HybridCNNGRUPersResid` of §7.6, whose predictions on val and test are saved
to `outputs/tables/step6_v2_predictions.parquet`.

Non-conformity score: absolute residual ``|y_true - y_pred|`` (per horizon).
This is the standard choice for regression Split CP and yields *symmetric*
two-sided intervals ``[y_pred - q_alpha, y_pred + q_alpha]``.

References
----------
Vovk, Gammerman & Shafer 2005 — Algorithmic Learning in a Random World.
Lei, G'Sell, Rinaldo, Tibshirani & Wasserman 2018 — Distribution-Free
  Predictive Inference for Regression. JASA.
Romano, Patterson & Candès 2019 — Conformalized Quantile Regression. NeurIPS.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

try:
    from . import evaluate as E
except ImportError:
    import evaluate as E  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Core split-conformal primitives
# ---------------------------------------------------------------------------

def _finite_sample_quantile_level(n_cal: int, alpha: float) -> float:
    """Return the *finite-sample-corrected* quantile level for split CP.

    Per Lei et al. 2018 Theorem 1, the (1 - alpha) split-conformal interval
    uses the ``ceil((n_cal + 1) * (1 - alpha)) / n_cal`` empirical quantile
    of the absolute residuals on the calibration set. This produces marginal
    coverage at least ``1 - alpha`` in finite samples.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1); got {alpha!r}")
    if n_cal < 1:
        raise ValueError(f"n_cal must be >= 1; got {n_cal!r}")
    level = np.ceil((n_cal + 1) * (1.0 - alpha)) / n_cal
    return float(min(level, 1.0))


def split_conformal_quantile(
    residuals: np.ndarray,
    alpha: float,
) -> float:
    """Compute the split-conformal half-width quantile from val residuals.

    Parameters
    ----------
    residuals
        1-D array of *absolute* residuals on the calibration set, one entry
        per (sample, horizon) pair to be calibrated together.
    alpha
        Mis-coverage rate. ``alpha=0.10`` gives a nominal 90 % prediction
        interval; ``alpha=0.20`` gives a nominal 80 % interval.

    Returns
    -------
    q
        Half-width of the symmetric two-sided prediction interval, in mg/dL.
        Interval at test time is ``[y_pred - q, y_pred + q]``.
    """
    r = np.asarray(residuals, dtype=float).reshape(-1)
    r = r[np.isfinite(r)]
    if r.size == 0:
        raise ValueError("All calibration residuals were non-finite.")
    level = _finite_sample_quantile_level(r.size, alpha)
    return float(np.quantile(r, level, method="higher"))


def mondrian_conformal_quantiles(
    residuals: np.ndarray,
    zones: np.ndarray,
    alpha: float,
    zone_labels: Iterable[str] = E.ZONE_LABELS,
) -> dict[str, float]:
    """Compute one split-conformal quantile per glycaemic zone (Mondrian CP).

    Parameters
    ----------
    residuals, zones
        Aligned 1-D arrays. ``zones[i]`` is the glycaemic zone of the
        reference glucose for sample ``i``; ``residuals[i]`` is the absolute
        residual at that sample.
    alpha
        Mis-coverage rate (same convention as :func:`split_conformal_quantile`).
    zone_labels
        Zone labels to compute quantiles for. Default is ``("hypo", "tir",
        "hyper")``.

    Returns
    -------
    q_by_zone
        Mapping ``{zone_label: half_width_q_in_mgdl}``. Zones with empty
        calibration sub-samples receive the marginal Split CP quantile as
        fallback (with a warning printed to stderr).
    """
    r = np.asarray(residuals, dtype=float).reshape(-1)
    z = np.asarray(zones).reshape(-1)
    if r.shape != z.shape:
        raise ValueError(f"residuals and zones shape mismatch: {r.shape} vs {z.shape}")

    finite = np.isfinite(r)
    r = r[finite]
    z = z[finite]

    out: dict[str, float] = {}
    fallback = split_conformal_quantile(r, alpha)
    for label in zone_labels:
        mask = z == label
        if not mask.any():
            import sys
            print(
                f"[mondrian_conformal_quantiles] empty calibration zone "
                f"{label!r}; falling back to marginal quantile {fallback:.3f}",
                file=sys.stderr,
            )
            out[str(label)] = fallback
            continue
        out[str(label)] = split_conformal_quantile(r[mask], alpha)
    return out


# ---------------------------------------------------------------------------
# Interval construction and coverage evaluation
# ---------------------------------------------------------------------------

def build_intervals_split(
    y_pred: np.ndarray, q: float
) -> tuple[np.ndarray, np.ndarray]:
    """Build symmetric two-sided intervals for the Split CP variant."""
    y_pred = np.asarray(y_pred, dtype=float)
    return y_pred - q, y_pred + q


def build_intervals_mondrian(
    y_pred: np.ndarray,
    zones: np.ndarray,
    q_by_zone: Mapping[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Build symmetric two-sided intervals using the per-zone Mondrian CP quantiles.

    Each prediction uses the quantile corresponding to the zone of its own
    reference glucose. Samples whose zone is not in ``q_by_zone`` keep
    ``NaN`` interval bounds.
    """
    y_pred = np.asarray(y_pred, dtype=float)
    z = np.asarray(zones)
    half = np.full_like(y_pred, np.nan, dtype=float)
    for label, q in q_by_zone.items():
        half[z == label] = q
    return y_pred - half, y_pred + half


def coverage_metrics(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict[str, float]:
    """Empirical coverage and interval-width summary for a prediction-interval bundle.

    Returns a dict with empirical coverage (fraction of ``y_true`` in
    ``[lower, upper]``), mean and median interval width, and the count of
    samples used.
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    lower = np.asarray(lower, dtype=float).reshape(-1)
    upper = np.asarray(upper, dtype=float).reshape(-1)

    finite = np.isfinite(y_true) & np.isfinite(lower) & np.isfinite(upper)
    yt = y_true[finite]
    lo = lower[finite]
    up = upper[finite]

    if yt.size == 0:
        return {
            "n": 0,
            "coverage": float("nan"),
            "mean_width": float("nan"),
            "median_width": float("nan"),
        }

    covered = (yt >= lo) & (yt <= up)
    widths = up - lo
    return {
        "n": int(yt.size),
        "coverage": float(covered.mean()),
        "mean_width": float(np.mean(widths)),
        "median_width": float(np.median(widths)),
    }


# ---------------------------------------------------------------------------
# End-to-end orchestration over a master predictions DataFrame
# ---------------------------------------------------------------------------

@dataclass
class ConformalRun:
    """Container for a single (model, alpha, method) Conformal run.

    Attributes
    ----------
    method
        Either ``"split"`` or ``"mondrian"``.
    alpha
        Mis-coverage rate (1 - nominal coverage).
    q_by_horizon
        Mapping ``{horizon_min: q}`` for the Split CP variant, OR
        ``{horizon_min: {zone: q}}`` for the Mondrian variant.
    coverage_table
        Long-form DataFrame with columns ``model, method, alpha, horizon_min,
        zone, n, coverage, mean_width, median_width``. Zone "all" rows are
        the marginal (zone-pooled) coverage values.
    """

    method: str
    alpha: float
    q_by_horizon: dict
    coverage_table: pd.DataFrame


def calibrate_and_evaluate(
    df: pd.DataFrame,
    *,
    model_name: str,
    alphas: Iterable[float] = (0.10, 0.20),
    zone_labels: Iterable[str] = E.ZONE_LABELS,
    horizons: Iterable[int] = (30, 60, 90),
) -> list[ConformalRun]:
    """Calibrate Split CP and Mondrian CP on val, evaluate on test.

    Parameters
    ----------
    df
        Long-form predictions DataFrame with columns at least ``model``,
        ``split``, ``horizon_min``, ``y_true``, ``y_pred``, ``zone``.
        Both ``split == 'val'`` (calibration) and ``split == 'test'``
        (evaluation) rows must be present.
    model_name
        Value of the ``model`` column to filter on (e.g.
        ``"step6_v2_pers_resid"``).
    alphas
        Iterable of mis-coverage rates to run. Default ``(0.10, 0.20)`` for
        90 % and 80 % nominal coverage.
    zone_labels
        Iterable of glycaemic zone labels for Mondrian CP. Default
        ``("hypo", "tir", "hyper")``.
    horizons
        Iterable of horizon values (minutes) to process. Default
        ``(30, 60, 90)``.

    Returns
    -------
    runs
        List of :class:`ConformalRun`, one per (method, alpha) combination.
    """
    sub = df[df["model"] == model_name].copy()
    if sub.empty:
        raise ValueError(f"No rows for model={model_name!r} in input DataFrame.")
    if not {"val", "test"}.issubset(sub["split"].unique()):
        raise ValueError(
            "Input DataFrame must contain both 'val' and 'test' rows for "
            f"model={model_name!r}; found {sorted(sub['split'].unique())!r}."
        )

    # Absolute residuals.
    sub["abs_resid"] = (sub["y_true"] - sub["y_pred"]).abs()

    runs: list[ConformalRun] = []

    for alpha in alphas:
        # ----- Split CP (marginal, per-horizon) ----------------------------
        q_split: dict[int, float] = {}
        rows: list[dict] = []
        for h in horizons:
            cal = sub[(sub["split"] == "val") & (sub["horizon_min"] == h)]
            tst = sub[(sub["split"] == "test") & (sub["horizon_min"] == h)]
            q = split_conformal_quantile(cal["abs_resid"].to_numpy(), alpha)
            q_split[int(h)] = q

            lo, up = build_intervals_split(tst["y_pred"].to_numpy(), q)
            # Marginal coverage
            m_all = coverage_metrics(tst["y_true"].to_numpy(), lo, up)
            rows.append({
                "model": model_name, "method": "split", "alpha": alpha,
                "horizon_min": int(h), "zone": "all", "q": q, **m_all,
            })
            # Per-zone coverage (still under the marginal Split CP quantile)
            for zone in zone_labels:
                mask = tst["zone"].to_numpy() == zone
                if not mask.any():
                    continue
                m = coverage_metrics(
                    tst["y_true"].to_numpy()[mask], lo[mask], up[mask]
                )
                rows.append({
                    "model": model_name, "method": "split", "alpha": alpha,
                    "horizon_min": int(h), "zone": str(zone), "q": q, **m,
                })

        runs.append(ConformalRun(
            method="split", alpha=alpha, q_by_horizon=q_split,
            coverage_table=pd.DataFrame(rows),
        ))

        # ----- Mondrian CP (per-zone, per-horizon) -------------------------
        q_mondrian: dict[int, dict[str, float]] = {}
        rows = []
        for h in horizons:
            cal = sub[(sub["split"] == "val") & (sub["horizon_min"] == h)]
            tst = sub[(sub["split"] == "test") & (sub["horizon_min"] == h)]
            q_zones = mondrian_conformal_quantiles(
                cal["abs_resid"].to_numpy(),
                cal["zone"].to_numpy(),
                alpha,
                zone_labels=zone_labels,
            )
            q_mondrian[int(h)] = q_zones

            lo, up = build_intervals_mondrian(
                tst["y_pred"].to_numpy(),
                tst["zone"].to_numpy(),
                q_zones,
            )
            m_all = coverage_metrics(tst["y_true"].to_numpy(), lo, up)
            rows.append({
                "model": model_name, "method": "mondrian", "alpha": alpha,
                "horizon_min": int(h), "zone": "all", "q": float("nan"),
                **m_all,
            })
            for zone in zone_labels:
                mask = tst["zone"].to_numpy() == zone
                if not mask.any():
                    continue
                m = coverage_metrics(
                    tst["y_true"].to_numpy()[mask], lo[mask], up[mask]
                )
                rows.append({
                    "model": model_name, "method": "mondrian", "alpha": alpha,
                    "horizon_min": int(h), "zone": str(zone),
                    "q": q_zones[zone], **m,
                })

        runs.append(ConformalRun(
            method="mondrian", alpha=alpha, q_by_horizon=q_mondrian,
            coverage_table=pd.DataFrame(rows),
        ))

    return runs


def quantile_table(runs: list[ConformalRun]) -> pd.DataFrame:
    """Flatten the calibration quantiles into a long-form table for reporting."""
    rows: list[dict] = []
    for r in runs:
        if r.method == "split":
            for h, q in r.q_by_horizon.items():
                rows.append({
                    "method": "split", "alpha": r.alpha,
                    "horizon_min": int(h), "zone": "all", "q_mgdl": float(q),
                })
        elif r.method == "mondrian":
            for h, q_zones in r.q_by_horizon.items():
                for zone, q in q_zones.items():
                    rows.append({
                        "method": "mondrian", "alpha": r.alpha,
                        "horizon_min": int(h), "zone": str(zone),
                        "q_mgdl": float(q),
                    })
        else:
            raise ValueError(f"Unknown method {r.method!r}")
    return pd.DataFrame(rows)


def coverage_summary(runs: list[ConformalRun]) -> pd.DataFrame:
    """Concatenate the per-run coverage tables into one DataFrame."""
    return pd.concat([r.coverage_table for r in runs], ignore_index=True)


# ---------------------------------------------------------------------------
# Convenience for building per-sample interval columns on the master parquet
# ---------------------------------------------------------------------------

def _quantile_with_finite_correction(
    residuals_sorted: np.ndarray, alpha: float
) -> float:
    """Compute the split-conformal half-width using the finite-sample-corrected
    empirical quantile from a *pre-sorted* residual array.

    Returns the residual at position ``ceil((n+1)(1-alpha)) - 1`` (0-indexed)
    after clamping to ``[0, n-1]``. Equivalent to
    :func:`split_conformal_quantile` for a fixed residual sample but faster
    when the residual array is sorted once and queried many times.
    """
    n = residuals_sorted.size
    if n == 0:
        raise ValueError("Empty residual array")
    alpha = float(np.clip(alpha, 1.0 / (n + 1), 1.0 - 1.0 / (n + 1)))
    pos = int(np.ceil((n + 1) * (1.0 - alpha))) - 1
    pos = int(np.clip(pos, 0, n - 1))
    return float(residuals_sorted[pos])


# ---------------------------------------------------------------------------
# Adaptive Conformal Inference (Gibbs & Candès 2021)
# ---------------------------------------------------------------------------

def adaptive_conformal_inference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cal_residuals_sorted_by_zone: dict[str, np.ndarray],
    zones: np.ndarray,
    *,
    alpha_target: float = 0.10,
    gamma: float = 0.005,
    alpha_min: float = 0.005,
    alpha_max: float = 0.50,
    return_trajectory: bool = True,
) -> dict[str, np.ndarray]:
    """Per-zone Adaptive Conformal Inference along a sequential time-series.

    Implements the Gibbs & Candès 2021 update rule with one independent
    state variable ``alpha_z`` per glycaemic zone:

        alpha_{t+1, z(t)} = alpha_{t, z(t)} + gamma * (alpha_target - err_t)

    where ``err_t`` is 1 if the realised ``y_true[t]`` lies outside the
    interval emitted at step ``t``, else 0; only the alpha for the zone of
    the *current* reference glucose updates at each step. ``alpha_t`` is
    clipped to the open interval ``(alpha_min, alpha_max)``.

    Parameters
    ----------
    y_true, y_pred
        1-D arrays of equal length giving the realised and point-forecast
        glucose values in mg/dL, in chronological order.
    cal_residuals_sorted_by_zone
        Mapping ``{zone_label: np.ndarray}`` of *pre-sorted* absolute
        residuals on the calibration (validation) split. One array per
        zone; the array is queried at any ``alpha_t`` to recover the
        Split-CP half-width for that zone.
    zones
        1-D array of length ``len(y_true)`` giving the glycaemic zone of
        the reference glucose at each step.
    alpha_target
        Long-run target mis-coverage rate. The ACI update drives the
        empirical miss rate toward this value.
    gamma
        Learning rate. Larger ``gamma`` -> faster adaptation but noisier
        coverage; smaller ``gamma`` -> slower adaptation but more stable
        intervals. Default 0.005 follows the published recommendation
        (1/200) for daily-resolution time series; on the 5-minute HUPA
        grid this corresponds to a window-of-influence of roughly
        16 hours.
    alpha_min, alpha_max
        Safety clipping range for ``alpha_t``.
    return_trajectory
        If ``True`` (default), the returned dict additionally contains the
        per-step ``alpha_t`` trajectory and per-step interval bounds.

    Returns
    -------
    result
        Dict with keys ``empirical_coverage``, ``empirical_coverage_by_zone``,
        ``mean_width``, ``mean_width_by_zone`` and optionally ``alpha_t``,
        ``lower``, ``upper``, ``hit``.
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    zones = np.asarray(zones).reshape(-1)
    n = len(y_true)
    if not (len(y_pred) == n == len(zones)):
        raise ValueError("y_true, y_pred, zones must have equal length")

    alpha_by_zone: dict[str, float] = {z: float(alpha_target) for z in cal_residuals_sorted_by_zone}

    alpha_traj = np.full(n, np.nan, dtype=float)
    lower = np.full(n, np.nan, dtype=float)
    upper = np.full(n, np.nan, dtype=float)
    hit = np.zeros(n, dtype=bool)

    for t in range(n):
        z = str(zones[t])
        if z not in cal_residuals_sorted_by_zone:
            continue  # unknown zone — leave NaN
        a = alpha_by_zone[z]
        q = _quantile_with_finite_correction(cal_residuals_sorted_by_zone[z], a)
        lo = y_pred[t] - q
        up = y_pred[t] + q
        covered = (y_true[t] >= lo) & (y_true[t] <= up)
        err = 0.0 if covered else 1.0
        alpha_traj[t] = a
        lower[t] = lo
        upper[t] = up
        hit[t] = bool(covered)
        # Update rule (only the alpha for the current zone changes)
        a_new = a + gamma * (alpha_target - err)
        alpha_by_zone[z] = float(np.clip(a_new, alpha_min, alpha_max))

    valid = ~np.isnan(alpha_traj)
    out: dict[str, np.ndarray | float | dict] = {}
    out["empirical_coverage"] = float(hit[valid].mean()) if valid.any() else float("nan")
    out["mean_width"] = float((upper[valid] - lower[valid]).mean()) if valid.any() else float("nan")
    cov_zone: dict[str, float] = {}
    width_zone: dict[str, float] = {}
    n_zone: dict[str, int] = {}
    final_alpha: dict[str, float] = {}
    for z in cal_residuals_sorted_by_zone:
        mask = (zones == z) & valid
        if mask.any():
            cov_zone[z] = float(hit[mask].mean())
            width_zone[z] = float((upper[mask] - lower[mask]).mean())
            n_zone[z] = int(mask.sum())
            final_alpha[z] = float(alpha_by_zone[z])
        else:
            cov_zone[z] = float("nan")
            width_zone[z] = float("nan")
            n_zone[z] = 0
            final_alpha[z] = float(alpha_by_zone[z])
    out["empirical_coverage_by_zone"] = cov_zone
    out["mean_width_by_zone"] = width_zone
    out["n_by_zone"] = n_zone
    out["final_alpha_by_zone"] = final_alpha

    if return_trajectory:
        out["alpha_t"] = alpha_traj
        out["lower"] = lower
        out["upper"] = upper
        out["hit"] = hit
    return out


def attach_intervals_to_predictions(
    df_predictions: pd.DataFrame,
    runs: list[ConformalRun],
    *,
    model_name: str,
    split: str = "test",
) -> pd.DataFrame:
    """Append `lower_<method>_<alpha>` / `upper_<method>_<alpha>` columns.

    Returns a copy of the subset ``df_predictions[(model, split)]`` with one
    extra (lower, upper) column pair per run. Useful for plotting time-series
    overlays.
    """
    out = df_predictions[
        (df_predictions["model"] == model_name)
        & (df_predictions["split"] == split)
    ].copy()
    if out.empty:
        raise ValueError(f"No rows for {model_name!r} / split={split!r}")

    for r in runs:
        suffix = f"{r.method}_a{int(round(r.alpha * 100))}"
        lo_col = f"lower_{suffix}"
        up_col = f"upper_{suffix}"
        out[lo_col] = np.nan
        out[up_col] = np.nan
        for h, q_or_zones in r.q_by_horizon.items():
            mask = out["horizon_min"] == int(h)
            if r.method == "split":
                q = float(q_or_zones)  # type: ignore[arg-type]
                out.loc[mask, lo_col] = out.loc[mask, "y_pred"] - q
                out.loc[mask, up_col] = out.loc[mask, "y_pred"] + q
            elif r.method == "mondrian":
                q_zones = q_or_zones  # type: ignore[assignment]
                for zone, q in q_zones.items():
                    m = mask & (out["zone"] == zone)
                    out.loc[m, lo_col] = out.loc[m, "y_pred"] - float(q)
                    out.loc[m, up_col] = out.loc[m, "y_pred"] + float(q)
            else:
                raise ValueError(f"Unknown method {r.method!r}")
    return out
