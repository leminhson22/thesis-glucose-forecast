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
