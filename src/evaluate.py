"""Evaluation framework for HUPA-UCM glucose forecasting baselines and models.

Computes:
  * Per-horizon MAE / RMSE (pooled, row-weighted).
  * Per-zone errors binned by ``y_true`` using the CLAUDE.md mg/dL thresholds
    (hypo <70, TIR 70-180, hyper >180).
  * Per-patient MAE / RMSE plus patient-averaged summary
    (long-patient-strategy memory: report both pooled and patient-averaged).
  * Clarke Error Grid Analysis zones A-E (Clarke et al. 1987) using the
    canonical piecewise lines, vectorised.

All inputs and outputs use mg/dL. Predictions and targets are 2-D arrays of
shape ``(n_samples, n_horizons)`` with horizon order ``(30, 60, 90)`` minutes.
"""
from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
import pandas as pd

try:  # package import
    from . import config as C
except ImportError:  # flat import path (notebook adds src/ to sys.path)
    import config as C  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Glycaemic zones (mg/dL thresholds defined in CLAUDE.md and SKILL.md)
# ---------------------------------------------------------------------------

ZONE_LABELS: tuple[str, ...] = ("hypo", "tir", "hyper")
CLARKE_ZONES: tuple[str, ...] = ("A", "B", "C", "D", "E")


def zone_of(y: np.ndarray) -> np.ndarray:
    """Bin glucose values into hypo / tir / hyper labels."""
    y = np.asarray(y)
    out = np.empty(y.shape, dtype="<U5")
    out[:] = "tir"
    out[y < C.GLUCOSE_HYPO_THRESHOLD] = "hypo"
    out[y > C.GLUCOSE_HYPER_THRESHOLD] = "hyper"
    return out


# ---------------------------------------------------------------------------
# Metric primitives
# ---------------------------------------------------------------------------

def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


# ---------------------------------------------------------------------------
# Per-horizon (pooled, row-weighted)
# ---------------------------------------------------------------------------

def per_horizon_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon_min: Iterable[int] = C.HORIZON_MINUTES,
) -> pd.DataFrame:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")
    rows = []
    for h_idx, h in enumerate(horizon_min):
        yt = y_true[:, h_idx]
        yp = y_pred[:, h_idx]
        n = int(len(yt))
        rows.append({"horizon_min": int(h), "metric": "mae", "value": _mae(yt, yp), "n_samples": n})
        rows.append({"horizon_min": int(h), "metric": "rmse", "value": _rmse(yt, yp), "n_samples": n})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-zone (binned by y_true)
# ---------------------------------------------------------------------------

def per_zone_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon_min: Iterable[int] = C.HORIZON_MINUTES,
) -> pd.DataFrame:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rows = []
    for h_idx, h in enumerate(horizon_min):
        yt = y_true[:, h_idx]
        yp = y_pred[:, h_idx]
        zones = zone_of(yt)
        for z in ZONE_LABELS:
            mask = zones == z
            n = int(mask.sum())
            if n == 0:
                for metric in ("mae", "rmse"):
                    rows.append({
                        "horizon_min": int(h), "zone": z, "n_samples": 0,
                        "metric": metric, "value": np.nan,
                    })
                continue
            rows.append({
                "horizon_min": int(h), "zone": z, "n_samples": n,
                "metric": "mae", "value": _mae(yt[mask], yp[mask]),
            })
            rows.append({
                "horizon_min": int(h), "zone": z, "n_samples": n,
                "metric": "rmse", "value": _rmse(yt[mask], yp[mask]),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-patient + patient-averaged
# ---------------------------------------------------------------------------

def per_patient_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    participant_ids: np.ndarray,
    horizon_min: Iterable[int] = C.HORIZON_MINUTES,
) -> pd.DataFrame:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    pids = np.asarray(participant_ids)
    rows = []
    for pid in np.unique(pids):
        mask = pids == pid
        n = int(mask.sum())
        for h_idx, h in enumerate(horizon_min):
            yt = y_true[mask, h_idx]
            yp = y_pred[mask, h_idx]
            rows.append({
                "participant_id": str(pid), "horizon_min": int(h),
                "metric": "mae", "value": _mae(yt, yp), "n_samples": n,
            })
            rows.append({
                "participant_id": str(pid), "horizon_min": int(h),
                "metric": "rmse", "value": _rmse(yt, yp), "n_samples": n,
            })
    return pd.DataFrame(rows)


def patient_averaged_summary(per_patient_df: pd.DataFrame) -> pd.DataFrame:
    """Unweighted mean and std across patients for each (horizon, metric)."""
    grouped = (
        per_patient_df
        .groupby(["horizon_min", "metric"], as_index=False)["value"]
        .agg(patient_avg="mean", patient_sd="std", n_patients="count")
    )
    return grouped


# ---------------------------------------------------------------------------
# Clarke Error Grid Analysis
# ---------------------------------------------------------------------------

def clarke_eg_zones(ref: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Vectorised Clarke EGA labelling. Inputs are mg/dL arrays of equal length.

    Replicates the canonical if/elif chain with precedence A > E > D > C > B by
    overwriting masks in reverse precedence order so A wins overlap ties.
    """
    ref = np.asarray(ref, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if ref.shape != pred.shape:
        raise ValueError("ref and pred must have the same shape")
    zones = np.full(ref.shape, "B", dtype="<U1")
    # Zone C — overcorrection
    mask_c = (
        ((ref >= 70) & (ref <= 290) & (pred >= ref + 110))
        | ((ref >= 130) & (ref <= 180) & (pred <= (7.0 / 5.0) * ref - 182))
    )
    zones[mask_c] = "C"
    # Zone D — failure to detect dangerous condition
    mask_d = (
        ((ref >= 240) & (pred >= 70) & (pred <= 180))
        | ((ref <= 70) & (pred >= 70) & (pred <= 180))
    )
    zones[mask_d] = "D"
    # Zone E — erroneous treatment (opposite direction)
    mask_e = (
        ((ref >= 180) & (pred <= 70))
        | ((ref <= 70) & (pred >= 180))
    )
    zones[mask_e] = "E"
    # Zone A — clinically accurate; wins overlap ties
    mask_a = (
        ((ref <= 70) & (pred <= 70))
        | ((pred <= 1.2 * ref) & (pred >= 0.8 * ref))
    )
    zones[mask_a] = "A"
    return zones


def clarke_eg_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon_min: Iterable[int] = C.HORIZON_MINUTES,
) -> pd.DataFrame:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rows = []
    for h_idx, h in enumerate(horizon_min):
        zones = clarke_eg_zones(y_true[:, h_idx], y_pred[:, h_idx])
        n_total = int(len(zones))
        for z in CLARKE_ZONES:
            n = int((zones == z).sum())
            rows.append({
                "horizon_min": int(h), "zone": z, "n_samples": n,
                "pct": 100.0 * n / n_total if n_total else np.nan,
                "total": n_total,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Bundled entry point
# ---------------------------------------------------------------------------

def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    participant_ids: np.ndarray,
    model_name: str,
    split_name: str,
    horizon_min: Iterable[int] = C.HORIZON_MINUTES,
) -> dict[str, pd.DataFrame]:
    """Compute the full metric bundle for one model on one split.

    Every returned DataFrame has ``model`` and ``split`` columns prepended so
    results from multiple runs concatenate cleanly into a single CSV.
    """
    horizon_min = tuple(int(h) for h in horizon_min)
    bundle: dict[str, pd.DataFrame] = {
        "per_horizon": per_horizon_metrics(y_true, y_pred, horizon_min),
        "per_zone": per_zone_metrics(y_true, y_pred, horizon_min),
        "per_patient": per_patient_metrics(y_true, y_pred, participant_ids, horizon_min),
        "clarke_eg": clarke_eg_summary(y_true, y_pred, horizon_min),
    }
    for df in bundle.values():
        df.insert(0, "split", split_name)
        df.insert(0, "model", model_name)
    bundle["patient_averaged"] = patient_averaged_summary(bundle["per_patient"])
    bundle["patient_averaged"].insert(0, "split", split_name)
    bundle["patient_averaged"].insert(0, "model", model_name)
    return bundle


def compact_summary(bundle: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Single-row-per-(model, split, horizon) wide summary for quick printing."""
    per_h = bundle["per_horizon"].pivot_table(
        index=["model", "split", "horizon_min"], columns="metric", values="value"
    ).reset_index()
    pat = bundle["patient_averaged"].pivot_table(
        index=["model", "split", "horizon_min"], columns="metric", values="patient_avg"
    ).rename(columns={"mae": "mae_pat_avg", "rmse": "rmse_pat_avg"}).reset_index()
    clarke = bundle["clarke_eg"].pivot_table(
        index=["model", "split", "horizon_min"], columns="zone", values="pct"
    ).add_prefix("clarke_pct_").reset_index()
    out = per_h.merge(pat, on=["model", "split", "horizon_min"])
    out = out.merge(clarke, on=["model", "split", "horizon_min"])
    return out
