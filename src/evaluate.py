"""Evaluation framework for HUPA-UCM glucose forecasting baselines and models.

Computes:
  * Per-horizon MAE / RMSE (pooled, row-weighted).
  * Per-zone errors binned by ``y_true`` using the CLAUDE.md mg/dL thresholds
    (hypo <70, TIR 70-180, hyper >180).
  * Per-patient MAE / RMSE plus patient-averaged summary
    (long-patient-strategy memory: report both pooled and patient-averaged).
  * **Continuous Glucose-Error Grid Analysis (CG-EGA)** — Kovatchev et al.
    Diabetes Technology & Therapeutics 2004. Primary clinical-safety metric
    for CGM forecasting per SKILL.md v2.0. Combines a Point-EGA (P-EGA)
    classification of (reference, prediction) value pairs with a Rate-EGA
    (R-EGA) classification of (reference rate-of-change, predicted
    rate-of-change) over a 15-minute window; the joint class is mapped to
    Accurate Prediction (AP), Benign Error (BE), or Erroneous Prediction
    (EP) by a glycaemic-zone-conditional matrix.
  * Clarke Error Grid Analysis zones A-E (Clarke et al. 1987) retained as
    a legacy point-glucose grid for backwards compatibility with Step 5
    artefacts; do not use as the primary clinical-safety claim per
    SKILL.md v2.0 §5.5.

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
CG_EGA_CLASSES: tuple[str, ...] = ("AP", "BE", "EP")  # Accurate / Benign / Erroneous

# CG-EGA rate-of-change window. Kovatchev 2004 uses 15 min on a 5-min CGM grid
# (lag = 3 samples). HUPA-UCM is on a 5-min grid (SAMPLING_STEP_MIN = 5).
CG_EGA_RATE_LAG_STEPS = 3
CG_EGA_RATE_WINDOW_MIN = CG_EGA_RATE_LAG_STEPS * 5  # 15 minutes


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
# Continuous Glucose-Error Grid Analysis (CG-EGA) — Kovatchev 2004
# ---------------------------------------------------------------------------
#
# CG-EGA is the primary CGM-specific clinical-safety metric for this thesis
# (SKILL.md v2.0 §5.5). It supersedes Clarke EGA as the headline error-grid
# method because Clarke EGA evaluates only point glucose accuracy and does
# not penalise direction/rate errors that are clinically dangerous for CGM
# forecasting (e.g. predicting flat glucose when the true value is falling).
#
# CG-EGA combines two sub-grids:
#   * Point-EGA (P-EGA): A/B/C/D/E zones over (reference glucose, predicted
#     glucose). The standard Clarke EGA static grid is used as P-EGA;
#     Kovatchev did not redefine the point grid for CG-EGA.
#   * Rate-EGA (R-EGA): rA/rB/rC/rD/rE zones over (reference rate-of-change,
#     predicted rate-of-change) in mg/dL/min, computed over a 15-minute
#     window (lag = 3 samples on the HUPA 5-min grid).
#
# The combined classification depends on the reference glucose's glycaemic
# zone (hypo / euglycaemia / hyper), because the clinical cost of a
# rate error differs across zones: an under-detected fall in hypo is
# always erroneous, but the same rate error in euglycaemia may be benign.
# Each combination ``(zone, point_zone, rate_zone)`` maps to
# ``AP`` (Accurate Prediction), ``BE`` (Benign Error), or ``EP``
# (Erroneous Prediction).
#
# The combination matrices below follow Kovatchev et al. (2004) Table 2 and
# its standard re-implementations (e.g. the ``ega`` Python package).


def _rate_zones(ref_rate: np.ndarray, pred_rate: np.ndarray) -> np.ndarray:
    """Vectorised Rate-EGA labelling. Inputs are mg/dL/min, output ``A``-``E``.

    Rate-EGA partitions the (ref_rate, pred_rate) plane into five zones
    following Kovatchev et al. 2004:

    * ``A`` (accurate): predicted rate is within 1 mg/dL/min of the
      reference rate AND the two rates share sign or are both within the
      ±1 mg/dL/min slow band.
    * ``E`` (erroneous opposite direction): predicted and reference rates
      have opposite signs with both magnitudes above 1 mg/dL/min — a
      treatment in the wrong direction would be initiated.
    * ``D`` (failure to detect rate): one rate is in the rapid band
      (|rate| > 2 mg/dL/min) while the other is in the slow band
      (|rate| ≤ 1 mg/dL/min) — a clinically rapid change was missed or
      manufactured.
    * ``C`` (overcorrection): same direction but |pred - ref| > 2
      mg/dL/min — clinically significant magnitude error.
    * ``B`` (benign): everything else.

    Precedence (highest first): ``E > D > C > A > B``. ``A`` overrides
    ``B`` after the precedence pass so accurate-within-1 cases are always
    labelled ``A``.
    """
    ref = np.asarray(ref_rate, dtype=float)
    pred = np.asarray(pred_rate, dtype=float)
    if ref.shape != pred.shape:
        raise ValueError("ref_rate and pred_rate must have the same shape")
    abs_ref = np.abs(ref)
    abs_pred = np.abs(pred)
    diff = np.abs(pred - ref)
    sign_prod = np.sign(ref) * np.sign(pred)  # negative when opposite
    both_slow = (abs_ref <= 1) & (abs_pred <= 1)
    same_dir = (sign_prod >= 0) | both_slow

    zones = np.full(ref.shape, "B", dtype="<U1")
    # rA (accurate) — small absolute rate error AND consistent direction
    mask_a = (diff <= 1.0) & same_dir
    zones[mask_a] = "A"
    # rC (overcorrection) — same direction but |pred - ref| > 2
    mask_c = same_dir & (diff > 2.0)
    zones[mask_c & (zones != "A")] = "C"
    # rD (failure to detect rapid change)
    mask_d = ((abs_ref > 2) & (abs_pred <= 1)) | ((abs_ref <= 1) & (abs_pred > 2))
    zones[mask_d] = "D"
    # rE (erroneous treatment direction)
    mask_e = (sign_prod < 0) & (abs_ref > 1) & (abs_pred > 1)
    zones[mask_e] = "E"
    return zones


# Combination matrices — keyed by (point_zone, rate_zone) and selected by
# the reference glycaemic zone (hypo / tir / hyper).
# Values: "AP" (Accurate), "BE" (Benign Error), "EP" (Erroneous Prediction).
# Source: Kovatchev et al. (2004) Table 2.

_CG_EGA_MATRIX_HYPO: dict[tuple[str, str], str] = {
    ("A", "A"): "AP", ("A", "B"): "AP", ("A", "C"): "BE", ("A", "D"): "EP", ("A", "E"): "EP",
    ("B", "A"): "BE", ("B", "B"): "BE", ("B", "C"): "EP", ("B", "D"): "EP", ("B", "E"): "EP",
    ("C", "A"): "EP", ("C", "B"): "EP", ("C", "C"): "EP", ("C", "D"): "EP", ("C", "E"): "EP",
    ("D", "A"): "EP", ("D", "B"): "EP", ("D", "C"): "EP", ("D", "D"): "EP", ("D", "E"): "EP",
    ("E", "A"): "EP", ("E", "B"): "EP", ("E", "C"): "EP", ("E", "D"): "EP", ("E", "E"): "EP",
}

_CG_EGA_MATRIX_TIR: dict[tuple[str, str], str] = {
    ("A", "A"): "AP", ("A", "B"): "AP", ("A", "C"): "BE", ("A", "D"): "BE", ("A", "E"): "EP",
    ("B", "A"): "AP", ("B", "B"): "BE", ("B", "C"): "BE", ("B", "D"): "BE", ("B", "E"): "EP",
    ("C", "A"): "BE", ("C", "B"): "BE", ("C", "C"): "BE", ("C", "D"): "BE", ("C", "E"): "EP",
    ("D", "A"): "EP", ("D", "B"): "EP", ("D", "C"): "EP", ("D", "D"): "EP", ("D", "E"): "EP",
    ("E", "A"): "EP", ("E", "B"): "EP", ("E", "C"): "EP", ("E", "D"): "EP", ("E", "E"): "EP",
}

_CG_EGA_MATRIX_HYPER: dict[tuple[str, str], str] = {
    ("A", "A"): "AP", ("A", "B"): "AP", ("A", "C"): "AP", ("A", "D"): "BE", ("A", "E"): "EP",
    ("B", "A"): "AP", ("B", "B"): "AP", ("B", "C"): "BE", ("B", "D"): "BE", ("B", "E"): "EP",
    ("C", "A"): "BE", ("C", "B"): "BE", ("C", "C"): "BE", ("C", "D"): "BE", ("C", "E"): "EP",
    ("D", "A"): "BE", ("D", "B"): "BE", ("D", "C"): "BE", ("D", "D"): "EP", ("D", "E"): "EP",
    ("E", "A"): "EP", ("E", "B"): "EP", ("E", "C"): "EP", ("E", "D"): "EP", ("E", "E"): "EP",
}


def _cg_ega_lookup(zones_glycaemic: np.ndarray, p_zones: np.ndarray, r_zones: np.ndarray) -> np.ndarray:
    """Per-element CG-EGA classification using the zone-specific matrices."""
    out = np.empty(zones_glycaemic.shape, dtype="<U2")
    for i in range(zones_glycaemic.shape[0]):
        gz = zones_glycaemic[i]
        if gz == "hypo":
            mat = _CG_EGA_MATRIX_HYPO
        elif gz == "hyper":
            mat = _CG_EGA_MATRIX_HYPER
        else:
            mat = _CG_EGA_MATRIX_TIR
        out[i] = mat[(str(p_zones[i]), str(r_zones[i]))]
    return out


def compute_cg_ega_arrays(
    y_true_curr: np.ndarray,
    y_true_lag: np.ndarray,
    y_pred_curr: np.ndarray,
    y_pred_lag: np.ndarray,
    rate_window_min: float = float(CG_EGA_RATE_WINDOW_MIN),
) -> dict[str, np.ndarray]:
    """Compute per-sample CG-EGA labels from current and lagged glucose values.

    Parameters
    ----------
    y_true_curr, y_pred_curr
        Reference and predicted glucose at the target time (mg/dL).
    y_true_lag, y_pred_lag
        Reference and predicted glucose one rate-window earlier
        (15 minutes earlier for the HUPA 5-min grid, i.e. lag-3).
    rate_window_min
        Width of the rate-of-change window in minutes; defaults to
        ``CG_EGA_RATE_WINDOW_MIN``.

    Returns
    -------
    dict
        Keys: ``ref_rate``, ``pred_rate`` (mg/dL/min); ``point_zone``,
        ``rate_zone`` (Clarke A/B/C/D/E and Rate A/B/C/D/E single-char
        labels); ``glycaemic_zone`` (hypo/tir/hyper of the reference);
        ``cg_ega`` (AP/BE/EP per Kovatchev 2004).
    """
    ref_curr = np.asarray(y_true_curr, dtype=float)
    pred_curr = np.asarray(y_pred_curr, dtype=float)
    ref_lag = np.asarray(y_true_lag, dtype=float)
    pred_lag = np.asarray(y_pred_lag, dtype=float)
    if not (ref_curr.shape == pred_curr.shape == ref_lag.shape == pred_lag.shape):
        raise ValueError("all four arrays must share the same shape")
    if rate_window_min <= 0:
        raise ValueError("rate_window_min must be positive")

    ref_rate = (ref_curr - ref_lag) / float(rate_window_min)
    pred_rate = (pred_curr - pred_lag) / float(rate_window_min)
    p_zone = clarke_eg_zones(ref_curr, pred_curr)
    r_zone = _rate_zones(ref_rate, pred_rate)
    g_zone = zone_of(ref_curr)
    cg = _cg_ega_lookup(g_zone, p_zone, r_zone)
    return {
        "ref_rate": ref_rate,
        "pred_rate": pred_rate,
        "point_zone": p_zone,
        "rate_zone": r_zone,
        "glycaemic_zone": g_zone,
        "cg_ega": cg,
    }


def cg_ega_from_predictions(
    df: pd.DataFrame,
    group_keys: tuple[str, ...] = ("model", "split", "participant_id", "horizon_min"),
    sort_key: str = "sample_idx",
    rate_lag_steps: int = CG_EGA_RATE_LAG_STEPS,
    sample_step_min: int = 5,
) -> pd.DataFrame:
    """Attach CG-EGA labels to a tall master-predictions DataFrame.

    Assumes one row per ``(model, split, participant_id, horizon_min,
    sample_idx)`` with ``y_true`` and ``y_pred`` columns and that consecutive
    rows inside each ``group_keys`` group differ in ``sort_key`` by exactly
    one step (val/test stride = 1 in the HUPA pipeline). Rows whose lagged
    counterpart is missing or whose ``sort_key`` gap is larger than
    ``rate_lag_steps`` are dropped from the CG-EGA output.

    Returns a new DataFrame containing the input columns plus
    ``ref_rate``, ``pred_rate``, ``point_zone``, ``rate_zone``,
    ``glycaemic_zone`` (CG-EGA-internal, equal to the cohort
    ``zone`` column at the current time), and ``cg_ega``.
    """
    required = {"y_true", "y_pred", sort_key, *group_keys}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"input DataFrame missing required columns: {sorted(missing)}")

    df_sorted = df.sort_values(list(group_keys) + [sort_key]).reset_index(drop=True)
    grp = df_sorted.groupby(list(group_keys), sort=False)
    df_sorted["__y_true_lag"] = grp["y_true"].shift(int(rate_lag_steps))
    df_sorted["__y_pred_lag"] = grp["y_pred"].shift(int(rate_lag_steps))
    df_sorted["__sort_lag"] = grp[sort_key].shift(int(rate_lag_steps))
    # Drop rows where the lag is absent or the spacing is not exactly
    # rate_lag_steps (defends against accidental gaps in sample_idx).
    valid = (
        df_sorted["__sort_lag"].notna()
        & ((df_sorted[sort_key] - df_sorted["__sort_lag"]).astype("Int64") == int(rate_lag_steps))
        & df_sorted["__y_true_lag"].notna()
        & df_sorted["__y_pred_lag"].notna()
    )
    keep = df_sorted[valid].copy()
    if keep.empty:
        return keep.drop(columns=["__y_true_lag", "__y_pred_lag", "__sort_lag"])
    arrays = compute_cg_ega_arrays(
        y_true_curr=keep["y_true"].to_numpy(dtype=float),
        y_true_lag=keep["__y_true_lag"].to_numpy(dtype=float),
        y_pred_curr=keep["y_pred"].to_numpy(dtype=float),
        y_pred_lag=keep["__y_pred_lag"].to_numpy(dtype=float),
        rate_window_min=float(rate_lag_steps * sample_step_min),
    )
    for col in ("ref_rate", "pred_rate"):
        keep[col] = arrays[col].astype(np.float32)
    keep["point_zone"] = arrays["point_zone"]
    keep["rate_zone"] = arrays["rate_zone"]
    keep["glycaemic_zone"] = arrays["glycaemic_zone"]
    keep["cg_ega"] = arrays["cg_ega"]
    return keep.drop(columns=["__y_true_lag", "__y_pred_lag", "__sort_lag"]).reset_index(drop=True)


def cg_ega_summary(
    cg_df: pd.DataFrame,
    group_cols: tuple[str, ...] = ("model", "split", "horizon_min"),
    include_zone: bool = True,
) -> pd.DataFrame:
    """Aggregate AP / BE / EP percentages per group (optionally per zone).

    Parameters
    ----------
    cg_df
        Output of :func:`cg_ega_from_predictions` (must contain ``cg_ega``
        and, if ``include_zone=True``, ``glycaemic_zone``).
    group_cols
        Grouping keys for the aggregate (e.g. ``(model, split,
        horizon_min)`` for the master Step 5/6 comparison).
    include_zone
        If True, also stratify by ``glycaemic_zone`` (hypo / tir / hyper)
        so the table can be read as model × split × horizon × zone.
    """
    keys = list(group_cols) + (["glycaemic_zone"] if include_zone else [])
    counts = (
        cg_df.assign(_one=1)
        .groupby(keys + ["cg_ega"], as_index=False)["_one"].sum()
        .rename(columns={"_one": "n"})
    )
    totals = counts.groupby(keys, as_index=False)["n"].sum().rename(columns={"n": "n_total"})
    out = counts.merge(totals, on=keys)
    out["pct"] = 100.0 * out["n"] / out["n_total"].clip(lower=1)
    # Pivot to wide so each (group) row has AP_pct / BE_pct / EP_pct.
    wide_pct = (
        out.pivot_table(index=keys, columns="cg_ega", values="pct", fill_value=0.0)
        .add_suffix("_pct")
        .reset_index()
    )
    wide_n = (
        out.pivot_table(index=keys, columns="cg_ega", values="n", fill_value=0)
        .add_suffix("_n")
        .reset_index()
    )
    summary = wide_pct.merge(wide_n, on=keys)
    summary = summary.merge(totals, on=keys)
    # Ensure every expected column exists even if a class never appeared.
    for cls in CG_EGA_CLASSES:
        if f"{cls}_pct" not in summary.columns:
            summary[f"{cls}_pct"] = 0.0
        if f"{cls}_n" not in summary.columns:
            summary[f"{cls}_n"] = 0
    ordered = list(keys) + [
        "AP_pct", "BE_pct", "EP_pct",
        "AP_n", "BE_n", "EP_n",
        "n_total",
    ]
    return summary[ordered]


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
