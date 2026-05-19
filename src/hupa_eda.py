"""Meaningful EDA for the HUPA-UCM glucose forecasting pipeline.

The script is intentionally evidence-oriented: each output table answers a
specific modelling question required by skills/SKILL.md Step 2.

Run from project root:
    python src/hupa_eda.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_loading import load_hupa_all, load_patient_characteristics


HORIZONS = {"30m": 6, "60m": 12, "90m": 18}
LOOKBACKS = {"120m": 24, "180m": 36}


def ensure_dirs(base_path: Path) -> dict[str, Path]:
    paths = {
        "interim": base_path / "data" / "interim",
        "tables": base_path / "outputs" / "tables",
        "figures": base_path / "outputs" / "figures",
        "outputs": base_path / "outputs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"])
    out["date"] = out["time"].dt.date
    out["hour"] = out["time"].dt.hour + out["time"].dt.minute / 60.0
    out["dayofweek"] = out["time"].dt.dayofweek
    out["glucose_zone"] = np.select(
        [out["glucose"] < 70, out["glucose"] > 180],
        ["hypo", "hyper"],
        default="in_range",
    )
    return out


def structural_quality_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pid, g in df.groupby("participant_id", sort=True):
        gt = g.sort_values("time")
        dt = gt["time"].diff().dt.total_seconds().dropna()
        rows.append(
            {
                "participant_id": pid,
                "n_rows": len(gt),
                "start": gt["time"].min(),
                "end": gt["time"].max(),
                "duration_days": len(gt) * 5 / 1440,
                "missing_total": int(gt.isna().sum().sum()),
                "duplicate_timestamps": int(gt["time"].duplicated().sum()),
                "out_of_order_steps": int((g["time"].diff().dt.total_seconds().dropna() < 0).sum()),
                "strict_5min_pct": 100.0 * float((dt == 300).mean()) if len(dt) else np.nan,
                "glucose_mean": gt["glucose"].mean(),
                "glucose_std": gt["glucose"].std(),
                "glucose_min": gt["glucose"].min(),
                "glucose_max": gt["glucose"].max(),
                "hypo_pct": 100.0 * (gt["glucose"] < 70).mean(),
                "tir_pct": 100.0 * ((gt["glucose"] >= 70) & (gt["glucose"] <= 180)).mean(),
                "hyper_pct": 100.0 * (gt["glucose"] > 180).mean(),
                "low_cap_pct": 100.0 * (gt["glucose"] == 40).mean(),
                "high_extreme_pct": 100.0 * (gt["glucose"] > 400).mean(),
                "basal_positive_pct": 100.0 * (gt["basal_rate"] > 0).mean(),
                "bolus_event_count": int((gt["bolus_volume_delivered"] > 0).sum()),
                "carb_event_count": int((gt["carb_input"] > 0).sum()),
                "steps_active_pct": 100.0 * (gt["steps"] > 0).mean(),
            }
        )
    return pd.DataFrame(rows)


def glucose_velocity_summary(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for pid, g in df.groupby("participant_id", sort=True):
        gt = g.sort_values("time").copy()
        gt["velocity_mgdl_per_5m"] = gt["glucose"].diff()
        gt["abs_velocity"] = gt["velocity_mgdl_per_5m"].abs()
        parts.append(gt[["participant_id", "velocity_mgdl_per_5m", "abs_velocity"]])
    v = pd.concat(parts, ignore_index=True).dropna()
    return pd.DataFrame(
        [
            {
                "mean_velocity": v["velocity_mgdl_per_5m"].mean(),
                "std_velocity": v["velocity_mgdl_per_5m"].std(),
                "p50_abs_velocity": v["abs_velocity"].quantile(0.50),
                "p90_abs_velocity": v["abs_velocity"].quantile(0.90),
                "p95_abs_velocity": v["abs_velocity"].quantile(0.95),
                "p99_abs_velocity": v["abs_velocity"].quantile(0.99),
            }
        ]
    )


def acf_pacf_tables(df: pd.DataFrame, max_lag: int = 72) -> tuple[pd.DataFrame, pd.DataFrame]:
    acf_rows = []
    pacf_rows = []
    try:
        from statsmodels.tsa.stattools import pacf
    except Exception:
        pacf = None

    for pid, g in df.groupby("participant_id", sort=True):
        series = g.sort_values("time")["glucose"].astype(float).reset_index(drop=True)
        for lag in range(1, max_lag + 1):
            acf_rows.append(
                {
                    "participant_id": pid,
                    "lag_steps": lag,
                    "lag_minutes": lag * 5,
                    "acf": series.autocorr(lag=lag),
                }
            )
        if pacf is not None:
            sample = series.iloc[: min(len(series), 20000)]
            vals = pacf(sample, nlags=max_lag, method="ywm")
            for lag in range(1, max_lag + 1):
                pacf_rows.append(
                    {
                        "participant_id": pid,
                        "lag_steps": lag,
                        "lag_minutes": lag * 5,
                        "pacf": vals[lag],
                    }
                )
    return pd.DataFrame(acf_rows), pd.DataFrame(pacf_rows)


def circadian_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("hour", as_index=False)
        .agg(
            glucose_mean=("glucose", "mean"),
            glucose_median=("glucose", "median"),
            hypo_pct=("glucose", lambda s: 100.0 * (s < 70).mean()),
            hyper_pct=("glucose", lambda s: 100.0 * (s > 180).mean()),
            n=("glucose", "size"),
        )
        .sort_values("hour")
    )


def _event_indices_with_coevent_mask(
    df_patient: pd.DataFrame,
    event_col: str,
    coevent_col: str,
    event_thr: float = 0.0,
    coevent_thr: float = 0.0,
    coevent_window_steps: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (with_coevent_idx, alone_idx) for one patient.

    A `event_col` event at index i is classified as 'with coevent' if any
    `coevent_col > coevent_thr` occurs within +/- coevent_window_steps of i.
    coevent_window_steps=3 corresponds to +/- 15 minutes at 5-min sampling.
    """
    n = len(df_patient)
    ev_vals = df_patient[event_col].values
    co_vals = df_patient[coevent_col].values
    ev_mask = ev_vals > event_thr
    ev_idx = np.where(ev_mask)[0]
    if len(ev_idx) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    co_mask_window = np.zeros(n, dtype=bool)
    # convolve coevent>thr indicator over +/- window
    co_bool = co_vals > coevent_thr
    for shift in range(-coevent_window_steps, coevent_window_steps + 1):
        if shift == 0:
            co_mask_window |= co_bool
        elif shift > 0:
            co_mask_window[:n - shift] |= co_bool[shift:]
        else:
            co_mask_window[-shift:] |= co_bool[:n + shift]
    with_co = ev_idx[co_mask_window[ev_idx]]
    alone = ev_idx[~co_mask_window[ev_idx]]
    return with_co, alone


def peri_event_summary_by_subtype(
    df: pd.DataFrame,
    horizons_steps: tuple[int, ...] = (6, 12, 18, 24),
    rng_seed: int = 17,
) -> pd.DataFrame:
    """Peri-event glucose response split by physiological subtype.

    Splits four event categories:
      * bolus_meal: bolus event with any carb_input>0 within +/- 15 min
      * bolus_correction: bolus event with no carb in that window
      * carb_meal: carb event with any bolus>0 within +/- 15 min
      * carb_solo: carb event with no bolus in that window

    For each subtype, computes mean Δglucose at each horizon vs a
    same-patient random non-event control. This isolates the pure insulin
    effect (correction bolus) from the blended meal effect.
    """
    rng = np.random.default_rng(rng_seed)
    subtypes = ["bolus_meal", "bolus_correction", "carb_meal", "carb_solo"]
    treat = {(s, h): [] for s in subtypes for h in horizons_steps}
    control = {(s, h): [] for s in subtypes for h in horizons_steps}
    n_events_total = {s: 0 for s in subtypes}

    for pid, g in df.groupby("participant_id", sort=True):
        gt = g.sort_values("time").reset_index(drop=True)
        glu = gt["glucose"].values
        n = len(glu)
        if n == 0:
            continue
        non_event_mask = (
            (gt["bolus_volume_delivered"].values == 0)
            & (gt["carb_input"].values == 0)
        )
        non_idx = np.where(non_event_mask)[0]

        bolus_meal, bolus_corr = _event_indices_with_coevent_mask(
            gt, "bolus_volume_delivered", "carb_input"
        )
        carb_meal, carb_solo = _event_indices_with_coevent_mask(
            gt, "carb_input", "bolus_volume_delivered"
        )
        subtype_idx = {
            "bolus_meal": bolus_meal,
            "bolus_correction": bolus_corr,
            "carb_meal": carb_meal,
            "carb_solo": carb_solo,
        }

        for sname, sidx in subtype_idx.items():
            if len(sidx) == 0 or len(non_idx) == 0:
                continue
            n_events_total[sname] += len(sidx)
            ctrl = rng.choice(non_idx, size=min(len(sidx), len(non_idx)), replace=False)
            for h in horizons_steps:
                ti = sidx[sidx + h < n]
                ci = ctrl[ctrl + h < n]
                treat[(sname, h)].extend((glu[ti + h] - glu[ti]).tolist())
                control[(sname, h)].extend((glu[ci + h] - glu[ci]).tolist())

    rows = []
    for sname in subtypes:
        for h in horizons_steps:
            t = np.asarray(treat[(sname, h)], dtype=float)
            c = np.asarray(control[(sname, h)], dtype=float)
            rows.append(
                {
                    "event_subtype": sname,
                    "horizon_steps": h,
                    "horizon_minutes": h * 5,
                    "n_events": int(len(t)),
                    "mean_delta_event": float(np.mean(t)) if len(t) else np.nan,
                    "std_delta_event": float(np.std(t, ddof=1)) if len(t) > 1 else np.nan,
                    "n_control": int(len(c)),
                    "mean_delta_control": float(np.mean(c)) if len(c) else np.nan,
                    "mean_delta_diff": float(np.mean(t) - np.mean(c))
                    if (len(t) and len(c))
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def peri_event_per_patient_stats(
    df: pd.DataFrame,
    horizons_steps: tuple[int, ...] = (6, 12, 18, 24),
    rng_seed: int = 17,
) -> pd.DataFrame:
    """Per-patient peri-event mean Δglucose, then aggregated as mean ± SD across patients.

    The pooled peri-event analysis lets one long patient (HUPA0027 = 53% of
    events) dominate. This function computes the mean Δglucose **per patient
    first**, then reports the cross-patient mean, SD, and the number of
    patients that actually contributed any events for the given type.
    """
    rng = np.random.default_rng(rng_seed)
    events = {
        "bolus_volume_delivered": 0.0,
        "carb_input": 0.0,
        "steps": 99.0,
    }
    rows_per_patient: dict[tuple[str, int], list[float]] = {}

    for pid, g in df.groupby("participant_id", sort=True):
        gt = g.sort_values("time").reset_index(drop=True)
        glu = gt["glucose"].values
        n = len(glu)
        for ev, thr in events.items():
            ev_mask = (gt[ev].values > thr)
            ev_idx = np.where(ev_mask)[0]
            non_idx = np.where(~ev_mask)[0]
            if len(ev_idx) == 0 or len(non_idx) == 0:
                continue
            ctrl = rng.choice(non_idx, size=min(len(ev_idx), len(non_idx)), replace=False)
            for h in horizons_steps:
                ti = ev_idx[ev_idx + h < n]
                ci = ctrl[ctrl + h < n]
                if len(ti) == 0:
                    continue
                d_event = float(np.mean(glu[ti + h] - glu[ti]))
                d_ctrl = float(np.mean(glu[ci + h] - glu[ci])) if len(ci) else 0.0
                rows_per_patient.setdefault((ev, h), []).append(d_event - d_ctrl)

    out = []
    for (ev, h), vals in rows_per_patient.items():
        arr = np.asarray(vals, dtype=float)
        out.append(
            {
                "event_type": ev,
                "horizon_steps": h,
                "horizon_minutes": h * 5,
                "n_patients": len(arr),
                "patient_mean_diff": float(np.mean(arr)),
                "patient_std_diff": float(np.std(arr, ddof=1)) if len(arr) > 1 else np.nan,
                "patient_min_diff": float(np.min(arr)),
                "patient_max_diff": float(np.max(arr)),
            }
        )
    return pd.DataFrame(out).sort_values(["event_type", "horizon_minutes"]).reset_index(drop=True)


def velocity_by_zone_filtered(df: pd.DataFrame) -> pd.DataFrame:
    """Velocity by glycaemic zone, excluding sensor-cap rows.

    The naive `velocity_by_zone_table` includes records where glucose is
    censored at 40 or above 400. Stuck-at-40 segments produce velocity=0
    bursts that deflate the hypo zone quantiles. This filtered variant
    excludes those rows and is compared against the unfiltered version to
    quantify the artifact.
    """
    parts = []
    for pid, g in df.groupby("participant_id", sort=True):
        gt = g.sort_values("time").copy()
        gt["velocity"] = gt["glucose"].diff()
        gt["abs_velocity"] = gt["velocity"].abs()
        keep = (gt["glucose"] != 40) & (gt["glucose"] <= 400)
        gt = gt[keep]
        parts.append(gt[["glucose_zone", "velocity", "abs_velocity"]])
    v = pd.concat(parts, ignore_index=True).dropna()
    rows = []
    for zone, gv in v.groupby("glucose_zone", sort=True):
        rows.append(
            {
                "zone": zone,
                "n": int(len(gv)),
                "p50_abs_velocity": gv["abs_velocity"].quantile(0.50),
                "p90_abs_velocity": gv["abs_velocity"].quantile(0.90),
                "p95_abs_velocity": gv["abs_velocity"].quantile(0.95),
                "p99_abs_velocity": gv["abs_velocity"].quantile(0.99),
            }
        )
    return pd.DataFrame(rows)


def sequence_feasibility(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pid, g in df.groupby("participant_id", sort=True):
        n = len(g)
        for lb_name, lb_steps in LOOKBACKS.items():
            usable = max(0, n - lb_steps - max(HORIZONS.values()) + 1)
            rows.append(
                {
                    "participant_id": pid,
                    "lookback": lb_name,
                    "lookback_steps": lb_steps,
                    "max_horizon_steps": max(HORIZONS.values()),
                    "usable_sequences_stride1": usable,
                    "usable_sequences_stride6": int(np.ceil(usable / 6)) if usable else 0,
                    "usable_sequences_stride12": int(np.ceil(usable / 12)) if usable else 0,
                }
            )
    return pd.DataFrame(rows)


def glucose_distribution_table(df: pd.DataFrame) -> pd.DataFrame:
    """Overall glucose distribution: histogram bin counts plus pooled quantiles.

    Returns a long table with `bin_left`, `bin_right`, `count`, `pct` for an
    overall histogram. Bin width is 10 mg/dL, range 40-450.
    """
    bins = np.arange(40, 460, 10)
    counts, edges = np.histogram(df["glucose"].dropna().values, bins=bins)
    total = counts.sum()
    return pd.DataFrame(
        {
            "bin_left": edges[:-1],
            "bin_right": edges[1:],
            "count": counts,
            "pct": 100.0 * counts / max(total, 1),
        }
    )


def velocity_by_zone_table(df: pd.DataFrame) -> pd.DataFrame:
    """Velocity (5-min glucose change) stratified by current glycaemic zone.

    Justifies whether the model needs zone-specific evaluation: if hypo/hyper
    samples exhibit different velocity distributions, pooled RMSE may hide
    safety-relevant errors.
    """
    parts = []
    for pid, g in df.groupby("participant_id", sort=True):
        gt = g.sort_values("time").copy()
        gt["velocity"] = gt["glucose"].diff()
        gt["abs_velocity"] = gt["velocity"].abs()
        parts.append(gt[["glucose_zone", "velocity", "abs_velocity"]])
    v = pd.concat(parts, ignore_index=True).dropna()
    rows = []
    for zone, gv in v.groupby("glucose_zone", sort=True):
        rows.append(
            {
                "zone": zone,
                "n": int(len(gv)),
                "mean_velocity": gv["velocity"].mean(),
                "std_velocity": gv["velocity"].std(),
                "p50_abs_velocity": gv["abs_velocity"].quantile(0.50),
                "p90_abs_velocity": gv["abs_velocity"].quantile(0.90),
                "p95_abs_velocity": gv["abs_velocity"].quantile(0.95),
                "p99_abs_velocity": gv["abs_velocity"].quantile(0.99),
            }
        )
    return pd.DataFrame(rows)


def dayofweek_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Glucose distribution by ISO day of week (0=Monday ... 6=Sunday).

    Justifies cyclical day-of-week features required by CLAUDE.md schema:
    if zone proportions vary materially across the week, dayofweek_sin/cos
    add information beyond hour-of-day.
    """
    out = (
        df.groupby("dayofweek", as_index=False)
        .agg(
            glucose_mean=("glucose", "mean"),
            glucose_median=("glucose", "median"),
            hypo_pct=("glucose", lambda s: 100.0 * (s < 70).mean()),
            tir_pct=("glucose", lambda s: 100.0 * ((s >= 70) & (s <= 180)).mean()),
            hyper_pct=("glucose", lambda s: 100.0 * (s > 180).mean()),
            n=("glucose", "size"),
        )
        .sort_values("dayofweek")
    )
    out["dayofweek_label"] = out["dayofweek"].map(
        {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    )
    return out


def per_patient_acf_halflife(acf_df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """First lag (minutes) at which each patient's ACF falls below `threshold`.

    Heterogeneity in the half-life across patients is a direct argument for
    a patient-conditioned model (static embedding branch) rather than a
    single pooled model.
    """
    rows = []
    for pid, g in acf_df.groupby("participant_id"):
        g = g.sort_values("lag_minutes")
        below = g[g["acf"] < threshold]
        first_below = int(below["lag_minutes"].iloc[0]) if len(below) else int(g["lag_minutes"].iloc[-1])
        rows.append(
            {
                "participant_id": pid,
                "acf_threshold": threshold,
                "first_lag_below_threshold_minutes": first_below,
                "acf_at_120m": float(g.loc[g["lag_minutes"] == 120, "acf"].iloc[0])
                if (g["lag_minutes"] == 120).any()
                else np.nan,
                "acf_at_180m": float(g.loc[g["lag_minutes"] == 180, "acf"].iloc[0])
                if (g["lag_minutes"] == 180).any()
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def peri_event_summary(
    df: pd.DataFrame,
    event_col: str,
    threshold: float = 0.0,
    horizons_steps: tuple[int, ...] = (6, 12, 18, 24),
    rng_seed: int = 17,
) -> pd.DataFrame:
    """Mean glucose change (mg/dL) in the H minutes following an event.

    For sparse zero-inflated streams (bolus, carb, large step bursts), a
    Pearson correlation between the raw stream and future glucose is near
    zero by construction. The physiologically meaningful question is
    "given an event at time t, what does glucose do at t+H?" This function
    answers that, plus a same-patient non-event control to filter out drift.

    Parameters
    ----------
    event_col
        Column to threshold (e.g. ``"bolus_volume_delivered"``).
    threshold
        Event defined as ``df[event_col] > threshold``.
    horizons_steps
        Future horizons in 5-min steps (default 30/60/90/120 min).
    rng_seed
        Seed for the matched non-event control sample.
    """
    rng = np.random.default_rng(rng_seed)
    horizon_minutes = [h * 5 for h in horizons_steps]
    treat_rows = {h: [] for h in horizons_steps}
    control_rows = {h: [] for h in horizons_steps}

    for pid, g in df.groupby("participant_id", sort=True):
        gt = g.sort_values("time").reset_index(drop=True)
        glu = gt["glucose"].values
        ev_mask = (gt[event_col] > threshold).values
        if ev_mask.sum() == 0:
            continue
        ev_idx = np.where(ev_mask)[0]
        non_idx = np.where(~ev_mask)[0]
        if len(non_idx) == 0:
            continue
        ctrl_idx = rng.choice(non_idx, size=min(len(ev_idx), len(non_idx)), replace=False)
        for h in horizons_steps:
            t_idx = ev_idx[ev_idx + h < len(glu)]
            c_idx = ctrl_idx[ctrl_idx + h < len(glu)]
            treat_rows[h].extend((glu[t_idx + h] - glu[t_idx]).tolist())
            control_rows[h].extend((glu[c_idx + h] - glu[c_idx]).tolist())

    rows = []
    for h in horizons_steps:
        t = np.asarray(treat_rows[h], dtype=float)
        c = np.asarray(control_rows[h], dtype=float)
        rows.append(
            {
                "event_type": event_col,
                "threshold": threshold,
                "horizon_steps": h,
                "horizon_minutes": h * 5,
                "n_events": int(len(t)),
                "mean_delta_event": float(np.mean(t)) if len(t) else np.nan,
                "std_delta_event": float(np.std(t, ddof=1)) if len(t) > 1 else np.nan,
                "median_delta_event": float(np.median(t)) if len(t) else np.nan,
                "n_control": int(len(c)),
                "mean_delta_control": float(np.mean(c)) if len(c) else np.nan,
                "std_delta_control": float(np.std(c, ddof=1)) if len(c) > 1 else np.nan,
                "mean_delta_diff": float(np.mean(t) - np.mean(c)) if (len(t) and len(c)) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def all_peri_event_summaries(df: pd.DataFrame) -> pd.DataFrame:
    """Bundle peri-event analysis for bolus, carb, and high-activity events."""
    bolus = peri_event_summary(df, "bolus_volume_delivered", threshold=0.0)
    carb = peri_event_summary(df, "carb_input", threshold=0.0)
    # High-activity threshold: >=100 steps in a single 5-min bin
    # (= >=20 steps/min, brisk walking). Captures purposeful activity rather
    # than incidental movement (steps>0 is 29% of bins, too liberal).
    steps_high = peri_event_summary(df, "steps", threshold=99.0)
    return pd.concat([bolus, carb, steps_high], ignore_index=True)


def build_overall_summary(
    df: pd.DataFrame,
    patient_summary: pd.DataFrame,
    acf_df: pd.DataFrame,
    pacf_df: pd.DataFrame,
    seq_df: pd.DataFrame,
    halflife_df: pd.DataFrame | None = None,
    peri_event_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    acf_mean = acf_df.groupby("lag_minutes")["acf"].mean()
    acf_below_02 = acf_mean[acf_mean < 0.2]
    pacf_mean = pacf_df.groupby("lag_minutes")["pacf"].mean() if not pacf_df.empty else pd.Series(dtype=float)
    pacf_abs_below_01 = pacf_mean[pacf_mean.abs() < 0.1] if not pacf_mean.empty else pd.Series(dtype=float)
    patient_avg_hypo = float(patient_summary["hypo_pct"].mean())
    patient_avg_tir = float(patient_summary["tir_pct"].mean())
    patient_avg_hyper = float(patient_summary["hyper_pct"].mean())
    strict_5min_min = float(patient_summary["strict_5min_pct"].min())
    halflife_min = halflife_min_minutes = halflife_max_minutes = np.nan
    if halflife_df is not None and len(halflife_df):
        halflife_min_minutes = int(halflife_df["first_lag_below_threshold_minutes"].min())
        halflife_max_minutes = int(halflife_df["first_lag_below_threshold_minutes"].max())
    bolus_60m = carb_60m = np.nan
    if peri_event_df is not None and len(peri_event_df):
        b = peri_event_df[(peri_event_df["event_type"] == "bolus_volume_delivered") & (peri_event_df["horizon_minutes"] == 60)]
        c = peri_event_df[(peri_event_df["event_type"] == "carb_input") & (peri_event_df["horizon_minutes"] == 60)]
        bolus_60m = float(b["mean_delta_diff"].iloc[0]) if len(b) else np.nan
        carb_60m = float(c["mean_delta_diff"].iloc[0]) if len(c) else np.nan
    return pd.DataFrame(
        [
            {
                "n_rows": len(df),
                "n_participants": df["participant_id"].nunique(),
                "glucose_min": df["glucose"].min(),
                "glucose_max": df["glucose"].max(),
                "hypo_pct": 100.0 * (df["glucose"] < 70).mean(),
                "tir_pct": 100.0 * ((df["glucose"] >= 70) & (df["glucose"] <= 180)).mean(),
                "hyper_pct": 100.0 * (df["glucose"] > 180).mean(),
                "patient_avg_hypo_pct": patient_avg_hypo,
                "patient_avg_tir_pct": patient_avg_tir,
                "patient_avg_hyper_pct": patient_avg_hyper,
                "low_cap_pct": 100.0 * (df["glucose"] == 40).mean(),
                "high_extreme_pct": 100.0 * (df["glucose"] > 400).mean(),
                "median_duration_days": patient_summary["duration_days"].median(),
                "top_patient_row_share_pct": 100.0
                * patient_summary["n_rows"].max()
                / patient_summary["n_rows"].sum(),
                "min_strict_5min_pct": strict_5min_min,
                "mean_acf_lag_120m": float(acf_mean.get(120, np.nan)),
                "mean_acf_lag_180m": float(acf_mean.get(180, np.nan)),
                "first_mean_acf_below_0_2_minutes": int(acf_below_02.index.min())
                if len(acf_below_02)
                else np.nan,
                "first_abs_mean_pacf_below_0_1_minutes": int(pacf_abs_below_01.index.min())
                if len(pacf_abs_below_01)
                else np.nan,
                "patient_acf_halflife_min_minutes": halflife_min_minutes,
                "patient_acf_halflife_max_minutes": halflife_max_minutes,
                "bolus_minus_control_delta_60m": bolus_60m,
                "carb_minus_control_delta_60m": carb_60m,
                "usable_sequences_120m_stride1": int(
                    seq_df.loc[seq_df["lookback"] == "120m", "usable_sequences_stride1"].sum()
                ),
                "usable_sequences_180m_stride1": int(
                    seq_df.loc[seq_df["lookback"] == "180m", "usable_sequences_stride1"].sum()
                ),
            }
        ]
    )


def make_overview_figure(
    patient_summary: pd.DataFrame,
    acf_df: pd.DataFrame,
    circadian_df: pd.DataFrame,
    peri_event_df: pd.DataFrame,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].hist(patient_summary["glucose_mean"], bins=12, color="#4C78A8", edgecolor="white")
    axes[0, 0].set_title("Per-patient mean glucose")
    axes[0, 0].set_xlabel("Mean glucose (mg/dL)")
    axes[0, 0].set_ylabel("Participants")

    acf_mean = acf_df.groupby("lag_minutes")["acf"].mean().reset_index()
    axes[0, 1].plot(acf_mean["lag_minutes"], acf_mean["acf"], color="#F58518")
    axes[0, 1].axhline(0.2, color="black", linewidth=0.8, linestyle="--")
    axes[0, 1].set_title("Mean glucose autocorrelation")
    axes[0, 1].set_xlabel("Lag (minutes)")
    axes[0, 1].set_ylabel("ACF")

    axes[1, 0].plot(circadian_df["hour"], circadian_df["glucose_mean"], color="#54A24B")
    axes[1, 0].fill_between(
        circadian_df["hour"],
        circadian_df["glucose_mean"],
        circadian_df["glucose_median"],
        color="#54A24B",
        alpha=0.2,
    )
    axes[1, 0].set_title("Circadian glucose profile")
    axes[1, 0].set_xlabel("Hour of day")
    axes[1, 0].set_ylabel("Glucose (mg/dL)")

    # Peri-event mini view: pooled difference (event - control) for the three
    # candidate modalities at 30/60/90/120 min horizons. Replaces the former
    # raw-Pearson panel, which is structurally near zero on sparse streams.
    event_labels = {
        "bolus_volume_delivered": "bolus",
        "carb_input": "carb",
        "steps": "steps (>=100/5min)",
    }
    colors = {"bolus_volume_delivered": "#1F77B4",
              "carb_input": "#FF7F0E",
              "steps": "#2CA02C"}
    for ev, lbl in event_labels.items():
        sub = peri_event_df[peri_event_df["event_type"] == ev].sort_values("horizon_minutes")
        if not len(sub):
            continue
        axes[1, 1].plot(sub["horizon_minutes"], sub["mean_delta_diff"],
                        marker="o", label=lbl, color=colors[ev])
    axes[1, 1].axhline(0, color="black", linewidth=0.8)
    axes[1, 1].set_title("Peri-event Δglucose (event − same-patient control)")
    axes[1, 1].set_xlabel("Minutes after event")
    axes[1, 1].set_ylabel("Δglucose (mg/dL)")
    axes[1, 1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def make_individual_figures(
    df: pd.DataFrame,
    patient_summary: pd.DataFrame,
    acf_df: pd.DataFrame,
    circadian_df: pd.DataFrame,
    halflife_df: pd.DataFrame,
    dow_df: pd.DataFrame,
    velocity_zone_df: pd.DataFrame,
    peri_event_df: pd.DataFrame,
    peri_subtype_df: pd.DataFrame,
    figures_dir: Path,
) -> dict[str, Path]:
    outputs = {
        "figure_glucose_distribution": figures_dir / "02_eda_glucose_distribution.png",
        "figure_acf": figures_dir / "02_eda_acf.png",
        "figure_circadian": figures_dir / "02_eda_circadian_profile.png",
        "figure_per_patient_heterogeneity": figures_dir / "02_eda_per_patient_heterogeneity.png",
        "figure_peri_event": figures_dir / "02_eda_peri_event.png",
        "figure_peri_event_subtypes": figures_dir / "02_eda_peri_event_subtypes.png",
        "figure_dayofweek_velocity": figures_dir / "02_eda_dayofweek_velocity.png",
    }

    # Replacement: actual glucose distribution (overall histogram + zone bands)
    # plus per-patient mean as a secondary panel.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].hist(df["glucose"].dropna().values, bins=np.arange(40, 460, 10),
                 color="#4C78A8", edgecolor="white")
    axes[0].axvline(70, color="#D62728", linestyle="--", linewidth=1)
    axes[0].axvline(180, color="#FF7F0E", linestyle="--", linewidth=1)
    axes[0].set_title("Overall glucose distribution (all 309,392 records)")
    axes[0].set_xlabel("Glucose (mg/dL)")
    axes[0].set_ylabel("Records")
    axes[0].text(70, axes[0].get_ylim()[1] * 0.95, " hypo 70",
                 color="#D62728", fontsize=8, va="top")
    axes[0].text(180, axes[0].get_ylim()[1] * 0.95, " hyper 180",
                 color="#FF7F0E", fontsize=8, va="top")

    axes[1].hist(patient_summary["glucose_mean"], bins=12,
                 color="#A0CBE8", edgecolor="white")
    axes[1].set_title("Per-patient mean glucose (n=25)")
    axes[1].set_xlabel("Mean glucose (mg/dL)")
    axes[1].set_ylabel("Participants")
    fig.tight_layout()
    fig.savefig(outputs["figure_glucose_distribution"], dpi=160)
    plt.close(fig)

    acf_mean = acf_df.groupby("lag_minutes")["acf"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(acf_mean["lag_minutes"], acf_mean["acf"], color="#F58518")
    ax.axhline(0.2, color="black", linewidth=0.8, linestyle="--")
    ax.axvline(120, color="#666666", linewidth=0.8, linestyle=":")
    ax.axvline(180, color="#666666", linewidth=0.8, linestyle=":")
    ax.set_title("Mean glucose autocorrelation")
    ax.set_xlabel("Lag (minutes)")
    ax.set_ylabel("ACF")
    fig.tight_layout()
    fig.savefig(outputs["figure_acf"], dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(circadian_df["hour"], circadian_df["glucose_mean"], color="#54A24B", label="Mean")
    ax.plot(circadian_df["hour"], circadian_df["glucose_median"], color="#2F6B2F", linestyle="--", label="Median")
    ax.set_title("Circadian glucose profile")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Glucose (mg/dL)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outputs["figure_circadian"], dpi=160)
    plt.close(fig)

    # Per-patient heterogeneity: zone proportions per patient + ACF half-life.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ps_sorted = patient_summary.sort_values("glucose_mean")
    pids = ps_sorted["participant_id"].values
    x = np.arange(len(pids))
    axes[0].bar(x, ps_sorted["hypo_pct"], color="#D62728", label="hypo (<70)")
    axes[0].bar(x, ps_sorted["tir_pct"], bottom=ps_sorted["hypo_pct"],
                color="#2CA02C", label="in range")
    axes[0].bar(x, ps_sorted["hyper_pct"],
                bottom=ps_sorted["hypo_pct"] + ps_sorted["tir_pct"],
                color="#FF7F0E", label="hyper (>180)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([p.replace("HUPA", "").replace("P", "") for p in pids],
                            rotation=90, fontsize=7)
    axes[0].set_title("Glycaemic zone composition per patient (sorted by mean glucose)")
    axes[0].set_ylabel("% of records")
    axes[0].legend(fontsize=8, loc="lower right")
    axes[0].set_ylim(0, 100)

    hl_sorted = halflife_df.sort_values("first_lag_below_threshold_minutes")
    axes[1].barh(np.arange(len(hl_sorted)),
                 hl_sorted["first_lag_below_threshold_minutes"],
                 color="#4C78A8")
    axes[1].set_yticks(np.arange(len(hl_sorted)))
    axes[1].set_yticklabels(
        [p.replace("HUPA", "").replace("P", "") for p in hl_sorted["participant_id"]],
        fontsize=7,
    )
    axes[1].set_title("Per-patient glucose ACF half-life (first lag with ACF < 0.5)")
    axes[1].set_xlabel("Minutes")
    fig.tight_layout()
    fig.savefig(outputs["figure_per_patient_heterogeneity"], dpi=160)
    plt.close(fig)

    # Peri-event analysis: mean glucose change in the H minutes after an event.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    event_titles = {
        "bolus_volume_delivered": "After bolus insulin (any dose)",
        "carb_input": "After carbohydrate intake (any amount)",
        "steps": "After high-activity bin (>=100 steps in 5 min)",
    }
    for ax, ev in zip(axes, event_titles.keys()):
        sub = peri_event_df[peri_event_df["event_type"] == ev].sort_values("horizon_minutes")
        if len(sub) == 0:
            continue
        ax.errorbar(sub["horizon_minutes"], sub["mean_delta_event"],
                    yerr=sub["std_delta_event"] / np.sqrt(sub["n_events"].clip(lower=1)),
                    marker="o", color="#D62728", label="event-triggered",
                    capsize=3)
        ax.errorbar(sub["horizon_minutes"], sub["mean_delta_control"],
                    yerr=sub["std_delta_control"] / np.sqrt(sub["n_control"].clip(lower=1)),
                    marker="s", color="#7F7F7F", label="random control",
                    capsize=3)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_title(event_titles[ev], fontsize=10)
        ax.set_xlabel("Minutes after event")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Mean glucose change (mg/dL)")
    fig.suptitle("Peri-event glucose change: event-triggered vs same-patient control",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(outputs["figure_peri_event"], dpi=160)
    plt.close(fig)

    # Peri-event subtype: meal_bolus vs correction_bolus, meal_carb vs solo_carb.
    # Isolates pure-insulin and pure-meal effects that the pooled view conflates.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    sub_pairs = [
        ("Bolus: meal-coincident vs correction-only",
         [("bolus_meal", "#1F77B4"), ("bolus_correction", "#D62728")]),
        ("Carb: meal (with bolus) vs solo (no bolus nearby)",
         [("carb_meal", "#FF7F0E"), ("carb_solo", "#2CA02C")]),
    ]
    for ax, (title, items) in zip(axes, sub_pairs):
        for sname, color in items:
            sub = peri_subtype_df[peri_subtype_df["event_subtype"] == sname].sort_values("horizon_minutes")
            if not len(sub) or sub["n_events"].iloc[0] == 0:
                continue
            n = int(sub["n_events"].iloc[0])
            ax.errorbar(sub["horizon_minutes"], sub["mean_delta_diff"],
                        yerr=sub["std_delta_event"] / np.sqrt(sub["n_events"].clip(lower=1)),
                        marker="o", color=color,
                        label=f"{sname} (n={n})", capsize=3)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Minutes after event")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Δglucose vs same-patient control (mg/dL)")
    fig.suptitle("Peri-event subtype: separating pure-insulin and pure-meal responses",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(outputs["figure_peri_event_subtypes"], dpi=160)
    plt.close(fig)

    # Combined day-of-week + velocity-by-zone supplementary panel.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    axes[0].bar(dow_df["dayofweek_label"], dow_df["glucose_mean"], color="#4C78A8")
    axes[0].set_title("Mean glucose by day of week")
    axes[0].set_ylabel("Glucose (mg/dL)")
    ymin = float(dow_df["glucose_mean"].min())
    ymax = float(dow_df["glucose_mean"].max())
    axes[0].set_ylim(ymin - 5, ymax + 5)

    zones = ["hypo", "in_range", "hyper"]
    palette = {"hypo": "#D62728", "in_range": "#2CA02C", "hyper": "#FF7F0E"}
    metrics = ["p50_abs_velocity", "p90_abs_velocity", "p95_abs_velocity", "p99_abs_velocity"]
    metric_labels = ["P50", "P90", "P95", "P99"]
    x = np.arange(len(metrics))
    width = 0.27
    for i, z in enumerate(zones):
        row = velocity_zone_df[velocity_zone_df["zone"] == z]
        if not len(row):
            continue
        vals = [float(row[m].iloc[0]) for m in metrics]
        axes[1].bar(x + (i - 1) * width, vals, width=width,
                    color=palette[z], label=z)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metric_labels)
    axes[1].set_title("Abs. 5-min glucose velocity by glycaemic zone")
    axes[1].set_ylabel("|Δglucose| (mg/dL per 5 min)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outputs["figure_dayofweek_velocity"], dpi=160)
    plt.close(fig)

    return outputs


def write_text_summary(
    out_path: Path,
    overall: pd.DataFrame,
    patient_summary: pd.DataFrame,
    peri_event_df: pd.DataFrame,
    dow_df: pd.DataFrame,
    velocity_zone_df: pd.DataFrame,
    halflife_df: pd.DataFrame,
    peri_subtype_df: pd.DataFrame | None = None,
    peri_per_patient_df: pd.DataFrame | None = None,
    velocity_zone_filtered_df: pd.DataFrame | None = None,
) -> None:
    row = overall.iloc[0]
    top = patient_summary.sort_values("n_rows", ascending=False).head(3)
    lines = [
        "HUPA-UCM Step 2 EDA summary",
        "=" * 80,
        f"Participants: {int(row['n_participants'])}",
        f"Rows: {int(row['n_rows'])}",
        f"Glucose range: {row['glucose_min']:.1f}-{row['glucose_max']:.1f} mg/dL",
        f"Zones (row-weighted): hypo {row['hypo_pct']:.2f}%, TIR {row['tir_pct']:.2f}%, hyper {row['hyper_pct']:.2f}%",
        f"Zones (patient-averaged): hypo {row['patient_avg_hypo_pct']:.2f}%, TIR {row['patient_avg_tir_pct']:.2f}%, hyper {row['patient_avg_hyper_pct']:.2f}%",
        f"Censoring flags: glucose==40 {row['low_cap_pct']:.2f}%, glucose>400 {row['high_extreme_pct']:.2f}%",
        f"Median duration: {row['median_duration_days']:.2f} days",
        f"Top patient row share: {row['top_patient_row_share_pct']:.2f}%",
        f"Min strict 5-min step %: {row['min_strict_5min_pct']:.2f}",
        f"Mean ACF at 120 min: {row['mean_acf_lag_120m']:.3f}",
        f"Mean ACF at 180 min: {row['mean_acf_lag_180m']:.3f}",
        f"First mean ACF below 0.2: {row['first_mean_acf_below_0_2_minutes']} minutes",
        f"First abs mean PACF below 0.1: {row['first_abs_mean_pacf_below_0_1_minutes']} minutes",
        f"Patient ACF half-life (lag <0.5) range: {row['patient_acf_halflife_min_minutes']}-{row['patient_acf_halflife_max_minutes']} minutes",
        f"Usable stride-1 sequences, 120m lookback: {int(row['usable_sequences_120m_stride1'])}",
        f"Usable stride-1 sequences, 180m lookback: {int(row['usable_sequences_180m_stride1'])}",
        "",
        "Top 3 participants by row count:",
    ]
    for _, r in top.iterrows():
        lines.append(f"- {r['participant_id']}: {int(r['n_rows'])} rows ({r['duration_days']:.1f} days)")

    lines.append("")
    lines.append("Peri-event glucose change (event minus same-patient random control):")
    for ev, label in [
        ("bolus_volume_delivered", "Bolus insulin"),
        ("carb_input", "Carbohydrate intake"),
        ("steps", "High-activity bin (>=100 steps)"),
    ]:
        sub = peri_event_df[peri_event_df["event_type"] == ev].sort_values("horizon_minutes")
        if not len(sub):
            continue
        lines.append(f"- {label} (n_events={int(sub['n_events'].max())}):")
        for _, r in sub.iterrows():
            lines.append(
                f"    +{int(r['horizon_minutes']):3d} min  event Δ {r['mean_delta_event']:+6.2f}  "
                f"control Δ {r['mean_delta_control']:+6.2f}  "
                f"diff {r['mean_delta_diff']:+6.2f} mg/dL"
            )

    lines.append("")
    lines.append("Glucose by day of week (Mon..Sun):")
    for _, r in dow_df.iterrows():
        lines.append(
            f"  {r['dayofweek_label']}: mean {r['glucose_mean']:.1f}  TIR {r['tir_pct']:.1f}%  "
            f"hypo {r['hypo_pct']:.2f}%  hyper {r['hyper_pct']:.2f}%"
        )

    if peri_subtype_df is not None and len(peri_subtype_df):
        lines.append("")
        lines.append("Peri-event by SUBTYPE (event - same-patient control):")
        subtype_labels = {
            "bolus_meal": "Bolus + carb within +/- 15min (meal bolus)",
            "bolus_correction": "Bolus with no carb in window (correction bolus)",
            "carb_meal": "Carb + bolus within +/- 15min (meal carb)",
            "carb_solo": "Carb with no bolus in window (solo carb)",
        }
        for sname, label in subtype_labels.items():
            sub = peri_subtype_df[peri_subtype_df["event_subtype"] == sname].sort_values("horizon_minutes")
            if not len(sub):
                continue
            n_ev = int(sub["n_events"].max())
            lines.append(f"- {label} (n_events={n_ev}):")
            for _, r in sub.iterrows():
                lines.append(
                    f"    +{int(r['horizon_minutes']):3d} min  event Δ {r['mean_delta_event']:+6.2f}  "
                    f"diff {r['mean_delta_diff']:+6.2f} mg/dL"
                )

    if peri_per_patient_df is not None and len(peri_per_patient_df):
        lines.append("")
        lines.append("Peri-event aggregated PER PATIENT first (mean +/- SD across patients with events):")
        for ev, label in [
            ("bolus_volume_delivered", "Bolus"),
            ("carb_input", "Carb"),
            ("steps", "Steps>=100"),
        ]:
            sub = peri_per_patient_df[peri_per_patient_df["event_type"] == ev].sort_values("horizon_minutes")
            if not len(sub):
                continue
            n_pat = int(sub["n_patients"].iloc[0])
            lines.append(f"- {label} (n_patients={n_pat}):")
            for _, r in sub.iterrows():
                lines.append(
                    f"    +{int(r['horizon_minutes']):3d} min  patient-mean diff {r['patient_mean_diff']:+6.2f}  "
                    f"SD {r['patient_std_diff']:.2f}  "
                    f"[min {r['patient_min_diff']:+.1f}, max {r['patient_max_diff']:+.1f}]"
                )

    lines.append("")
    lines.append("Absolute 5-min velocity quantiles by glycaemic zone (UNFILTERED, includes sensor caps):")
    for _, r in velocity_zone_df.iterrows():
        lines.append(
            f"  {r['zone']:<9}  P50 {r['p50_abs_velocity']:.2f}  "
            f"P90 {r['p90_abs_velocity']:.2f}  P95 {r['p95_abs_velocity']:.2f}  "
            f"P99 {r['p99_abs_velocity']:.2f}  (n={int(r['n'])})"
        )

    if velocity_zone_filtered_df is not None and len(velocity_zone_filtered_df):
        lines.append("")
        lines.append("Same velocity quantiles, FILTERED to exclude rows at sensor caps (==40 or >400):")
        for _, r in velocity_zone_filtered_df.iterrows():
            lines.append(
                f"  {r['zone']:<9}  P50 {r['p50_abs_velocity']:.2f}  "
                f"P90 {r['p90_abs_velocity']:.2f}  P95 {r['p95_abs_velocity']:.2f}  "
                f"P99 {r['p99_abs_velocity']:.2f}  (n={int(r['n'])})"
            )

    lines.append("")
    lines.extend(
        [
            "Downstream implications:",
            "- Use chronological and patient-aware evaluation; long participants dominate pooled rows.",
            "- Keep 120m lookback as the first setting and evaluate 180m as an ablation.",
            "- Add censoring flags for glucose floor/ceiling values.",
            "- Add missing-modality indicators before multimodal modelling.",
            "- Peri-event evidence is the meaningful test for modality usefulness, not Pearson r.",
            "- Subtype analysis: correction-bolus rows isolate pure-insulin effect; solo-carb rows isolate pure-meal effect.",
            "- Per-patient peri-event SD across patients quantifies the response heterogeneity the model has to absorb.",
            "- Compare filtered vs unfiltered velocity by zone to confirm hypo dynamics are not deflated by sensor caps.",
            "- Per-patient ACF half-life heterogeneity supports a patient embedding / static branch.",
            "- Report per-horizon, per-patient, and zone-specific metrics.",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def run_eda(base_path: Path) -> dict[str, Path]:
    paths = ensure_dirs(base_path)
    df = add_time_features(load_hupa_all(base_path))
    static = load_patient_characteristics(base_path)
    df = df.merge(static[["participant_id", "gender", "treatment", "hba1c_pct", "age_years", "bmi"]], on="participant_id", how="left")

    patient_summary = structural_quality_summary(df)
    velocity = glucose_velocity_summary(df)
    acf_df, pacf_df = acf_pacf_tables(df)
    circadian = circadian_summary(df)
    seq = sequence_feasibility(df)
    glucose_dist = glucose_distribution_table(df)
    velocity_zone = velocity_by_zone_table(df)
    velocity_zone_filtered = velocity_by_zone_filtered(df)
    dow = dayofweek_summary(df)
    halflife = per_patient_acf_halflife(acf_df, threshold=0.5)
    peri_event = all_peri_event_summaries(df)
    peri_subtype = peri_event_summary_by_subtype(df)
    peri_per_patient = peri_event_per_patient_stats(df)
    overall = build_overall_summary(df, patient_summary, acf_df, pacf_df, seq, halflife, peri_event)

    outputs = {
        "patient_summary": paths["interim"] / "hupa_eda_patient_summary.csv",
        "overall": paths["tables"] / "hupa_eda_overall_summary.csv",
        "velocity": paths["tables"] / "hupa_eda_velocity_summary.csv",
        "acf": paths["tables"] / "hupa_eda_acf_by_lag.csv",
        "pacf": paths["tables"] / "hupa_eda_pacf_by_lag.csv",
        "circadian": paths["tables"] / "hupa_eda_circadian_by_hour.csv",
        "sequence": paths["tables"] / "hupa_eda_sequence_feasibility.csv",
        "glucose_distribution": paths["tables"] / "hupa_eda_glucose_distribution.csv",
        "velocity_by_zone": paths["tables"] / "hupa_eda_velocity_by_zone.csv",
        "velocity_by_zone_filtered": paths["tables"] / "hupa_eda_velocity_by_zone_filtered.csv",
        "dayofweek": paths["tables"] / "hupa_eda_dayofweek.csv",
        "acf_halflife": paths["tables"] / "hupa_eda_acf_halflife.csv",
        "peri_event": paths["tables"] / "hupa_eda_peri_event.csv",
        "peri_event_subtype": paths["tables"] / "hupa_eda_peri_event_subtype.csv",
        "peri_event_per_patient": paths["tables"] / "hupa_eda_peri_event_per_patient.csv",
        "figure_overview": paths["figures"] / "02_eda_overview.png",
        "text": paths["outputs"] / "hupa_eda_summary.txt",
    }
    patient_summary.to_csv(outputs["patient_summary"], index=False)
    overall.to_csv(outputs["overall"], index=False)
    velocity.to_csv(outputs["velocity"], index=False)
    acf_df.to_csv(outputs["acf"], index=False)
    pacf_df.to_csv(outputs["pacf"], index=False)
    circadian.to_csv(outputs["circadian"], index=False)
    seq.to_csv(outputs["sequence"], index=False)
    glucose_dist.to_csv(outputs["glucose_distribution"], index=False)
    velocity_zone.to_csv(outputs["velocity_by_zone"], index=False)
    velocity_zone_filtered.to_csv(outputs["velocity_by_zone_filtered"], index=False)
    dow.to_csv(outputs["dayofweek"], index=False)
    halflife.to_csv(outputs["acf_halflife"], index=False)
    peri_event.to_csv(outputs["peri_event"], index=False)
    peri_subtype.to_csv(outputs["peri_event_subtype"], index=False)
    peri_per_patient.to_csv(outputs["peri_event_per_patient"], index=False)
    make_overview_figure(patient_summary, acf_df, circadian, peri_event, outputs["figure_overview"])
    outputs.update(
        make_individual_figures(
            df,
            patient_summary,
            acf_df,
            circadian,
            halflife,
            dow,
            velocity_zone,
            peri_event,
            peri_subtype,
            paths["figures"],
        )
    )
    write_text_summary(
        outputs["text"], overall, patient_summary, peri_event, dow,
        velocity_zone, halflife, peri_subtype, peri_per_patient, velocity_zone_filtered,
    )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-path", default=".", help="Project root / Colab BASE_PATH")
    args = parser.parse_args()
    outputs = run_eda(Path(args.base_path).resolve())
    print("Generated EDA artefacts:")
    for name, path in outputs.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
