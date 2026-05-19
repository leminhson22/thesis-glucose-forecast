"""HUPA-UCM preprocessing and feature engineering for thesis modelling.

This module builds the thesis-ready training table on top of the glUCModel-
preprocessed Excel files (Hidalgo et al., 2024 §4.2). The author preprocessing
already produced a strict 5-min grid; this module focuses on the parts that are
the thesis's own methodological contribution:

* Sensor-cap (LO / HI) flagging.
* Modality-availability flags + 24-hour basal coverage rolling fraction.
* Time, glucose-derived, and rolling event features.
* Multi-horizon target construction (30 / 60 / 90 min).
* Chronological per-patient 70 / 15 / 15 split with a boundary buffer.
* Per-subject Z-score for continuous signals fit on the train portion only.
* log1p + global Z-score for sparse event-rate features.
* Sliding-window sequence construction with adaptive-stride sub-sampling on
  TRAIN only (see memory: long-patient-strategy). Val and test stay stride=1.

All functions are pure (no global state, no file I/O except where named) so the
pipeline can be unit-tested and re-run deterministically.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

try:
    from . import config as C
    from .data_loading import (
        attach_static_metadata,
        list_hupa_participants,
        load_hupa_all,
        load_patient_characteristics,
    )
except ImportError:  # flat import path (notebook adds src/ to sys.path)
    import config as C  # type: ignore[no-redef]
    from data_loading import (  # type: ignore[no-redef]
        attach_static_metadata,
        list_hupa_participants,
        load_hupa_all,
        load_patient_characteristics,
    )


# ---------------------------------------------------------------------------
# Static-metadata fixes
# ---------------------------------------------------------------------------


def apply_treatment_override(static_df: pd.DataFrame) -> pd.DataFrame:
    """Reclassify HUPA0011P from CSII to MDI (Pitfall #8).

    The original characteristic file lists HUPA0011P as CSII, but the patient
    has zero basal records. Leaving the label as CSII would teach the model
    that "CSII therapy implies no basal signal", which is the opposite of the
    intended physiological prior.
    """
    out = static_df.copy()
    pid, new_label = C.P11_TREATMENT_OVERRIDE
    mask = out["participant_id"] == pid
    out.loc[mask, "treatment"] = new_label
    return out


# ---------------------------------------------------------------------------
# Censoring flags
# ---------------------------------------------------------------------------


def add_censoring_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Flag censored glucose values without modifying the readings themselves.

    FreeStyle Libre 2 reports `LO` for any reading <= 40 mg/dL and `HI` for
    any reading > 400 mg/dL. After glUCModel preprocessing these become the
    literal numbers 40 and ~440. Retain them but expose two binary indicators
    so the model can either down-weight them in the loss or be evaluated
    separately on censored vs. uncensored windows.
    """
    out = df.copy()
    out["glucose_low_cap"] = (out["glucose"] <= C.GLUCOSE_LOW_CAP).astype(np.int8)
    out["glucose_high_extreme"] = (
        out["glucose"] > C.GLUCOSE_HIGH_EXTREME_THRESHOLD
    ).astype(np.int8)
    return out


# ---------------------------------------------------------------------------
# Time features
# ---------------------------------------------------------------------------


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical encodings for hour-of-day and day-of-week.

    Justified by EDA §4.5 circadian profile showing a glucose peak in the
    7–10 a.m. dawn window and a smaller post-dinner peak.
    """
    out = df.copy()
    ts = pd.to_datetime(out["time"])
    hour = ts.dt.hour + ts.dt.minute / 60.0
    dow = ts.dt.dayofweek
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0).astype(np.float32)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0).astype(np.float32)
    out["dayofweek_sin"] = np.sin(2 * np.pi * dow / 7.0).astype(np.float32)
    out["dayofweek_cos"] = np.cos(2 * np.pi * dow / 7.0).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Glucose-derived features
# ---------------------------------------------------------------------------


def _group_rolling(
    df: pd.DataFrame,
    column: str,
    window_steps: int,
    agg: str,
    new_col: str,
    min_periods: int | None = None,
) -> pd.Series:
    """Per-patient rolling aggregate aligned to the end of the window.

    Uses ``min_periods=1`` by default so the first few timesteps of each
    patient still produce a valid (partial) value instead of NaN. This keeps
    sequence-build windows valid; the small noisiness at the very start of a
    patient's timeline is then absorbed by per-subject Z-scoring.
    """
    if min_periods is None:
        min_periods = 1
    series = df.groupby("participant_id", group_keys=False)[column].rolling(
        window=window_steps, min_periods=min_periods
    )
    if agg == "mean":
        result = series.mean()
    elif agg == "sum":
        result = series.sum()
    else:
        raise ValueError(f"Unsupported aggregation: {agg}")
    result = result.reset_index(level=0, drop=True).astype(np.float32)
    result.name = new_col
    return result


def add_glucose_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Velocity, acceleration, rolling means, and rolling std of glucose."""
    out = df.sort_values(["participant_id", "time"]).reset_index(drop=True)
    g = out.groupby("participant_id")["glucose"]
    # Velocity is mg/dL per minute. Δt is 5 min on the HUPA grid.
    # The first row of each patient has no predecessor -> set the derivative
    # to 0 so sequence-validity checks downstream do not reject windows that
    # otherwise contain fully observed glucose.
    out["glucose_velocity"] = (g.diff() / C.SAMPLING_STEP_MIN).astype(np.float32)
    out["glucose_velocity"] = out["glucose_velocity"].fillna(0).astype(np.float32)
    # Clinical-lens revisit: glucose_acceleration removed. No CGM device
    # displays a 2nd-derivative trend arrow and no clinical guideline uses
    # it; velocity captures the actionable trend signal.

    for steps in C.ROLL_GLUCOSE_MEAN_STEPS:
        col = f"glucose_{steps * C.SAMPLING_STEP_MIN}m_mean"
        out[col] = _group_rolling(out, "glucose", steps, "mean", col)

    # Rolling standard deviation — short-horizon glucose variability proxy.
    # min_periods=2 because std of one observation is undefined; the resulting
    # first-row NaN is then filled with 0 because a window of one constant
    # value has zero variation by definition.
    for steps in C.ROLL_GLUCOSE_STD_STEPS:
        col = f"glucose_{steps * C.SAMPLING_STEP_MIN}m_std"
        series = (
            out.groupby("participant_id", group_keys=False)["glucose"]
            .rolling(window=steps, min_periods=2)
            .std()
            .reset_index(level=0, drop=True)
        )
        out[col] = series.fillna(0).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Pharmacokinetic-decay aggregations: Insulin On Board, Carbs On Board
# ---------------------------------------------------------------------------


def _exponential_iir(values: np.ndarray, tau_min: float, dt_min: float = C.SAMPLING_STEP_MIN) -> np.ndarray:
    """Single-pole IIR exponential rolling sum: out[t] = alpha*out[t-1] + x[t].

    Equivalent to convolving ``values`` with the infinite kernel
    ``exp(-k * dt / tau)`` for ``k = 0, 1, 2, ...``. Each new event x[t]
    contributes its full magnitude immediately and decays exponentially in
    subsequent bins. The starting condition is out[0] = values[0] because we
    have no information about pre-history.
    """
    alpha = np.float32(np.exp(-dt_min / tau_min))
    out = np.empty_like(values, dtype=np.float32)
    if values.size == 0:
        return out
    out[0] = values[0]
    for i in range(1, values.size):
        out[i] = alpha * out[i - 1] + values[i]
    return out


def add_pharmacokinetic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add IOB and COB columns based on exponential decay of bolus/carb events.

    IOB (Insulin On Board) approximates the active insulin still acting on the
    patient at time t. tau=75 min reflects rapid-acting insulin analogs.

    COB (Carbs On Board) approximates the still-absorbing carbohydrate at
    time t. tau=60 min reflects a first-order Hovorka-style absorption rate.

    Both are computed per-participant via the recurrence in
    :func:`_exponential_iir`. The recurrence is reset at the start of each
    patient because we cannot infer pre-timeline events.
    """
    out = df.sort_values(["participant_id", "time"]).reset_index(drop=True)
    iob_vals = np.zeros(len(out), dtype=np.float32)
    cob_vals = np.zeros(len(out), dtype=np.float32)
    for pid, sub in out.groupby("participant_id", sort=False):
        idx = sub.index.to_numpy()
        iob_vals[idx] = _exponential_iir(
            sub["bolus_volume_delivered"].to_numpy(dtype=np.float32),
            tau_min=C.IOB_TAU_MIN,
        )
        cob_vals[idx] = _exponential_iir(
            sub["carb_input"].to_numpy(dtype=np.float32),
            tau_min=C.COB_TAU_MIN,
        )
    out["insulin_on_board"] = iob_vals
    out["carbs_on_board"] = cob_vals
    return out


# ---------------------------------------------------------------------------
# Rolling features on event streams
# ---------------------------------------------------------------------------


def add_event_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["participant_id", "time"]).reset_index(drop=True)
    plans = [
        ("bolus_volume_delivered", C.ROLL_BOLUS_SUM_STEPS, "sum", "bolus_{m}m_sum"),
        ("carb_input", C.ROLL_CARB_SUM_STEPS, "sum", "carb_{m}m_sum"),
        ("steps", C.ROLL_STEPS_SUM_STEPS, "sum", "steps_{m}m_sum"),
        ("calories", C.ROLL_CALORIES_SUM_STEPS, "sum", "calories_{m}m_sum"),
        ("heart_rate", C.ROLL_HR_MEAN_STEPS, "mean", "heart_rate_{m}m_mean"),
    ]
    for col, spans, agg, template in plans:
        for steps in spans:
            new_col = template.format(m=steps * C.SAMPLING_STEP_MIN)
            out[new_col] = _group_rolling(out, col, steps, agg, new_col)
    return out


# ---------------------------------------------------------------------------
# Modality availability
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModalityAvailability:
    """Per-patient summary of which event streams were recorded at all."""

    basal_available: Mapping[str, int]
    bolus_available: Mapping[str, int]
    carb_available: Mapping[str, int]

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for pid in self.basal_available:
            rows.append(
                {
                    "participant_id": pid,
                    "basal_available": self.basal_available[pid],
                    "bolus_available": self.bolus_available[pid],
                    "carb_available": self.carb_available[pid],
                }
            )
        return pd.DataFrame(rows)


def compute_modality_availability(df: pd.DataFrame) -> ModalityAvailability:
    """Compute three binary per-patient flags from the raw event streams."""
    basal = {}
    bolus = {}
    carb = {}
    for pid, sub in df.groupby("participant_id"):
        basal[pid] = int((sub["basal_rate"] > 0).any())
        bolus[pid] = int((sub["bolus_volume_delivered"] > 0).any())
        carb[pid] = int((sub["carb_input"] > 0).any())
    return ModalityAvailability(basal, bolus, carb)


def add_modality_availability(
    df: pd.DataFrame, availability: ModalityAvailability
) -> pd.DataFrame:
    """Broadcast per-patient flags to every row + rolling basal-coverage."""
    out = df.sort_values(["participant_id", "time"]).reset_index(drop=True)
    avail_frame = availability.to_frame()
    out = out.merge(avail_frame, on="participant_id", how="left")
    out[["basal_available", "bolus_available", "carb_available"]] = (
        out[["basal_available", "bolus_available", "carb_available"]]
        .fillna(0)
        .astype(np.int8)
    )
    # Rolling fraction of bins in the last 24 h where basal > 0.
    basal_active = (out["basal_rate"] > 0).astype(np.float32)
    out["basal_coverage_24h"] = (
        basal_active.groupby(out["participant_id"])
        .rolling(window=C.BASAL_COVERAGE_WINDOW_STEPS, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        .astype(np.float32)
    )
    return out


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add target_30m / 60m / 90m by per-patient negative shift of glucose.

    The shift is patient-grouped so that horizons never read a value from a
    different participant if the long table is concatenated.
    """
    out = df.sort_values(["participant_id", "time"]).reset_index(drop=True)
    for steps, minutes in zip(C.HORIZON_STEPS, C.HORIZON_MINUTES):
        col = f"target_{minutes}m"
        out[col] = out.groupby("participant_id")["glucose"].shift(-steps).astype(
            np.float32
        )
    return out


# ---------------------------------------------------------------------------
# Chronological split
# ---------------------------------------------------------------------------


def assign_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign each row of each patient to 'train', 'val', or 'test'.

    The split is strictly chronological inside each patient. A buffer of
    ``MAX_HORIZON_STEPS`` rows is removed at each boundary so the target of a
    train row cannot fall into the val partition (and similarly val→test).
    Boundary rows are marked 'buffer' and discarded during sequence build.

    Returns
    -------
    (df_with_split, boundaries) where ``boundaries`` is a one-row-per-patient
    summary that gets persisted to outputs/tables for the report.
    """
    out = df.sort_values(["participant_id", "time"]).reset_index(drop=True)
    out["split"] = "train"

    rows = []
    for pid, sub in out.groupby("participant_id"):
        n = len(sub)
        train_end = int(n * C.SPLIT_TRAIN_FRAC)
        val_end = int(n * (C.SPLIT_TRAIN_FRAC + C.SPLIT_VAL_FRAC))
        idx = sub.index
        buf = C.SPLIT_BOUNDARY_BUFFER_STEPS

        # train: [0, train_end - buf)
        # buffer: [train_end - buf, train_end)
        # val:   [train_end, val_end - buf)
        # buffer: [val_end - buf, val_end)
        # test:  [val_end, n)
        out.loc[idx[: max(0, train_end - buf)], "split"] = "train"
        out.loc[idx[max(0, train_end - buf) : train_end], "split"] = "buffer"
        out.loc[idx[train_end : max(train_end, val_end - buf)], "split"] = "val"
        out.loc[idx[max(train_end, val_end - buf) : val_end], "split"] = "buffer"
        out.loc[idx[val_end:], "split"] = "test"

        rows.append(
            {
                "participant_id": pid,
                "n_rows": n,
                "train_end_row": train_end,
                "val_end_row": val_end,
                "buffer_steps": buf,
                "train_start_time": sub["time"].iloc[0],
                "val_start_time": sub["time"].iloc[train_end],
                "test_start_time": sub["time"].iloc[val_end] if val_end < n else None,
            }
        )

    boundaries = pd.DataFrame(rows)
    return out, boundaries


# ---------------------------------------------------------------------------
# Derived static features (TRAIN-ONLY computation, then broadcast to all rows)
# ---------------------------------------------------------------------------


def compute_derived_static_features(df_train: pd.DataFrame) -> pd.DataFrame:
    """Compute per-patient statistics from train-portion rows ONLY.

    No leakage: this function must be called on rows where ``split == 'train'``.

    Step 4-revisit dropped ``subject_tir_pct`` and ``mean_daily_steps``.
    Clinical-lens revisit additionally drops:
      * ``carb_events_per_day`` — measures meal-logging frequency, not
        eating frequency; misleading for patients who do not log meals.
      * ``data_duration_days`` — pure data artefact, no clinical meaning,
        misleading at deployment when a new patient has tiny duration.
      * ``basal_recording_pct`` — correlates ~0.9 with treatment_CSII and is
        already captured by the binary basal_available flag.
    """
    rows = []
    for pid, sub in df_train.groupby("participant_id"):
        n = len(sub)
        g = sub["glucose"]
        active = (sub["steps"] > C.STEPS_ACTIVE_THRESHOLD).mean()
        days = n / C.STEPS_PER_DAY
        rows.append(
            {
                "participant_id": pid,
                "subject_mean_glucose": float(g.mean()),
                "subject_std_glucose": float(g.std()),
                "subject_hypo_pct": float((g < C.GLUCOSE_HYPO_THRESHOLD).mean() * 100),
                "subject_hyper_pct": float(
                    (g > C.GLUCOSE_HYPER_THRESHOLD).mean() * 100
                ),
                "bolus_events_per_day": float(
                    (sub["bolus_volume_delivered"] > 0).sum() / max(days, 1e-6)
                ),
                "steps_active_pct": float(active * 100),
                "mean_heart_rate": float(sub["heart_rate"].mean()),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-subject Z-score scaler (fit on train portion only)
# ---------------------------------------------------------------------------


def fit_scalers(df_train: pd.DataFrame) -> dict:
    """Compute scaler parameters from train rows.

    Returns a JSON-serialisable dict with two top-level keys:
        "per_subject": {feature: {pid: {mean, std}}}
        "global_log1p": {feature: {mean, std}}
    """
    scalers: dict = {"per_subject": {}, "global_log1p": {}}

    for feat in C.PER_SUBJECT_ZSCORE_FEATURES:
        if feat not in df_train.columns:
            continue
        per_pid = {}
        for pid, sub in df_train.groupby("participant_id"):
            s = sub[feat].dropna()
            if len(s) < 2:
                per_pid[pid] = {"mean": float(s.mean() if len(s) else 0.0), "std": 1.0}
            else:
                std = float(s.std())
                per_pid[pid] = {
                    "mean": float(s.mean()),
                    "std": std if std > 1e-9 else 1.0,
                }
        scalers["per_subject"][feat] = per_pid

    for feat in C.LOG1P_THEN_GLOBAL_ZSCORE_FEATURES:
        if feat not in df_train.columns:
            continue
        s = np.log1p(df_train[feat].clip(lower=0).to_numpy(dtype=np.float32))
        std = float(s.std())
        scalers["global_log1p"][feat] = {
            "mean": float(s.mean()),
            "std": std if std > 1e-9 else 1.0,
        }
    return scalers


def apply_scalers(df: pd.DataFrame, scalers: dict) -> pd.DataFrame:
    """Apply scalers fit on train to any df (train / val / test)."""
    out = df.copy()
    for feat, per_pid in scalers["per_subject"].items():
        if feat not in out.columns:
            continue
        # Map mean/std by participant_id.
        mean_map = {pid: v["mean"] for pid, v in per_pid.items()}
        std_map = {pid: v["std"] for pid, v in per_pid.items()}
        means = out["participant_id"].map(mean_map).astype(np.float32)
        stds = out["participant_id"].map(std_map).astype(np.float32)
        out[feat] = ((out[feat].astype(np.float32) - means) / stds).astype(np.float32)

    for feat, params in scalers["global_log1p"].items():
        if feat not in out.columns:
            continue
        x = np.log1p(out[feat].clip(lower=0).to_numpy(dtype=np.float32))
        out[feat] = ((x - params["mean"]) / params["std"]).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Sequence construction
# ---------------------------------------------------------------------------


# Order of dynamic features in the (T, F) tensor. Constructed from config so
# adding a feature stays a one-line change.
def dynamic_feature_order() -> list[str]:
    """Final dynamic feature order after Step 4-revisit (19 columns).

    Raw `calories`, `steps`, `bolus_volume_delivered`, `carb_input` are no
    longer exposed: calories/steps are duplicated by their rolling
    aggregates; bolus/carb are recoverable from IOB / COB via the inverse
    recurrence. The bolus-sum span is reduced from three to one (60 min only)
    because IOB already encodes the full pharmacokinetic decay across spans.
    """
    base = [
        "glucose",
        "heart_rate",
        "basal_rate",
    ]
    rolling = [
        f"glucose_{s * C.SAMPLING_STEP_MIN}m_mean" for s in C.ROLL_GLUCOSE_MEAN_STEPS
    ] + [
        f"glucose_{s * C.SAMPLING_STEP_MIN}m_std" for s in C.ROLL_GLUCOSE_STD_STEPS
    ] + ["glucose_velocity"] + [
        f"bolus_{s * C.SAMPLING_STEP_MIN}m_sum" for s in C.ROLL_BOLUS_SUM_STEPS
    ] + [
        f"carb_{s * C.SAMPLING_STEP_MIN}m_sum" for s in C.ROLL_CARB_SUM_STEPS
    ] + [
        f"steps_{s * C.SAMPLING_STEP_MIN}m_sum" for s in C.ROLL_STEPS_SUM_STEPS
    ] + [
        f"calories_{s * C.SAMPLING_STEP_MIN}m_sum"
        for s in C.ROLL_CALORIES_SUM_STEPS
    ] + [
        f"heart_rate_{s * C.SAMPLING_STEP_MIN}m_mean"
        for s in C.ROLL_HR_MEAN_STEPS
    ] + ["insulin_on_board", "carbs_on_board"]
    flags = list(C.PASSTHROUGH_FEATURES)
    return base + rolling + flags


def _adaptive_stride_indices(n_anchors: int, cap: int) -> np.ndarray:
    """Return ``min(n_anchors, cap)`` indices uniformly spread on [0, n_anchors)."""
    if n_anchors <= cap:
        return np.arange(n_anchors, dtype=np.int64)
    stride = n_anchors / cap
    return np.floor(np.arange(cap, dtype=np.float64) * stride).astype(np.int64)


@dataclass
class SequenceBundle:
    X_dynamic: np.ndarray            # (N, lookback, F)
    X_static: np.ndarray             # (N, S)
    y: np.ndarray                    # (N, len(HORIZON_STEPS))
    participant_ids: np.ndarray      # (N,) string
    split: np.ndarray                # (N,) 'train'|'val'|'test'
    anchor_time: np.ndarray          # (N,) datetime64[ns] — time at end of lookback
    feature_names_dynamic: list[str]
    feature_names_static: list[str]


def build_sequences(
    df: pd.DataFrame,
    static_table: pd.DataFrame,
    lookback: int = C.LOOKBACK_STEPS,
) -> SequenceBundle:
    """Build sliding-window sequences and apply the long-patient train cap.

    A window is valid when:
      * lookback rows are all on the strict 5-min grid for the same patient,
      * none of those rows fall in a 'buffer' partition,
      * all three target horizons are observed (not NaN),
      * the anchor row's split label is one of train/val/test.

    Train windows are sub-sampled per-patient with an adaptive deterministic
    stride to enforce ``N_TRAIN_CAP``. Val and test windows are kept at
    stride=1.

    Static features are joined by participant_id and broadcast.
    """
    feat_names = dynamic_feature_order()
    static_feat_names = [c for c in static_table.columns if c != "participant_id"]

    df = df.sort_values(["participant_id", "time"]).reset_index(drop=True)
    target_cols = [f"target_{m}m" for m in C.HORIZON_MINUTES]

    X_dyn_list = []
    X_stat_list = []
    y_list = []
    pid_list = []
    split_list = []
    anchor_time_list = []

    static_lookup = static_table.set_index("participant_id")[static_feat_names]

    for pid, sub in df.groupby("participant_id", sort=True):
        # Anchor index t spans [lookback-1, n - max_horizon - 1].
        n = len(sub)
        if n < lookback + C.MAX_HORIZON_STEPS:
            continue
        sub = sub.reset_index(drop=True)
        feat_block = sub[feat_names].to_numpy(dtype=np.float32, copy=False)
        target_block = sub[target_cols].to_numpy(dtype=np.float32, copy=False)
        split_arr = sub["split"].to_numpy()
        time_arr = pd.to_datetime(sub["time"]).to_numpy()

        anchor_idx = np.arange(lookback - 1, n - C.MAX_HORIZON_STEPS, dtype=np.int64)

        # Filter anchors: no buffer/NaN in window, all horizons observed.
        keep = np.ones_like(anchor_idx, dtype=bool)
        for k, t in enumerate(anchor_idx):
            window_split = split_arr[t - lookback + 1 : t + 1]
            if (window_split == "buffer").any():
                keep[k] = False
                continue
            if split_arr[t] == "buffer":
                keep[k] = False
                continue
            if np.isnan(target_block[t]).any():
                keep[k] = False
                continue
            if np.isnan(feat_block[t - lookback + 1 : t + 1]).any():
                keep[k] = False
        anchor_idx = anchor_idx[keep]
        if anchor_idx.size == 0:
            continue

        anchor_splits = split_arr[anchor_idx]
        train_mask = anchor_splits == "train"
        train_anchors = anchor_idx[train_mask]
        if train_anchors.size > 0:
            picked = _adaptive_stride_indices(train_anchors.size, C.N_TRAIN_CAP)
            train_anchors = train_anchors[picked]

        val_anchors = anchor_idx[anchor_splits == "val"]
        test_anchors = anchor_idx[anchor_splits == "test"]

        for anchors, split_label in (
            (train_anchors, "train"),
            (val_anchors, "val"),
            (test_anchors, "test"),
        ):
            if anchors.size == 0:
                continue
            wins = np.stack(
                [feat_block[t - lookback + 1 : t + 1] for t in anchors],
                axis=0,
            )
            X_dyn_list.append(wins)
            y_list.append(target_block[anchors])
            pid_list.append(np.array([pid] * anchors.size, dtype=object))
            split_list.append(
                np.array([split_label] * anchors.size, dtype=object)
            )
            anchor_time_list.append(time_arr[anchors])
            # Static vector broadcast for every anchor.
            if pid in static_lookup.index:
                stat_row = static_lookup.loc[pid].to_numpy(dtype=np.float32)
            else:
                stat_row = np.zeros(len(static_feat_names), dtype=np.float32)
            X_stat_list.append(
                np.broadcast_to(stat_row, (anchors.size, stat_row.size)).copy()
            )

    X_dynamic = np.concatenate(X_dyn_list, axis=0)
    X_static = np.concatenate(X_stat_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    participant_ids = np.concatenate(pid_list, axis=0)
    split = np.concatenate(split_list, axis=0)
    anchor_time = np.concatenate(anchor_time_list, axis=0)
    return SequenceBundle(
        X_dynamic=X_dynamic,
        X_static=X_static,
        y=y,
        participant_ids=participant_ids,
        split=split,
        anchor_time=anchor_time,
        feature_names_dynamic=feat_names,
        feature_names_static=static_feat_names,
    )


# ---------------------------------------------------------------------------
# Static-feature table (clinical + derived) preparation
# ---------------------------------------------------------------------------


def build_static_feature_table(
    static_clinical: pd.DataFrame,
    derived: pd.DataFrame,
    availability: ModalityAvailability,
) -> pd.DataFrame:
    """One row per patient with clinical + derived + availability features.

    Categorical columns ``gender`` and ``treatment`` are one-hot encoded; the
    redundant half of each one-hot pair (gender_Male, treatment_MDI) is then
    dropped because the complementary value is recoverable as 1 - x.

    Step 4-revisit also drops weight_kg and height_cm here because BMI fully
    summarises their clinical signal and ranks higher in selection.
    """
    base = static_clinical.copy()
    base = base.merge(derived, on="participant_id", how="left")
    base = base.merge(availability.to_frame(), on="participant_id", how="left")

    # Drop weight + height after BMI has been computed upstream.
    base = base.drop(columns=["weight_kg", "height_cm"], errors="ignore")

    base = pd.get_dummies(
        base,
        columns=["gender", "treatment"],
        prefix=["gender", "treatment"],
        dtype=np.float32,
    )
    # Drop the redundant one-hot half. ``errors="ignore"`` keeps the function
    # safe if the cohort happens to contain only one category for a column.
    base = base.drop(columns=list(C.ONE_HOT_DROP_AFTER_ENCODE), errors="ignore")
    return base


def normalise_static_numeric(static: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Z-score the numeric clinical + derived columns across patients.

    Fit and apply on the full 25-patient table because static covariates are
    fixed and not subject to chronological train/val/test leakage — every
    patient has exactly one row.
    """
    out = static.copy()
    params: dict = {}
    for col in list(C.CLINICAL_STATIC_NUMERIC) + list(C.DERIVED_STATIC_NUMERIC):
        if col not in out.columns:
            continue
        x = out[col].astype(np.float32)
        mean = float(x.mean())
        std = float(x.std()) or 1.0
        out[col] = ((x - mean) / std).astype(np.float32)
        params[col] = {"mean": mean, "std": std}
    return out, params


# ---------------------------------------------------------------------------
# Top-level pipeline orchestrator
# ---------------------------------------------------------------------------


def run_preprocessing(
    base_path: str | Path | None = None,
    participants: Iterable[str] | None = None,
    save: bool = True,
) -> dict:
    """End-to-end pipeline returning artefacts in a dict.

    Parameters
    ----------
    base_path
        Project root (``BASE_PATH`` from the Colab cell).
    participants
        Optional subset of participant ids for debug mode. If ``None`` use all.
    save
        If True, persist parquet / npz / csv / json artefacts to disk.
    """
    base = Path(base_path).resolve() if base_path else Path(__file__).resolve().parent.parent

    # ---------------- A. Load raw author-preprocessed data ----------------
    if participants is None:
        df = load_hupa_all(base_path=base)
    else:
        try:
            from .data_loading import load_hupa_patient
        except ImportError:
            from data_loading import load_hupa_patient  # type: ignore[no-redef]

        df = pd.concat(
            [load_hupa_patient(pid, base_path=base) for pid in participants],
            axis=0,
            ignore_index=True,
        )
    df["time"] = pd.to_datetime(df["time"])

    # ---------------- B. Feature engineering ------------------------------
    availability = compute_modality_availability(df)
    df = add_censoring_flags(df)
    df = add_time_features(df)
    df = add_glucose_derived_features(df)
    df = add_event_rolling_features(df)
    df = add_pharmacokinetic_features(df)
    df = add_modality_availability(df, availability)
    df = add_targets(df)

    # ---------------- C. Chronological split ------------------------------
    df, boundaries = assign_split(df)

    # ---------------- D. Static-features from train only ------------------
    train_df = df[df["split"] == "train"]
    derived = compute_derived_static_features(train_df)

    static_clinical = apply_treatment_override(load_patient_characteristics(base))
    # Keep only the patients actually loaded; this is a no-op in production
    # (all 25 patients) but prevents NaN availability flags in debug mode.
    present_pids = sorted(df["participant_id"].unique())
    static_clinical = static_clinical[
        static_clinical["participant_id"].isin(present_pids)
    ].reset_index(drop=True)
    static_table = build_static_feature_table(
        static_clinical=static_clinical,
        derived=derived,
        availability=availability,
    )
    static_table, static_norm_params = normalise_static_numeric(static_table)

    # ---------------- E. Scalers on dynamic features ----------------------
    scalers = fit_scalers(train_df)
    df_scaled = apply_scalers(df, scalers)

    # ---------------- F. Sequence build -----------------------------------
    bundle = build_sequences(df_scaled, static_table, lookback=C.LOOKBACK_STEPS)

    # ---------------- G. Persist artefacts --------------------------------
    if save:
        (base / "data" / "processed").mkdir(parents=True, exist_ok=True)
        (base / "outputs" / "models").mkdir(parents=True, exist_ok=True)
        (base / "outputs" / "tables").mkdir(parents=True, exist_ok=True)

        # Timestep parquet: keep raw + scaled? Save scaled because train uses
        # scaled values; raw can be recovered by inverse-scaler if needed.
        _safe_write(df_scaled, base / C.TIMESTEP_PARQUET, kind="parquet")
        _safe_write(static_table, base / C.STATIC_FEATURES_CSV, kind="csv")

        np.savez_compressed(
            base / C.SEQUENCES_NPZ,
            X_dynamic=bundle.X_dynamic,
            X_static=bundle.X_static,
            y=bundle.y,
            participant_ids=bundle.participant_ids,
            split=bundle.split,
            anchor_time=bundle.anchor_time.astype("datetime64[s]").astype(np.int64),
            feature_names_dynamic=np.array(bundle.feature_names_dynamic, dtype=object),
            feature_names_static=np.array(bundle.feature_names_static, dtype=object),
            horizon_minutes=np.array(C.HORIZON_MINUTES, dtype=np.int32),
            lookback_steps=np.int32(C.LOOKBACK_STEPS),
        )

        with open(base / C.SCALERS_JSON, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "dynamic": scalers,
                    "static": static_norm_params,
                    "config": {
                        "lookback_steps": C.LOOKBACK_STEPS,
                        "horizons_min": list(C.HORIZON_MINUTES),
                        "split": {
                            "train": C.SPLIT_TRAIN_FRAC,
                            "val": C.SPLIT_VAL_FRAC,
                            "test": C.SPLIT_TEST_FRAC,
                            "buffer_steps": C.SPLIT_BOUNDARY_BUFFER_STEPS,
                        },
                        "n_train_cap": C.N_TRAIN_CAP,
                    },
                },
                fh,
                indent=2,
            )

        _safe_write(boundaries, base / C.SPLIT_BOUNDARIES_CSV, kind="csv")

        summary = _summarise_pipeline(df_scaled, bundle, availability)
        _safe_write(summary, base / C.PREPROCESSING_SUMMARY_CSV, kind="csv")

    return {
        "timestep_df": df_scaled,
        "static_table": static_table,
        "bundle": bundle,
        "scalers": scalers,
        "boundaries": boundaries,
        "availability": availability,
    }


def _safe_write(obj, path: Path, kind: str) -> None:
    """Write a DataFrame to disk, falling back to a timestamped sibling path
    on PermissionError (the typical cause is the user holding the file open
    in Excel). The pipeline never aborts because of a file lock; downstream
    consumers read from the NPZ which is always written first.
    """
    import time as _t
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if kind == "parquet":
            obj.to_parquet(path, index=False)
        elif kind == "csv":
            obj.to_csv(path, index=False)
        else:
            raise ValueError(f"unknown kind={kind}")
    except PermissionError:
        stamp = _t.strftime("%Y%m%d_%H%M%S")
        fallback = path.with_name(f"{path.stem}.{stamp}{path.suffix}")
        if kind == "parquet":
            obj.to_parquet(fallback, index=False)
        else:
            obj.to_csv(fallback, index=False)
        print(
            f"[preprocessing] WARNING: {path.name} was locked; wrote {fallback.name} instead."
        )


def _summarise_pipeline(
    df: pd.DataFrame,
    bundle: SequenceBundle,
    availability: ModalityAvailability,
) -> pd.DataFrame:
    rows = []
    for pid, sub in df.groupby("participant_id"):
        n_train = int((sub["split"] == "train").sum())
        n_val = int((sub["split"] == "val").sum())
        n_test = int((sub["split"] == "test").sum())
        mask = bundle.participant_ids == pid
        seq_total = int(mask.sum())
        seq_train = int(((bundle.split == "train") & mask).sum())
        seq_val = int(((bundle.split == "val") & mask).sum())
        seq_test = int(((bundle.split == "test") & mask).sum())
        rows.append(
            {
                "participant_id": pid,
                "n_rows_total": len(sub),
                "n_rows_train": n_train,
                "n_rows_val": n_val,
                "n_rows_test": n_test,
                "n_sequences_total": seq_total,
                "n_sequences_train": seq_train,
                "n_sequences_val": seq_val,
                "n_sequences_test": seq_test,
                "basal_available": availability.basal_available.get(pid, 0),
                "bolus_available": availability.bolus_available.get(pid, 0),
                "carb_available": availability.carb_available.get(pid, 0),
            }
        )
    return pd.DataFrame(rows)
