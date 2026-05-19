"""Central configuration for the HUPA-UCM glucose-forecasting thesis.

All hyperparameters, paths, and reproducibility constants live here so that
notebooks and scripts can import a single source of truth. Importing this
module never reads files or hits Drive; it just exposes constants.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

SEED = 42

# ---------------------------------------------------------------------------
# Dataset constants (verified against data/data_hupa/ on 2026-05-18)
# ---------------------------------------------------------------------------

SAMPLING_STEP_MIN = 5  # HUPA is on a strict 5-min grid after glUCModel preprocessing
STEPS_PER_HOUR = 60 // SAMPLING_STEP_MIN  # 12
STEPS_PER_DAY = 24 * STEPS_PER_HOUR  # 288

# Carb conversion: 1 serving == 10 g (from Article_data.md §4.2)
CARB_GRAMS_PER_SERVING = 10.0

# Glucose unit and clinical thresholds (mg/dL)
GLUCOSE_HYPO_THRESHOLD = 70
GLUCOSE_HYPER_THRESHOLD = 180

# FreeStyle Libre 2 censored values (Article_data.md, Pitfall #7)
GLUCOSE_LOW_CAP = 40.0           # values == 40 are sensor "LO" reports
GLUCOSE_HIGH_EXTREME_THRESHOLD = 400.0  # values > 400 are "HI" reports

# Patient-level data anomalies (Pitfall #8)
# HUPA0011P is recorded as CSII but has zero basal records. Reclassify so the
# model does not learn "CSII == no basal signal".
P11_TREATMENT_OVERRIDE = ("HUPA0011P", "MDI")

# ---------------------------------------------------------------------------
# Forecasting horizons (fixed in CLAUDE.md)
# ---------------------------------------------------------------------------

HORIZON_STEPS = (6, 12, 18)         # 30, 60, 90 minutes
HORIZON_MINUTES = (30, 60, 90)
MAX_HORIZON_STEPS = max(HORIZON_STEPS)

# ---------------------------------------------------------------------------
# Lookback windows
# ---------------------------------------------------------------------------

LOOKBACK_STEPS = 24                 # 120 minutes — primary configuration
LOOKBACK_STEPS_ABLATION = 36        # 180 minutes — reported as ablation
LOOKBACK_MAX_MISSING_FRAC = 0.0     # any imputed/missing flag -> drop window

# ---------------------------------------------------------------------------
# Rolling-window feature spans (in number of 5-min steps)
# ---------------------------------------------------------------------------

ROLL_GLUCOSE_MEAN_STEPS = (6, 12, 24)   # 30 / 60 / 120 min
ROLL_GLUCOSE_STD_STEPS = (12,)          # 60 min — variability proxy
# Bolus / carb rolling sums were dropped at Step 4-revisit because the
# pharmacokinetic IOB/COB features carry the same information for a
# sequence model with full lookback access (raw bolus is recoverable from
# IOB[t-L:t] via the inverse recurrence). We keep one middle-span sum per
# event stream to give tree baselines a usable cumulative feature without
# duplicating IOB/COB at multiple spans.
ROLL_BOLUS_SUM_STEPS = (12,)            # 60 min only — middle span for trees
ROLL_CARB_SUM_STEPS = ()                # all spans dropped, COB covers
ROLL_STEPS_SUM_STEPS = (30,)            # 150 min only — long span useful for post-exercise window
# Clinical-lens revisit: `calories_30m_sum` was dropped because calories is a
# Fitbit-derived quantity from HR+steps (we already have raw HR + steps_150m)
# and "calories burned" is not a clinically used predictor of glucose.
ROLL_CALORIES_SUM_STEPS = ()
ROLL_HR_MEAN_STEPS = (6,)               # 30 min — smoothed HR baseline
BASAL_COVERAGE_WINDOW_STEPS = STEPS_PER_DAY  # 24 h

# ---------------------------------------------------------------------------
# Pharmacokinetic-decay aggregations
# ---------------------------------------------------------------------------
# IOB (Insulin On Board): exponential decay of bolus contributions.
# tau=75 min reflects the action time-constant of rapid-acting analogs
# (lispro, aspart, glulisine). After 5*tau (375 min) <1% of the dose remains.
# COB (Carbs On Board): exponential decay of carb contributions with
# tau=60 min, a first-order approximation of the meal-absorption model in
# Hovorka et al. (2004). After 5*tau (300 min) <1% remains.
# Implementation: 1-pole IIR recurrence  out[t] = alpha*out[t-1] + x[t]
# with alpha = exp(-dt_min / tau_min).
IOB_TAU_MIN = 75.0
COB_TAU_MIN = 60.0

# ---------------------------------------------------------------------------
# Long-patient handling (see memory: long-patient-strategy)
# Decision 2026-05-18: sample-cap with adaptive stride on TRAIN only.
# ---------------------------------------------------------------------------

N_TRAIN_CAP = 5000      # max sequences per patient in TRAIN
EVAL_STRIDE = 1         # val / test keep every timestep
TRAIN_STRIDE_MIN = 1    # smallest stride allowed on train

# ---------------------------------------------------------------------------
# Chronological train/val/test split (per-patient)
# ---------------------------------------------------------------------------

SPLIT_TRAIN_FRAC = 0.70
SPLIT_VAL_FRAC = 0.15
SPLIT_TEST_FRAC = 0.15
assert abs(SPLIT_TRAIN_FRAC + SPLIT_VAL_FRAC + SPLIT_TEST_FRAC - 1.0) < 1e-9

# Buffer of steps removed at each split boundary to prevent label leakage
# between train and val (and between val and test). MAX_HORIZON_STEPS is
# sufficient because the longest target reaches t+18.
SPLIT_BOUNDARY_BUFFER_STEPS = MAX_HORIZON_STEPS

# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

# Continuous dynamic features that get per-subject Z-score (fit on TRAIN only)
# Raw `calories` dropped at Step 4-revisit: it is a derived quantity from
# Fitbit's internal HR+steps classifier and the rolling sum `calories_30m_sum`
# preserves the aggregate signal.
# Clinical-lens revisit: dropped `glucose_acceleration` (2nd derivative is
# mathematically meaningful but no CGM device displays it and no clinical
# guideline uses it; velocity already captures the actionable trend).
PER_SUBJECT_ZSCORE_FEATURES = (
    "glucose",
    "heart_rate",
    "glucose_30m_mean",
    "glucose_60m_mean",
    "glucose_120m_mean",
    "glucose_60m_std",
    "glucose_velocity",
)
# Event-stream features get log1p then GLOBAL (pooled) Z-score on TRAIN.
# Log1p compresses sparse heavy-tailed event sums; pooling is OK because the
# scale of bolus / carb / steps is comparable across patients of same therapy.
# Dropped at Step 4-revisit (math-redundancy / signal-strength):
#   * raw `bolus_volume_delivered` and `carb_input`  → recoverable from
#     IOB / COB via the inverse recurrence; raw values mostly zero anyway.
#   * raw `steps` → captured by `steps_150m_sum`.
#   * `bolus_30m_sum`, `bolus_180m_sum`, `carb_60m_sum`, `carb_180m_sum`,
#     `steps_30m_sum` → multi-span redundancy with IOB / COB / 150m rollups.
LOG1P_THEN_GLOBAL_ZSCORE_FEATURES = (
    "basal_rate",
    "bolus_60m_sum",
    "steps_150m_sum",
    "heart_rate_30m_mean",
    "insulin_on_board",
    "carbs_on_board",
)

# Time / mask features pass through without scaling.
# Dropped at Step 4-revisit:
#   * dayofweek_sin/cos → weekend effect ≤4 pp TIR, weak signal.
#   * glucose_high_extreme → 0.04% frequency, insufficient to learn pattern.
#   * dynamic basal/bolus/carb_available → constant per-patient, MI=0,
#     fully duplicates the static-vector versions.
PASSTHROUGH_FEATURES = (
    "hour_sin",
    "hour_cos",
    "glucose_low_cap",
    "basal_coverage_24h",
)

# ---------------------------------------------------------------------------
# Static-feature schema (per patient)
# ---------------------------------------------------------------------------

# Clinical static features come from patient_data_characteristic.xlsx +
# computed BMI. `treatment` is one-hot encoded later.
# Dropped at Step 4-revisit: `weight_kg` and `height_cm` because BMI is a
# function of both and ranks higher in the selection analysis; keeping all
# three is a math-redundant superset.
CLINICAL_STATIC_NUMERIC = (
    "hba1c_pct",
    "age_years",
    "dx_time_years",
    "bmi",
)
CLINICAL_STATIC_CATEGORICAL = ("gender", "treatment")

# Derived static features computed from each patient's TRAIN portion only.
# Dropped at Step 4-revisit:
#   * `subject_tir_pct` → tir = 100 − hypo − hyper, math redundancy.
#   * `mean_daily_steps` → correlated with `steps_active_pct`, lower rank.
# Clinical-lens revisit (further drops):
#   * `carb_events_per_day` → measures meal LOGGING frequency, not eating
#     frequency; misleading for patients who do not log meals.
#   * `data_duration_days` → pure data artefact; no clinical meaning and
#     misleading at deployment (a new patient has tiny duration).
#   * `basal_recording_pct` → correlates ~0.9 with `treatment_CSII` (CSII
#     records ~99 %, MDI ~5 %); `basal_available` already encodes the binary.
DERIVED_STATIC_NUMERIC = (
    "subject_mean_glucose",
    "subject_std_glucose",
    "subject_hypo_pct",
    "subject_hyper_pct",
    "bolus_events_per_day",
    "steps_active_pct",
    "mean_heart_rate",
)
DERIVED_STATIC_BINARY = (
    "basal_available",
    "bolus_available",
    "carb_available",
)
# After get_dummies on gender + treatment, drop one column of each pair to
# eliminate one-hot redundancy. The remaining columns are gender_Female and
# treatment_CSII; the complementary indicators are recoverable via 1 − x.
ONE_HOT_DROP_AFTER_ENCODE = ("gender_Male", "treatment_MDI")

STEPS_ACTIVE_THRESHOLD = 99  # bins with steps > 99 considered "active"

# ---------------------------------------------------------------------------
# Output paths (relative to BASE_PATH supplied by the Colab cell)
# ---------------------------------------------------------------------------

TIMESTEP_PARQUET = "data/processed/hupa_5min_timestep.parquet"
SEQUENCES_NPZ = "data/processed/hupa_5min_sequences.npz"
STATIC_FEATURES_CSV = "data/processed/hupa_static_features.csv"
SCALERS_JSON = "outputs/models/scalers.json"
SPLIT_BOUNDARIES_CSV = "outputs/tables/hupa_split_boundaries.csv"
PREPROCESSING_SUMMARY_CSV = "outputs/tables/hupa_preprocessing_summary.csv"
