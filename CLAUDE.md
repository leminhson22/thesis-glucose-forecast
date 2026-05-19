# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Identity

This is **Son's undergraduate thesis project**: *"A Multimodal Deep Learning Approach for Short-Term Blood Glucose Forecasting in Type 1 Diabetes"*. It is a research codebase, not a production application. The build is a sequence of reproducible notebooks/scripts that produce experimental results, figures, and an academic report.

**Current dataset decision:** The project has switched to **HUPA-UCM as the primary modelling dataset**. Do **not** use T1D-UOM as the training dataset unless the user explicitly asks for a separate comparison or historical discussion.

**Language behaviour:**
- Explain concepts and decisions to the user in **Vietnamese**.
- Write `reports/report.md` sections in **formal academic English** unless the user explicitly requests Vietnamese.
- Write code comments in **clear English**.

## Governing Skill - Read First

`skills/SKILL.md` is the authoritative protocol for this project. Before any methodological decision (preprocessing strategy, model architecture, feature set, metric), consult it and follow its workflow. Key non-negotiable rules:

1. **Never speculate before inspecting data.** For this project, inspect `data/data_hupa/` before proposing training/preprocessing decisions.
2. **Every decision must be evidence-based**, backed by EDA findings or peer-reviewed literature. Mark unsupported claims `[citation needed]`.
3. **No data leakage.** Time-series splits must be strictly chronological. Fit scalers and derived static features on training data only.
4. **All code must run on Google Colab.** Use the `IN_COLAB` detection pattern and a `BASE_PATH` variable; never hardcode local-only paths.
5. **No medical advice.** This is research/decision-support only.
6. **Workflow order matters.** Do not start modelling before the HUPA-specific EDA/preprocessing decisions are documented.

## Primary Dataset: HUPA-UCM

The primary dataset is stored in:

```text
data/data_hupa/
```

It contains **25 Excel files**, one per participant:

```text
HUPA0001P.xlsx ... HUPA0028P.xlsx
```

Each file contains one sheet named `Data` with the same aligned schema:

```text
time
glucose
calories
heart_rate
steps
basal_rate
bolus_volume_delivered
carb_input
```

Current inspection summary:

```text
Participants: 25
Total rows: 309,392
Sampling: unified 5-minute grid
Main columns: 100% non-null in inspected files
Median duration: about 13.3 days
Long participants: HUPA0026, HUPA0027, HUPA0028
Short participants under 10 days: HUPA0006, HUPA0020, HUPA0021
Glucose unit: mg/dL
Observed glucose range: 40-444 mg/dL
```

Important modelling implications:

- There is **no Track A / Track C split** for the main model. HUPA is already a single 5-minute aligned dataset.
- Forecast horizons are fixed as:
  - 30 minutes = `t + 6`
  - 60 minutes = `t + 12`
  - 90 minutes = `t + 18`
- Primary lookback should start with:
  - 24 steps = 120 minutes
  - 36 steps = 180 minutes as an ablation
- Glycaemic zones must use **mg/dL thresholds**:
  - Hypoglycaemia: `< 70`
  - In range: `70-180`
  - Hyperglycaemia: `> 180`
- Values exactly `40 mg/dL` are likely low-end sensor/reporting caps for some participants. Retain them but add a `glucose_low_cap` flag.
- Values `> 400 mg/dL` are severe high/extreme readings. Retain them but add a `glucose_high_extreme` flag and evaluate high-glucose performance separately.
- HUPA appears pre-aligned/resampled. Treat it as a modelling-ready aligned dataset, but document that the raw sensor streams are not available in this folder.

## Dataset Role of T1D-UOM

T1D-UOM is no longer the main training dataset for this project.

The folder:

```text
data/raw/
```

contains the earlier T1D-UOM dataset and extensive EDA/reporting work exists for it. Do not use it for preprocessing, sequence generation, or model training unless explicitly instructed by the user. If referenced, treat it as:

- historical exploratory work,
- optional discussion material about real-world messy multimodal data,
- optional future external validation only if the user asks.

Do not mix HUPA and T1D-UOM in the same training table by default. They have different units, sampling assumptions, metadata, and preprocessing requirements.

## Repository Layout

```text
data/data_hupa/   <- primary HUPA-UCM Excel files for modelling
data/raw/         <- old T1D-UOM files; not used for main training unless requested
data/interim/     <- partially processed intermediate data
data/processed/   <- final feature-engineered data ready for training
notebooks/        <- numbered Colab-compatible Jupyter notebooks
src/              <- reusable Python modules
outputs/figures/  <- saved plots
outputs/models/   <- saved model weights + fitted scalers
outputs/logs/     <- per-epoch training CSVs
reports/          <- academic reports and review notes
references/       <- cached paper PDFs if any
skills/SKILL.md   <- governing protocol
app/              <- deployment artefacts, final stage only
```

Notebook order:

```text
01_data_understanding
02_eda
03_preprocessing_feature_engineering
04_model_training
05_evaluation_xai
06_colab_demo
```

For the current project phase, create HUPA-specific notebooks/scripts rather than extending old T1D-UOM assumptions.

## Environment & Commands

**Platform:** Windows 11, PowerShell. Use Windows-style absolute paths (`E:\claude-co-work\...`) and avoid `cd` chaining.

Common commands:

```powershell
# Install dependencies
pip install pandas numpy matplotlib seaborn statsmodels tqdm scikit-learn torch shap openpyxl

# Validate notebook JSON
python -c "import json; json.loads(open('notebooks/02_eda.ipynb', encoding='utf-8').read())"
```

Colab cell template:

```python
import os
try:
    import google.colab
    IN_COLAB = True
    from google.colab import drive
    drive.mount('/content/drive')
    BASE_PATH = '/content/drive/MyDrive/glucose-thesis/'
except ImportError:
    IN_COLAB = False
    BASE_PATH = os.path.abspath(os.path.join(os.getcwd(), '..'))
```

## HUPA Preprocessing Rules

Start from HUPA Excel files in `data/data_hupa/`.

Required preprocessing outputs:

```text
data/processed/hupa_5min_timestep.parquet
data/processed/hupa_5min_sequences.npz
data/processed/hupa_static_features.csv
```

Minimum timestep columns:

```text
participant_id
timestamp
glucose
calories
heart_rate
steps
basal_rate
bolus_volume_delivered
carb_input
hour_sin
hour_cos
dayofweek_sin
dayofweek_cos
glucose_velocity
glucose_acceleration
glucose_30m_mean
glucose_60m_mean
glucose_120m_mean
bolus_30m_sum
bolus_60m_sum
bolus_180m_sum
carb_60m_sum
carb_180m_sum
steps_30m_sum
steps_150m_sum
calories_30m_sum
heart_rate_30m_mean
glucose_low_cap
glucose_high_extreme
target_30m
target_60m
target_90m
split
```

Static features for the patient embedding branch come from two sources:

**1. True clinical metadata** available in `data/data_hupa/patient_data_characteristic.xlsx` (verified — covers all 25 patients):

```text
participant_id   (HUPA0001P ... HUPA0028P)
gender           (Female, Male)
hba1c_pct        (HbA1c [%], range 6.0 - 9.7)
age_years        (Age [years], range 18.0 - 61.8)
dx_time_years    (Diabetes duration in years, range 0.8 - 39.5)
weight_kg        (Weight [kg], range 51.0 - 104.8)
height_cm        (Height [cm], range 153 - 188)
treatment        (CSII or MDI; 14 CSII / 11 MDI in cohort)
```

Compute BMI as `weight_kg / (height_cm/100)^2`.

**2. Derived static features** computed per participant **from training data only** (no leakage):

```text
subject_mean_glucose
subject_std_glucose
subject_hypo_pct
subject_tir_pct
subject_hyper_pct
bolus_events_per_day
carb_events_per_day
steps_active_pct
mean_daily_steps
mean_heart_rate
data_duration_days
basal_recording_pct      (% of 5-min bins with basal>0; flags partial-missing patients)
modality_availability    (3-bit flag: [basal_recorded, bolus_recorded, carb_recorded])
```

Both groups of static features feed the patient-embedding branch. If a true clinical value is missing for a specific participant (rare in HUPA), do not invent it — flag with a missingness indicator and let the model learn from the indicator.

## Modelling Plan

The main model should use one unified 5-minute HUPA formulation:

```text
X_dynamic: [samples, lookback_steps, dynamic_features]
X_static:  [samples, static_features]
y:         [samples, 3 horizons: 30, 60, 90 min]
```

Recommended experiment ladder:

```text
M0: glucose history only
M1: glucose + time + derived static
M2: M1 + basal/bolus insulin
M3: M2 + carbohydrate input
M4: M3 + activity/calories/heart rate
```

Only keep additional modalities in the final model if ablation shows a meaningful improvement. Complexity must earn its place.

Recommended baselines:

```text
Persistence
Ridge/ElasticNet on engineered lag features
Random Forest or XGBoost if available
LSTM/GRU
Proposed CNN-GRU/attention model only after baselines are working
```

Evaluation:

```text
MAE and RMSE per horizon
Clarke/Parkes Error Grid if implemented for mg/dL
Zone-specific errors: hypo, in-range, hyper
Per-participant performance
Subgroup performance by data duration and modality/event density
```

## Reporting and Reproducibility Discipline

- Every plot in `outputs/figures/` must be regeneratable from a notebook or `src/` script.
- Every numeric claim in `reports/report.md` must trace back to a notebook cell or `src/` script.
- When EDA findings change, update both the notebook and the corresponding report section.
- Do not report HUPA results using T1D-UOM units or assumptions.
- Do not mix mmol/L and mg/dL. HUPA glucose is mg/dL.
- Use chronological splits per participant. Never randomly split time-series windows.

## Known Pitfalls

1. **Excel dependency.** HUPA files are `.xlsx`; use `openpyxl` with pandas. If unavailable, install it in the environment or parse via Excel XML only for lightweight inspection.
2. **Dataset unit mismatch.** T1D-UOM uses mmol/L; HUPA uses mg/dL. Keep them separate.
3. **Pre-aligned data can hide imputation.** HUPA has 100% non-null aligned columns. Document that the dataset is already aligned/resampled and avoid claiming raw sensor completeness.
4. **Static metadata location.** Patient demographics (HbA1c, age, gender, weight, height, DX time, treatment CSII/MDI) live in `data/data_hupa/patient_data_characteristic.xlsx`, not in the per-patient Excel files. Load this once and join on `participant_id` to build the static feature table. Do not claim the dataset lacks static metadata — it has all seven fields for all 25 patients.
5. **Long participants dominate training.** HUPA0027 alone is 53.43% of dataset rows (574 days), HUPA0026 13.12% (141 days), HUPA0028 8.37% (90 days). The three together = 74.92% of records. Naive subject-mixed split will let them dominate gradient updates. Use patient-level CV and consider truncating long participants to first 14 days or applying sample weighting.
6. **Missing-modality patients require masks.** 4 patients have no basal recorded (HUPA0011/0014/0015/0018; 4.96% of rows), 3 have no bolus (HUPA0011/0015/0018), 3 have no carb (HUPA0015/0018/0020). Additionally HUPA0024/0026/0027/0028 have only 40-66% basal coverage. Add binary `basal_available` / `bolus_available` / `carb_available` flags per patient plus a continuous `basal_coverage_24h` feature, and use modality dropout during training. Do not zero-fill silently.
7. **Sensor caps.** Glucose floor at 40 mg/dL (0.38% of all rows; up to 5.82% for HUPA0002) and ceiling at >400 mg/dL (0.04% all rows) are FreeStyle Libre 2 censored values, not measurements. Flag with `glucose_low_cap` and `glucose_high_extreme` binary features; report metrics with sensitivity analysis (include vs exclude censored windows).
8. **P11 is a CSII patient with no basal recorded** — this is a data-quality anomaly (pump users should have continuous basal). Treat as MDI for the modelling feature `treatment` to avoid the model learning "CSII = no basal signal".
9. **No medical advice.** Forecast outputs are research artefacts only.

## When You Don't Know What To Do

1. Re-read `skills/SKILL.md`.
2. Inspect `data/data_hupa/` directly.
3. Check existing notebooks/scripts for assumptions inherited from T1D-UOM.
4. Prefer the simplest defensible choice.
5. Document alternatives and why they were not used.
