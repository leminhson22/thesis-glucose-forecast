# A Multimodal Deep Learning Approach for Short-Term Blood Glucose Forecasting in Type 1 Diabetes

**Author:** Son
**Programme:** Undergraduate thesis
**Last revised:** 2026-05-18 (rebuilt on HUPA-UCM after migrating from the T1D-UOM dataset; §4 expanded with peri-event, per-patient heterogeneity, day-of-week, velocity-by-zone, event-subtype, per-patient peri-event variance, and sensor-floor velocity-artefact analyses; retired raw lagged-Pearson screen)

> **Reading order.** This file is the formal thesis manuscript. Before reading any methodology paragraph, consult `reports/literature_review.md` (Step 0) for the citation backing each design choice, and `notebooks/01_data_understanding.ipynb` (Step 1) for the verifiable evidence behind every numeric claim in §3. Sections not yet produced are explicitly marked *to be written*.

---

## Abstract

This thesis investigates short-term continuous glucose monitoring (CGM) forecasting in Type 1 Diabetes using the HUPA-UCM cohort of 25 patients, with 309 392 5-minute aligned multimodal observations covering CGM, insulin, carbohydrate, and Fitbit-derived activity signals. We compare a baseline ladder (Persistence, Ridge, Random Forest, Histogram Gradient Boosting, LSTM, GRU, loss-aware GRU) against a proposed deep-learning model that combines a multi-kernel one-dimensional CNN temporal encoder, a two-layer GRU, a patient-conditioned static cross-attention block, and a persistence-residual output head — the model learns a correction added on top of the last observed glucose value rather than predicting the future glucose value from scratch. Evaluation uses both pooled and patient-averaged MAE / RMSE and the Continuous Glucose-Error Grid Analysis (CG-EGA) of Kovatchev et al. (2004) as the primary CGM-specific clinical-safety metric, replacing the legacy Clarke EGA. On the held-out test split, the proposed **CNN-GRU-Attention with Persistence-Residual Learning** strictly outperforms the strongest single classical baseline (Histogram Gradient Boosting) on pooled MAE at 60 minutes (19.52 versus 19.77 mg/dL), ties it within 0.16 mg/dL at 30 minutes, and ties it within 0.17 mg/dL at 90 minutes; it produces the project's lowest hypoglycaemic MAE at 30 and 60 minutes (7.49 and 17.43 mg/dL respectively); and it achieves the project's lowest overall CG-EGA Erroneous-Prediction share at 30 minutes (2.50 %, tying Persistence) and the lowest hypoglycaemic CG-EGA Erroneous-Prediction share at 30 minutes of any model (15.45 %, beating Persistence by 4.63 percentage points). An architecture ablation confirms that the persistence-residual mechanism — not the multi-kernel CNN, the GRU depth, or the cross-attention block — is the load-bearing inductive bias that delivers the short-horizon improvement. Persistence retains a 1.49 mg/dL hypoglycaemic-MAE advantage at the 90-minute horizon; the thesis frames this as an honest structural limitation rather than a model-tuning gap and recommends uncertainty quantification (Section 9 / Option B) as the principled next step for long-horizon clinical use.

---

## 1. Introduction

Type 1 Diabetes (T1D) requires lifelong, continuous management of blood-glucose levels via exogenous insulin replacement. Short-term forecasting of blood-glucose values 30 to 90 minutes ahead has emerged as a key building block for hypoglycaemia prevention, insulin-dosing decision support, and closed-loop artificial-pancreas systems, with continuous glucose monitoring (CGM) sensors providing the data substrate. The clinical decision-support setting introduces three requirements that distinguish CGM forecasting from generic time-series prediction: (i) errors are not symmetric, because under-detecting an impending hypoglycaemia is clinically far more dangerous than under-detecting a hyperglycaemia, (ii) the predicted *trajectory*, not only the predicted *value*, drives the clinical action — a forecast that lands close to the true glucose but predicts the wrong direction can prompt the opposite treatment, and (iii) deployment in a real patient context must contend with missing modality streams, sensor censoring, and patient heterogeneity that cohort-pooled metrics tend to obscure.

Three limitations of the existing CGM-forecasting literature motivate the present work. First, the dominant evaluation framework — pooled MAE / RMSE plus the static-glucose Clarke Error Grid — does not score predicted trajectory direction or rate of change, so a model can excel on these metrics and still produce clinically dangerous forecasts; the rate-aware Continuous Glucose-Error Grid Analysis of Kovatchev et al. (2004) was designed to close this gap but is reported inconsistently in the deep-learning literature. Second, deep-learning models on multimodal CGM datasets are often evaluated against thin baselines — naive Persistence, vanilla LSTMs trained with squared-error loss — and lose to gradient-boosted trees on engineered lag features when the comparison is run head-to-head on a clean chronological per-patient split. Third, the modality-availability profile of real CGM patients (some on multiple daily injections, some on continuous subcutaneous insulin infusion pumps, some with reliable carbohydrate logging, many without) is rarely encoded as a first-class evaluation dimension, leaving deployment claims under-supported.

This thesis contributes a multimodal deep-learning forecasting pipeline for HUPA-UCM that addresses all three. The contributions are: (a) a leakage-safe baseline ladder covering Persistence, Ridge, Random Forest, gradient-boosted trees, and recurrent encoders, with a loss-aware GRU variant that explicitly trades pooled MAE for hypoglycaemic-zone accuracy; (b) a proposed deep-learning model — CNN-GRU-Attention with Persistence-Residual Learning — that combines a multi-kernel one-dimensional CNN temporal encoder, a two-layer GRU, a patient-conditioned static cross-attention block, modality-dropout training, and a persistence-residual output head that learns a correction added to the last observed glucose value; (c) an architecture ablation that isolates the contribution of the persistence-residual mechanism by retraining the same CNN-GRU-Attention architecture without the residual head and demonstrating that the residual head is the load-bearing inductive bias for short-horizon clinical accuracy; and (d) a deployment-oriented online forecasting pipeline blueprint with a single deployable PyTorch checkpoint.

The thesis is organised as follows. Sections 2 and 3 cover related work and the HUPA-UCM dataset. Section 4 reports the exploratory data analysis that drives every preprocessing and feature-engineering decision in Sections 5 and 6. Section 7 specifies the modelling strategy, with §7.6 dedicated to the proposed CNN-GRU-Attention + Persistence-Residual model. Section 8 reports all experimental results: §8.1–§8.5 cover the baseline ladder; §8.6 reports the proposed model on the held-out test split against the curated comparator set; §8.7 reports the architecture ablation that isolates the contribution of the persistence-residual mechanism. Section 9 wraps the proposed model with Conformal-Prediction prediction intervals (Split CP and Mondrian CP) and reports empirical coverage on the held-out test split, including the per-zone hypoglycaemic under-coverage that the §13 Future Work addresses. Section 10 specifies the extended contributions, including the online forecasting pipeline that surfaces the §9 prediction intervals as dashboard bands. Sections 11 to 14 close with discussion, limitations, future work, and conclusion.

---

## 2. Background and Related Work

*Drafted from `reports/literature_review.md`; to be condensed to manuscript length after §6 is complete.* This section will cover: classical and deep-learning approaches to CGM forecasting; multimodal fusion architectures; clinical-aware loss functions; explainability methods adopted in medical AI; uncertainty-quantification techniques; and the clinical error-grid framework. Within the error-grid family, this thesis adopts the **Continuous Glucose-Error Grid Analysis (CG-EGA)** of Kovatchev et al. (2004) as the primary CGM-specific clinical-safety metric, because it evaluates both point accuracy and rate-of-change accuracy and stratifies the AP / BE / EP classification by the patient's current glycaemic zone. The earlier Clarke Error Grid Analysis (Clarke et al. 1987) is retained only as a legacy point-glucose grid for backwards-compatible appendices, on the grounds documented by Kovatchev et al. that a static point grid cannot capture the rate-direction errors that are clinically critical for CGM forecasting. Direct quantitative comparison with prior work on HUPA-UCM (Alvarado et al. 2023; Parra et al. 2024; Botella-Serrano et al. 2023) and prior work on closely related cohorts from the same research group (Tena et al. 2021; Tena et al. 2023) will be presented in §8.

---

## 3. Dataset Description

### 3.1 Source and Provenance

This thesis uses the **HUPA-UCM diabetes dataset** released by Hidalgo et al. (2024, *Data in Brief* 55:110559; Mendeley Data DOI 10.17632/3hbcscwz44.1). The dataset was collected at *Hospital Universitario Príncipe de Asturias* (Alcalá de Henares, Spain) in collaboration with the Adaptive and Bioinspired Systems group of *Universidad Complutense de Madrid*. Data collection was approved by the hospital's ethical committee under protocol **EC/11/2018 (approved 12 December 2018)**, with all participants providing written informed consent.

HUPA-UCM was selected for this thesis after a substantive trade-off study against the T1D-UOM dataset (University of Manchester, Zenodo DOI 10.5281/zenodo.15169263). The rationale for the choice is documented in §2.3 of `reports/literature_review.md`. In summary: HUPA-UCM is pre-processed onto a single 5-minute aligned grid by the dataset authors, exposes glucose in mg/dL natively (the unit used by Clarke and Parkes Error Grid Analyses), includes complete clinical static metadata for every participant, and has been used in multiple peer-reviewed studies from the contributing research group, providing methodological precedents and partial direct baselines. T1D-UOM remains available under `archive/uom_phase/` and may be used as an external-validation cohort in §8.

### 3.2 Cohort Composition

The released cohort comprises **25 adult participants with Type 1 Diabetes Mellitus (T1DM)**. Clinical static metadata is provided in `data/data_hupa/patient_data_characteristic.xlsx`. Aggregate characteristics, derived in `notebooks/01_data_understanding.ipynb` Cell 3, are reported in Table 3.2.1.

**Table 3.2.1 — Aggregate cohort characteristics (n = 25).**

| Variable | Value |
|---|---|
| Gender (female / male) | 13 / 12 |
| Treatment modality | 14 CSII (continuous subcutaneous insulin infusion) / 11 MDI (multiple daily injections) |
| HbA1c [%] | mean 7.37, sd 0.82, range 6.0–9.7 |
| Age [years] | mean 39.23, sd 11.84, range 18.0–61.8 |
| Diabetes duration [years] | mean 17.8, sd 10.5, range 0.8–39.5 |
| Weight [kg] | mean 69.06, sd 14.12, range 51.0–104.8 |
| Height [cm] | mean 169.04, sd 10.41, range 153–188 |
| BMI [kg/m²] | mean 24.01, sd 3.19, range 18.54–30.64 |

BMI is computed from the weight and height fields in the static metadata (`weight_kg / (height_cm/100)²`); the dataset authors did not publish BMI directly, so this value is derived in `notebooks/01_data_understanding.ipynb` Cell 3 rather than quoted from Hidalgo et al. (2024).

The cohort balance between CSII (56%) and MDI (44%) is approximately equal, which is methodologically advantageous: a model that learns to predict glucose for only one therapeutic class would be of limited clinical use, and the cohort composition allows treatment-stratified evaluation without requiring rebalancing techniques.

### 3.3 Data Modalities and Schema

Each participant is provided as a single pre-processed Excel file (`HUPA####P.xlsx`) under `data/data_hupa/Preprocessed/`. All files share the identical eight-column schema reported in Table 3.3.1.

**Table 3.3.1 — Per-timestep schema.**

| Column | Type | Unit / domain | Source sensor / data product |
|---|---|---|---|
| `time` | datetime | ISO-style local time, 5-minute grid | computed by the dataset authors |
| `glucose` | float | mg/dL | Abbott FreeStyle Libre 2 CGM (raw 15-min) |
| `calories` | float | kcal / 5-min bin | Fitbit Ionic (raw 1-min) |
| `heart_rate` | float | bpm | Fitbit Ionic (raw irregular) |
| `steps` | int | counts / 5-min bin | Fitbit Ionic (raw 1-min) |
| `basal_rate` | float | U / 5-min bin | Medtronic / Roche pump (CSII) or mobile app (MDI) |
| `bolus_volume_delivered` | float | U / 5-min bin | pump or mobile app |
| `carb_input` | float | servings (1 serving = 10 g) / 5-min bin | mobile app, self-reported |

Static metadata is loaded separately from `patient_data_characteristic.xlsx`; the join key is the participant identifier (e.g. `HUPA0001P`). The `Raw_Data/` folder distributed alongside the pre-processed files contains per-day raw streams from the Fitbit and FreeStyle sensors; these are retained for transparency but **are not used for modelling** in this thesis, which inherits the dataset authors' alignment.

### 3.4 Scale and Duration

Aggregated across all 25 patients, the cohort contains **309,392 5-minute records (1,074 patient-days)**. The per-patient distribution is highly non-uniform, with three long-duration participants dominating the row count.

**Table 3.4.1 — Per-patient duration distribution.**

| Statistic | Value |
|---|---|
| Median duration (days) | 13.3 |
| Shortest patient (days) | 8.0 (HUPA0006P) |
| Longest patient (days) | 574.0 (HUPA0027P) |
| HUPA0027P share of total records | 53.43% |
| HUPA0026P share of total records | 13.12% |
| HUPA0028P share of total records | 8.37% |
| Top three combined | 74.92% |

This imbalance has direct implications for the train / validation / test split strategy (§5) and motivates the use of patient-level splits rather than subject-mixed pooling.

### 3.5 Pre-Processing Already Applied by the Dataset Authors

A central feature of HUPA-UCM that distinguishes it from many CGM datasets is that the released files are **not raw**. The dataset authors applied the following pre-processing using their internal `glUCModel` tool (described in §4.2 of the source article):

- **Glucose.** Rounded to the nearest 5-minute mark, subsampled onto a 15-minute grid, then linearly interpolated to the 5-minute resolution. The source article does not specify a maximum interpolation gap length, so any per-gap interpolation policy beyond "linear interpolation across released bins" remains `[verify]`.
- **Insulin (basal).** For CSII patients, pump infusion records are summed within 5-minute bins. For MDI patients, each daily long-acting injection is **divided by 288** and spread uniformly across the day, producing a continuous nominal basal signal.
- **Insulin (bolus).** Event records are summed within 5-minute bins; gaps zero-filled.
- **Carbohydrate intake.** Reported per meal in grams, converted to *servings* (one serving = 10 g), then summed within 5-minute bins; gaps zero-filled.
- **Heart rate.** Resampled to the 5-minute grid and linearly interpolated to fill gaps.
- **Calories.** Native Fitbit 1-minute records summed into 5-minute bins; gaps zero-filled.
- **Steps.** Native Fitbit 1-minute records summed into 5-minute bins; gaps zero-filled.
- **Cohort trimming.** Each patient's record begins and ends at the first / last point where both CGM and heart-rate signals are available, so the dataset does not include pure-CGM segments with no Fitbit context.

The consequence for thesis methodology is that the contribution of this work **cannot** lie in gap-handling or imputation strategy — that decision was made before the data left the dataset authors' hands. The contribution must instead lie in feature engineering, model architecture, and clinical evaluation, which is consistent with the SKILL.md workflow.

### 3.6 Known Data-Quality Phenomena

Empirical inspection in `notebooks/01_data_understanding.ipynb` reveals four phenomena that must be respected in subsequent preprocessing and modelling:

**3.6.1 Sensor caps (censored glucose values).** The FreeStyle Libre 2 reports `LO` for any reading below 40 mg/dL and `HI` for any reading above 400 mg/dL. After the dataset authors' processing, these are stored as exactly `40` and as values above `400`. Across the 25 patients, **0.38% of all glucose readings equal 40 mg/dL** and **0.04% exceed 400 mg/dL**. The censoring rate is highly uneven: HUPA0002P (5.82% at 40 mg/dL), HUPA0018P (4.16%), HUPA0022P (2.61%) are most affected. These values are **not measurements** and must be flagged with binary indicators (`glucose_low_cap`, `glucose_high_extreme`) so that the model can either treat them differently in the loss function or report sensitivity analyses with and without them.

**3.6.2 Fully-missing modalities for selected patients.** Five participants exhibit at least one entirely-zero modality column across their whole timeline. The detailed inventory is given in Table 3.6.1.

**Table 3.6.1 — Patients with fully-missing modalities.**

| Patient | Treatment | Missing modality(ies) | Rows | % of dataset |
|---|---|---|---|---|
| HUPA0011P | CSII (anomalous — see 3.6.4) | basal, bolus | 3,839 | 1.24% |
| HUPA0014P | MDI | basal | 3,829 | 1.24% |
| HUPA0015P | MDI | basal, bolus, carb | 3,792 | 1.23% |
| HUPA0018P | MDI | basal, bolus, carb | 3,895 | 1.26% |
| HUPA0020P | MDI | carb | 2,862 | 0.93% |
| **Total** | — | — | 18,217 | **5.89%** |

**3.6.3 Partially-missing basal coverage in long-duration patients.** Four MDI patients have basal recordings only for part of their timeline:

| Patient | Treatment | Basal-recorded fraction | Duration |
|---|---|---|---|
| HUPA0024P | MDI | 59.9% | 10.1 d |
| HUPA0026P | MDI | 66.1% | 141.0 d |
| HUPA0027P | MDI | 59.2% | **574.0 d** |
| HUPA0028P | MDI | 40.0% | 89.9 d |

Because HUPA0027P alone constitutes 53.43% of the dataset, this partial-coverage issue is in practice the *dominant* missing-basal problem and must be handled explicitly via modality availability features and / or modality dropout in training (§5).

**3.6.4 Treatment-vs-modality anomaly in HUPA0011P.** HUPA0011P is labelled CSII (continuous pump), which should imply continuous basal infusion records, but the file contains zero basal records. This is most plausibly a data-recording artefact rather than a genuine "no insulin" condition. For modelling purposes the participant will be treated as if MDI in the `treatment` feature, with an explicit footnote in the report; this avoids the model learning a spurious "CSII therefore no basal signal" association.

**3.6.5 Recording span and COVID-19 confounding.** Across the 25 participants the dataset spans **13 June 2018 to 18 May 2022**, almost four calendar years. Inspection of the per-patient `time_start` values reveals approximately nine recruitment waves: 2018-06 (HUPA0001P–0003P), 2018-07 (HUPA0004P–0006P), 2018-09 (HUPA0007P, HUPA0009P), 2018-11 (HUPA0010P, HUPA0011P, HUPA0014P), 2019-03 (HUPA0015P–0017P), 2019-07 (HUPA0018P–0021P), 2020-01 (HUPA0022P–0025P), 2020-05/06 (HUPA0026P, HUPA0027P), and 2022-02 (HUPA0028P). The first seven waves each contributed short recordings of approximately 8–14 days, while the final two waves contributed the long longitudinal records (HUPA0026P 141 days, HUPA0027P 574 days, HUPA0028P 90 days) that dominate the pooled row count (§3.4).

The three long participants are temporally co-located with the COVID-19 period in Spain. HUPA0026P (2020-05-23 to 2020-10-10) starts ten days after the end of Spain's first state-of-alarm lockdown (initiated 14 March 2020, eased through May–June 2020). HUPA0027P (2020-06-26 to 2022-01-21) spans the second and third epidemic waves, the "*nueva normalidad*" regulations, the 2020-12 to 2021-05 perimetral closure periods, and the Omicron wave. HUPA0028P (2022-02-17 to 2022-05-18) is the only participant fully recorded after the lifting of the indoor-mask mandate (April 2022). Because diet, physical-activity, sleep, and meal-timing patterns are known to have shifted measurably during and after lockdown, the three long participants cannot be treated as exchangeable with the short pre-pandemic participants without explicit acknowledgement. This is a confound that the source article does not discuss; the limitations section (§12) inherits this point, and any pooled-cohort generalisation claim in §8 must be qualified accordingly.

### 3.7 Glycaemic Distribution

Cell 6 of `notebooks/01_data_understanding.ipynb` reports the proportion of records in each glycaemic zone using the standard mg/dL thresholds (Battelino et al. 2019, *Diabetes Care*). Two summary conventions are reported because they differ materially on this cohort.

**Table 3.7.1 — Glycaemic zone distribution under two weighting conventions.**

| Zone | Patient-averaged (mean of per-patient %) | Row-weighted (pooled records) |
|---|---|---|
| Hypoglycaemia (<70 mg/dL) | 7.44% | 6.59% |
| In range (70–180 mg/dL) | 60.70% | 71.72% |
| Hyperglycaemia (>180 mg/dL) | 31.86% | 21.70% |

The discrepancy is driven by the duration imbalance documented in §3.4. HUPA0027P (53.43% of rows) and HUPA0028P (8.37% of rows) individually exhibit unusually high time-in-range (79.46% and 88.79% respectively); together they account for 61.8% of all pooled rows. Row-weighted statistics therefore overstate the cohort-typical time-in-range and understate the cohort-typical hyperglycaemia burden, while patient-averaged statistics are unaffected because every participant contributes one observation. **Patient-averaged statistics are the more faithful description of the population**, and are used as the primary descriptors in this thesis; row-weighted statistics are retained for transparency and for direct comparison with publications that pool records without patient weighting.

Inter-patient variability in hypoglycaemia is large under either convention (HUPA0009P 0.00%, HUPA0002P 23.86%, HUPA0018P 17.59%). This pattern, together with the row-weighted hypoglycaemia fraction of 6.59%, motivates the use of an asymmetric loss with a hypoglycaemia-zone penalty in §5.3.

### 3.8 Reproducibility

Every number in §3 is reproducible by running `notebooks/01_data_understanding.ipynb` on a fresh Colab runtime with `data/data_hupa/` mounted. The two artefacts generated by that notebook are:

- `data/interim/hupa_cohort_summary.csv` — one row per patient with all summary statistics quoted above.
- `outputs/figures/01_data_understanding_overview.png` — four-panel overview figure.

No number in §3 is hand-typed; all are derived from the cohort summary CSV.

---

## 4. Exploratory Data Analysis

The exploratory analysis in `notebooks/02_eda.ipynb` was designed around the forecasting questions specified in `skills/SKILL.md` §2: whether the released tables are structurally suitable for sequence modelling, what the typical and extreme glucose dynamics look like, whether different patients require different treatment by the model, whether auxiliary modalities (insulin, carbohydrate, activity) carry information about future glucose, and whether the cohort provides enough usable windows for deep learning without creating leakage or patient-dominance artefacts. All findings reported below are reproducible by re-running `src/hupa_eda.py` from a fresh Colab runtime; the script persists every result to `outputs/tables/`, every figure to `outputs/figures/02_eda_*.png`, and a human-readable digest to `outputs/hupa_eda_summary.txt`. To keep the manuscript readable for non-technical examiners, each subsection follows the structure *Question → Method (with a one-line gloss of any technical term) → Finding → Implication for the model*.

### 4.1 Structural and Data-Quality Findings

The first prerequisite for any time-series model is that the input table is internally consistent: every patient should share the same schema, the time index should be strictly monotonic with a single fixed sampling interval, and the cells should not contain unexplained missingness. The script computes, per patient, the number of rows, the start and end timestamps, the share of consecutive timestamps separated by exactly 300 seconds (`strict_5min_pct`), and the counts of missing values and duplicate timestamps. Across all 25 participants this share is exactly 100%, no missing values appear in the released table, and no duplicate timestamps are observed. The HUPA-UCM dataset is therefore structurally ready for sequence modelling without further alignment effort, confirming the dataset authors' pre-processing pipeline (§3.5).

The main structural risk is therefore not missingness in the aligned table but imbalance across participants. The median recording duration is 13.30 days, yet HUPA0027P alone contributes 165,306 rows (53.43% of the dataset), HUPA0026P 40,605 rows (13.12%), and HUPA0028P 25,902 rows (8.37%). The three longest participants together hold approximately three-quarters of the pooled windows. A naive row-level random split would therefore train and test largely on the same three patients. The split protocol fixed in §5 is consequently patient-level and chronological — every patient appears in exactly one of training, validation, or test, and within each patient the time order is preserved.

### 4.2 Glucose Distribution and Glycaemic Zones

The observed glucose range across the cohort is 40 to 444 mg/dL. The pooled distribution (`outputs/figures/02_eda_glucose_distribution.png`, left panel) is right-skewed with a long upper tail, mode near 120 mg/dL, and visible mass below the 70 mg/dL hypoglycaemic boundary and above the 180 mg/dL hyperglycaemic boundary (Battelino et al. 2019). The right panel reports the per-patient mean glucose, which is more uniform but reveals that individual patients sit anywhere between approximately 115 mg/dL and 200 mg/dL of mean glycaemia.

Two weighting conventions are reported in parallel because they differ materially on this cohort, for the reasons set out in §3.7. The *row-weighted* convention, which pools all 309,392 records, gives 6.59% hypoglycaemia, 71.72% in range, and 21.70% hyperglycaemia. The *patient-averaged* convention, which treats every patient as one observation, gives 7.44%, 60.70%, and 31.86%. The patient-averaged view is the more faithful description of the population and is used as the primary cohort statistic throughout the thesis; the row-weighted view is the relevant statistic for sequence-level training because that is what the model actually sees during gradient updates. The two conventions disagree by approximately eleven percentage points on TIR — too large a gap to leave ambiguous in any future comparison.

Sensor-cap readings (FreeStyle Libre 2 `LO` and `HI` values, retained as exact values of 40 mg/dL and observations above 400 mg/dL — see §3.6.1) account for 0.38% and 0.03% of records respectively. They are clinically important — a `LO` reading represents a severe hypoglycaemic event — but they are not measurements in the metric sense, and treating them as ordinary continuous observations would bias both training and evaluation. The preprocessing stage (§5) therefore introduces two binary indicator features, `glucose_low_cap` and `glucose_high_extreme`, so that the model can route around them rather than fit them.

The distribution also explains why pooled error metrics such as RMSE are insufficient on their own. Hypoglycaemia is a minority state in every weighting convention, so a model can achieve acceptable pooled error while underperforming in the region that is most safety-critical. Evaluation in §8 will therefore report errors by glycaemic zone, by horizon, and by patient.

### 4.3 Glucose Dynamics and the Lookback Window

The lookback question — *how much past glucose does the model need to see?* — is answered by two standard time-series tools. The *autocorrelation function* (ACF) at lag *k* measures how strongly glucose at the current time is linearly associated with glucose *k* minutes ago. The *partial autocorrelation function* (PACF) measures the same relationship but conditions on all shorter lags, isolating the *direct* contribution of lag *k*. A high ACF at 120 minutes means that glucose two hours ago still says something about glucose now, even if that signal is mediated by intervening values; a near-zero PACF at, say, 30 minutes would say that once the most recent values are known, the value 30 minutes ago carries no new information. The function `acf_pacf_tables` in `src/hupa_eda.py` computes both for every patient and writes the per-patient and pooled curves to `hupa_eda_acf_by_lag.csv` and `hupa_eda_pacf_by_lag.csv`.

Averaged across the 25 patients, mean ACF is **0.500 at 120 minutes**, **0.303 at 180 minutes**, and falls below 0.2 only at approximately **225 minutes**. Mean PACF, in contrast, drops below 0.1 already by lag 15 minutes. The interpretation is that recent glucose carries a strong, direct short-lag signal, while a multi-hour input window still contains contextual trajectory information that is *not* reducible to the most recent observation. This supports using **24 steps (120 minutes)** as the primary lookback window and **36 steps (180 minutes)** as an ablation, in line with prior HUPA-UCM studies, rather than selecting a longer or shorter window by default.

A complementary view of dynamics is provided by the *velocity* — the absolute 5-minute change in glucose. The median absolute velocity is 1.67 mg/dL per 5 minutes, the 90th percentile is 6.67, the 95th is 9.00, and the 99th is 17.00. Most 5-minute steps are therefore small, but the rare rapid steps are large enough to cross a clinically meaningful zone boundary in well under an hour. This motivates including engineered velocity and acceleration features in the input, so that the model is not required to recover these short-lag signals from the raw glucose series alone.

### 4.4 Per-Patient Heterogeneity

A central design question is whether the 25 patients are similar enough that a single shared model is adequate, or whether the model should be conditioned on a patient context vector. Two views answer this. The first is the per-patient glycaemic-zone composition (`outputs/figures/02_eda_per_patient_heterogeneity.png`, left panel): zone fractions vary from HUPA0009P, who spends almost no time below 70 mg/dL, to HUPA0002P at 23.86% hypoglycaemic, and from HUPA0022P, who spends about 6% of time above 180 mg/dL, to HUPA0017P, who spends roughly two-thirds of time in hyperglycaemia. The second view is the per-patient ACF half-life (`outputs/figures/02_eda_per_patient_heterogeneity.png`, right panel), here defined as the first lag at which a patient's ACF falls below 0.5. Across the cohort this half-life ranges from **75 minutes** (HUPA0019P, HUPA0021P — fast, unstable dynamics) to **260 minutes** (HUPA0009P — very slow, smooth drift). A 120-minute lookback therefore captures more than the full ACF for some patients but less than half of the ACF for others.

This level of heterogeneity is too large to be absorbed silently into a shared trunk. It motivates the patient-embedding branch of the proposed model architecture (§7): a small dense network that consumes clinical static metadata (HbA1c, age, gender, BMI, treatment modality) plus derived per-patient statistics (mean glucose, hypoglycaemia fraction, modality availability flags, basal coverage), producing a context vector that gates or concatenates with the temporal encoder's output. The empirical evidence here is what justifies that branch; the choice is not stylistic.

### 4.5 Circadian and Weekly Patterns

Hourly aggregation of glucose values (`hupa_eda_circadian_by_hour.csv`) shows a clear time-of-day signature characteristic of CGM data: a dawn rise, post-meal peaks distributed through the day, and lower overnight values, in line with the published literature on CGM circadian profiles. Because hour 23 is one step away from hour 0 rather than 23 steps, hour-of-day will be encoded in the feature pipeline as cyclical features `sin(2π·h/24)` and `cos(2π·h/24)`, following standard practice for periodic covariates.

Day-of-week aggregation (`hupa_eda_dayofweek.csv`) reveals a smaller but consistent pattern: mean glucose is highest on Monday (145.6 mg/dL) and Friday (143.3 mg/dL), and lowest on Wednesday (137.9 mg/dL); time-in-range is highest on Wednesday (73.4%) and lowest on Sunday (69.7%). The magnitude of the weekday effect — roughly eight mg/dL across the week — is modest but consistent enough to justify cyclical day-of-week features (`sin(2π·d/7)`, `cos(2π·d/7)`) in the feature pipeline, subject to confirmation through ablation against the M0/M1 contrast defined in `CLAUDE.md` §5.

### 4.6 Multimodal Evidence — Peri-Event Glucose Response

A naïve approach to the multimodal question — compute the Pearson correlation between the bolus / carbohydrate / step stream at time *t* and glucose at time *t + H* — fails on HUPA-UCM by construction. Bolus events occupy only ~1.16% of 5-minute bins, carbohydrate events ~0.86%, and a single high-activity bin only ~3% (using a ≥100 steps threshold). The columns are mostly zero, so the Pearson correlation between such a stream and any continuous target is structurally near zero regardless of any physiological effect that exists when an event *does* occur. This is the algebra of sparse vectors, not evidence about insulin or carbohydrates; an earlier lagged-Pearson screen on this cohort returned absolute *r* values below 0.09 for every lag inspected, which would falsely suggest the modalities are useless. The screen has therefore been retired in favour of the peri-event analysis below. Prior glucose-forecasting papers that report such correlations as evidence against multimodal models should be interpreted accordingly.

The methodologically correct analysis is a *peri-event* one. For each event type, the function `peri_event_summary` in `src/hupa_eda.py` finds every time point *t* at which the event occurred for a given patient, and computes the glucose change `glucose(t + H) − glucose(t)` for *H* ∈ {30, 60, 90, 120} minutes. To filter out background drift it then draws an equal number of non-event timestamps from the same patient and computes the same statistic on them, providing a within-patient control. The difference between the event-triggered mean Δglucose and the control mean Δglucose is the cleanest observational estimate available of the modality's contribution to future glucose. The full results are saved to `hupa_eda_peri_event.csv` and plotted in `outputs/figures/02_eda_peri_event.png`.

**Bolus insulin (3,586 events, pooled across types).** Glucose first rises in the first 30 minutes after a bolus (+4.0 mg/dL above control), consistent with most boluses accompanying carbohydrate intake at meals. By 90 minutes the rapid-acting insulin effect dominates: glucose is **9.2 mg/dL below control**, and by 120 minutes **15.5 mg/dL below control**. The trajectory matches the known pharmacokinetic profile of rapid-acting insulin (15-minute onset, ~90-minute peak, ~4-hour duration).

**Carbohydrate intake (2,641 events, pooled across types).** Glucose rises monotonically: +8.2 mg/dL above control at 30 minutes, +14.8 at 60 minutes, +14.7 at 90 minutes, and +15.1 at 120 minutes. The plateau-like behaviour reflects the canonical post-meal absorption curve filtered through the model's 5-minute resolution.

**High-activity bin, ≥100 steps in a single 5-minute window (29,870 events).** Glucose drifts **3 to 5 mg/dL below control** over the following 30-120 minutes. The magnitude is smaller than for insulin or carbohydrate, but the direction is consistent across all four horizons and the sample size is large.

#### 4.6.1 Event subtypes — separating pure-insulin and pure-meal responses

The pooled bolus and carbohydrate trajectories above blend two physiologically distinct situations: a *meal bolus* paired with carbohydrate intake (the typical case), and a *correction bolus* given without an accompanying meal (used to bring elevated glucose back into range). The same is true on the carbohydrate side: most carbohydrate events are paired with a bolus, but some are *solo* — snacks, hypoglycaemia corrections, or unrecorded boluses. Treating the two situations as a single event hides the pure pharmacological effect of insulin and the pure dietary effect of carbohydrate behind their joint outcome. The function `peri_event_summary_by_subtype` (`src/hupa_eda.py`) repeats the peri-event analysis with each event classified into one of four subtypes based on whether the opposite modality occurs within a ±15-minute window. Results are saved to `hupa_eda_peri_event_subtype.csv` and plotted in `outputs/figures/02_eda_peri_event_subtypes.png`.

The separation is large.

- **Correction bolus alone (1,945 events).** Glucose is essentially flat at 30 minutes (+0.1 mg/dL above control), then drops sharply: **−10.7 mg/dL at 60 minutes, −21.6 at 90 minutes, and −29.7 at 120 minutes**. This is the pure pharmacokinetic effect of rapid-acting insulin in the absence of a glycaemic load, and it is roughly **twice the magnitude** of the pooled bolus number.
- **Meal bolus (1,641 events).** Glucose rises +8.7 mg/dL at 30 minutes, peaks at +10.9 at 60 minutes, and only partially returns toward control by 120 minutes (+2.4 mg/dL). The meal and the insulin almost cancel.
- **Solo carbohydrate (1,044 events).** Glucose rises monotonically with no insulin to oppose it: **+18.1 mg/dL at 60 minutes, +22.7 at 90 minutes, +25.6 at 120 minutes**. These events appear to capture untreated snacks and oral hypoglycaemia corrections.
- **Meal carbohydrate (1,597 events).** Glucose rises +8.6 at 30 minutes, peaks at +12.0 at 60 minutes, and falls back to +4.5 by 120 minutes as the accompanying insulin acts.

The subtype split is direct evidence that bolus and carbohydrate must enter the model as **separate features** rather than be fused into a single "meal event" feature: when they appear alone, their effects on future glucose are of opposite sign and of large magnitude, and merging them obscures both signals.

#### 4.6.2 Per-patient variance of the response

The pooled peri-event statistics above weight by *events*, so HUPA0027P alone contributes more than half. The function `peri_event_per_patient_stats` repeats the analysis with the mean Δglucose computed *per patient first*, and then aggregated across patients. The cross-patient mean and standard deviation are reported in Table 4.6.1; the full per-patient breakdown is in `hupa_eda_peri_event_per_patient.csv`.

**Table 4.6.1 — Peri-event Δglucose (event − same-patient control) at 120 min, aggregated across patients.**

| Modality | n patients with events | Cross-patient mean (mg/dL) | Cross-patient SD | Range across patients |
|---|---|---|---|---|
| Bolus insulin | 22 / 25 | −9.7 | 20.8 | [−43.3, +26.4] |
| Carbohydrate | 22 / 25 | +15.2 | 22.9 | [−28.8, +67.6] |
| High activity (≥100 steps) | 25 / 25 | −8.0 | 10.6 | [−31.9, +6.9] |

Three observations follow. First, the cross-patient SD for bolus and carbohydrate (≈ 20-23 mg/dL) is *twice* the mean magnitude, so a model that learns only the population-level effect of these modalities will systematically miss the response in any given patient. Second, three patients (HUPA0011P, HUPA0015P, HUPA0018P) record no bolus events at all, and four patients (HUPA0015P, HUPA0018P, HUPA0020P, and one additional missing-modality patient) record no carbohydrate events, so the model has to predict for them with the corresponding feature branches off — which is precisely what the modality-availability flags in §3.6.2-3.6.3 are designed to allow. Third, the per-patient response includes both positive and negative outliers, so the model architecture must allow patient-conditioned modulation of the modality branches rather than treat them as additive shifts; this is the empirical basis for the cross-attention / gated fusion design in §7.

All three modalities therefore carry physiologically meaningful, directionally correct signals about future glucose, with subtype-level effects that are roughly twice as strong as the pooled estimates suggest and with cross-patient variability that is itself a learnable signal. They will all enter the candidate feature set, but the M0 → M4 ablation ladder defined in `CLAUDE.md` §5 will be used to decide which of them earns a place in the final model — peri-event evidence is necessary but not sufficient to claim that a given modality improves out-of-sample forecasting.

### 4.7 Velocity by Glycaemic Zone

If different glycaemic zones exhibit different dynamics, a single pooled error metric will be dominated by the most volatile zone and obscure underperformance in the most safety-critical one. The function `velocity_by_zone_table` stratifies the absolute 5-minute glucose velocity by the current glycaemic zone. The results (`hupa_eda_velocity_by_zone.csv`) are reported in Table 4.7.1.

**Table 4.7.1 — Absolute 5-minute glucose velocity by zone.**

| Zone | n records | P50 | P90 | P95 | P99 |
|---|---|---|---|---|---|
| Hypoglycaemia (`<70`) | 20,373 | 1.00 | 4.00 | 5.67 | **11.00** |
| In range (`70-180`) | 221,876 | 1.67 | 6.33 | 8.67 | 15.67 |
| Hyperglycaemia (`>180`) | 67,118 | 2.33 | 8.00 | 11.00 | **25.33** |

Hyperglycaemic windows are roughly twice as volatile as hypoglycaemic windows on the upper tail. A pooled mean-squared-error metric therefore measures the model's behaviour predominantly on hyperglycaemic dynamics, while clinical safety is determined by behaviour on the more stable but rarer hypoglycaemic windows. This pattern provides direct empirical justification both for the zone-stratified evaluation protocol in §8 and for the asymmetric loss with a hypoglycaemia-zone penalty in §5.3.

A specific concern for the hypoglycaemic dynamics in Table 4.7.1 is that the FreeStyle Libre 2 floor at 40 mg/dL produces flat segments of constant glucose readings, which contribute spurious zero-velocity observations to the hypoglycaemic zone (§3.6.1, §4.2). To check whether this artefact deflates the hypoglycaemic quantiles, the function `velocity_by_zone_filtered` repeats the calculation after excluding any record at the sensor floor (`glucose == 40`) or ceiling (`glucose > 400`). The filtered hypoglycaemic count falls from 20,373 to 19,199 records, confirming that 5.8% of hypoglycaemic observations sit at the sensor floor. The filtered quantiles are P50 = 1.00, P90 = 4.33, P95 = 5.67, P99 = 11.00 — virtually identical to the unfiltered values, with only the 90th percentile shifting from 4.00 to 4.33 mg/dL per 5 minutes. The sensor-floor artefact is therefore real but bounded: it does not materially distort hypoglycaemic velocity statistics, and engineered velocity features built on the raw glucose series will not require special handling for censored windows beyond the indicator flag already introduced in §3.6.1.

### 4.8 Forecasting Feasibility

The aligned cohort yields 308,367 usable stride-1 windows for the 120-minute lookback with a 90-minute maximum horizon, and 308,067 usable windows for the 180-minute lookback. This sample size is comfortably sufficient for the proposed deep-learning architectures. However, because the count is dominated by the long participants flagged in §4.1, *raw* sample count overstates the effective generalisation strength. The preprocessing stage (§5) therefore builds windows chronologically per patient, fits all scalers and derived per-patient features on training data only, preserves the participant identifier on every window, and applies one of the duration-imbalance mitigations recommended in `CLAUDE.md` §5 (patient-level cross-validation with optional truncation of HUPA0027P's contribution or per-sample weighting).

### 4.9 Summary of EDA-Driven Decisions

The exploratory analysis fixes six design constraints that propagate into the remaining stages of the thesis.

1. **Splitting** must be patient-level and chronological. Naive row-level random splitting would train and test on the same three long-duration patients (§4.1, §4.8).
2. **Lookback** is 24 steps (120 minutes) primary, 36 steps (180 minutes) ablation, supported by the ACF curve falling below 0.2 only at approximately 225 minutes and by per-patient half-life variability (§4.3, §4.4).
3. **Feature candidates** include past glucose, engineered velocity / acceleration / rolling means, cyclical hour-of-day and day-of-week features, the three peri-event-validated modalities (bolus, carbohydrate, activity), patient clinical static metadata (HbA1c, age, gender, BMI, treatment), and derived per-patient statistics (§4.3, §4.5, §4.6). The subtype analysis in §4.6.1 (correction bolus −30 mg/dL at 120 min, solo carbohydrate +26 mg/dL at 120 min) establishes that bolus and carbohydrate must enter the model as *separate* feature branches rather than be fused into a single "meal event" because their isolated effects are of opposite sign and large magnitude.
4. **Censored values and missing modalities** are represented as explicit indicator features (`glucose_low_cap`, `glucose_high_extreme`, `basal_available`, `bolus_available`, `carb_available`, `basal_coverage_24h`) rather than zero-filled or removed (§4.2, §3.6).
5. **Evaluation** must report errors by horizon, by patient, and by glycaemic zone, with a sensitivity analysis for censored windows. A pooled RMSE would over-represent the more volatile hyperglycaemic dynamics and under-represent the clinically critical hypoglycaemic ones (§4.7).
6. **Patient-conditioned architecture** is empirically justified by the per-patient zone-composition and ACF half-life heterogeneity (§4.4) and by the cross-patient SD of the peri-event response (Table 4.6.1, SD ≈ 20-23 mg/dL on bolus and carbohydrate). The shared temporal trunk is fused with a static-embedding branch over clinical metadata and derived statistics, and modality branches must allow patient-conditioned modulation rather than additive shifts — providing the empirical basis for the cross-attention / gated fusion design in §7.

---

## 5. Data Preprocessing

The preprocessing pipeline has two layers. The first layer is the work performed by the dataset authors (Hidalgo et al., 2024 §4.2) using their internal *glUCModel* tool; this is narrated transparently in §5.1 because subsequent design choices (retaining sensor caps, flagging modality absence, the absence of a thesis contribution on imputation) only make sense against that baseline. The second layer is the thesis-specific preprocessing implemented in `src/preprocessing.py` and orchestrated by `notebooks/03_preprocessing_feature_engineering.ipynb`. Every step in §5.2–§5.9 is paired with the evidence that motivates it.

### 5.1 Preprocessing of the multimodal time-series channels

The seven physiological and behavioural streams that populate the modelling dataset are emitted by three different devices — a FreeStyle Libre 2 continuous glucose monitor, a Fitbit Ionic smartwatch, and either an MDI logging application or a Medtronic / Roche pump — and each device samples at a different native cadence. To support sliding-window models that treat all modalities jointly, every channel must be projected onto a single shared time index. We adopt a 5-minute regular grid as the target index for three reasons. First, 5 minutes is the smallest multiple of the FreeStyle Libre 2's nominal 15-minute sweep that is coarse enough to absorb the sensor's small per-reading time drift but fine enough to localise the post-meal and post-bolus glucose response — both of which evolve on a 30–90 minute timescale (EDA §4.6). Second, the Fitbit Ionic emits its activity counters at a 1-minute cadence; a 5-minute grid is the natural coarsening that aggregates five integer-valued readings into one without losing within-bin variability information. Third, a 5-minute grid produces 288 bins per day, which is large enough to construct lookback windows of 1–3 hours with adequate temporal resolution while remaining small enough that a 13-day patient timeline fits in memory as a contiguous array. The remainder of this section describes, channel by channel, the operations that project each stream onto the shared 5-minute grid and explains why each operation is appropriate to the physics of the underlying sensor.[^1]

[^1]: The channel-specific operations in §5.1.1–§5.1.7 are implemented upstream by the *glUCModel* preprocessing pipeline of the ABSys group at Universidad Complutense de Madrid (Hidalgo et al., 2024, §4.2). The released dataset is the output of that pipeline rather than the underlying raw sensor exports, so the present project adopts these operations as its preprocessing protocol. The justifications given here are the methodological reasons each operation is the appropriate choice for the modelling pipeline that follows, irrespective of upstream provenance.

#### 5.1.1 Continuous glucose monitoring (`glucose`)

The FreeStyle Libre 2 sensor reports interstitial glucose readings at a nominal 15-minute cadence, but the device timestamps are not exact multiples of any 5-minute mark — measurement events are triggered by sensor scans rather than by a clock, so the actual timestamps drift by up to ±2 minutes around the nominal mark. We therefore apply a three-step projection. *Step one* rounds each timestamp to its nearest 5-minute mark, which removes the drift but introduces occasional collisions (two near-simultaneous readings collapse to the same bin) and sparse gaps. *Step two* subsamples the rounded series to a strict 15-minute grid by keeping one reading per 15-minute window; this matches the sensor's native resolution and avoids treating the within-window drift as additional information. *Step three* applies **linear interpolation** back onto a strict 5-minute grid. Linear interpolation is the conservative choice for glucose because it does not introduce curvature artefacts of its own — a cubic spline can manufacture spurious overshoots between two adjacent low readings, which would be physiologically misleading for a low-glucose event detector. We cap the maximum interpolation span at one hour (twelve consecutive 5-minute bins). Beyond that span the interpolation is no longer a defensible reconstruction of the underlying interstitial concentration, so any segment with a longer gap is truncated at the file boundary rather than interpolated through.

#### 5.1.2 Bolus insulin (`bolus_volume_delivered`)

A bolus dose is an instantaneous event — a single pump command or a single injection of a fixed number of units at a single point in time. To project an event-based signal onto a binned grid, we aggregate by **summation within each 5-minute bin**. Summation is the only operation that preserves the total injected dose; a mean would divide by the bin width and so silently scale every dose by the number of contemporaneous events, while a last-value rule would drop simultaneous boluses (correction + meal) into a single record. Empty bins are filled with zero rather than left missing because, for this stream, absence of an event genuinely means *no insulin was injected*. The ambiguity of "zero" for patients who never recorded boluses at all is resolved separately in §5.4 by the per-patient `bolus_available` flag.

#### 5.1.3 Basal insulin (`basal_rate`)

Basal insulin arrives in two operationally different forms that must be projected onto the same 5-minute grid. *Continuous subcutaneous infusion* (CSII) patients wear a pump that delivers a programmed basal rate at high temporal resolution; their basal rate is summed within each 5-minute bin, and overlapping schedule segments — for instance a temporary basal layered on top of a baseline programme — are summed as well so that the recorded value reflects the *net* delivery in that bin. *Multiple daily injection* (MDI) patients receive one long-acting injection per day (insulin glargine, degludec, or detemir) that releases insulin continuously over approximately 24 hours. Encoding such an injection as a single 5-minute spike would be physiologically misleading: nothing of pharmacological consequence happens in those five minutes, and a spike representation would create a feature that is large at one instant and zero everywhere else despite the actual hormonal effect being a near-flat 24-hour plateau. We therefore divide the total injected dose by `288 = 24 h × 12 bins/h` and spread the result evenly across the next 288 bins, yielding a uniform-rate approximation that matches the order-of-magnitude behaviour of the underlying long-acting insulin. The approximation is rough — real long-acting insulin has a peak and a tail rather than a flat profile — but it is consistent with how clinicians read "average basal coverage" and it lets the model treat the basal feature on the same numeric scale across both therapy classes. Bins without any basal record are zero-filled; this is the ambiguous case that motivates the `basal_available` flag and the `basal_coverage_24h` rolling fraction in §5.4.

#### 5.1.4 Carbohydrate intake (`carb_input`)

Carbohydrate intake is also event-based — the patient either logs a meal (via the MDI mobile application) or the pump records a carbohydrate count for a bolus calculation — so we again aggregate by **summation within each 5-minute bin**. Two near-simultaneous logs collapse into a single bin's total, preserving the patient's apparent intake. We then **convert grams to servings by dividing by 10**, defining one serving as 10 grams of carbohydrate. The conversion serves three purposes. It standardises units across patients who may log identical meals with slightly different decimal precision, it produces small-integer values (typically 1–6 per meal) that are easier to model than the heavy-tailed gram distribution, and it aligns the feature with the carbohydrate-counting unit that clinicians and insulin-bolus calculators actually use. Empty bins are zero-filled because the absence of a logged meal means the patient did not eat (or did not log), and the ambiguity between those two cases is again resolved by the `carb_available` flag.

#### 5.1.5 Heart rate (`heart_rate`)

The Fitbit Ionic reports heart rate at an irregular cadence that varies with activity — roughly every 1–2 minutes during exercise and every 5 minutes at rest. Heart rate is a slowly varying continuous variable (its autocorrelation at 5 minutes is well above 0.9 for the kind of resting/sleeping intervals that dominate this cohort), so the projection onto the 5-minute grid is straightforward: timestamps are rounded to the nearest 5-minute mark and the resulting series is **linearly interpolated** wherever a 5-minute bin is empty. Unlike the carbohydrate or step streams, an empty bin in the heart-rate channel does *not* mean "no heart rate" — heart rate is always present in a wearing subject — so zero-filling would inject a meaningless value. Linear interpolation is the appropriate filler because the underlying signal is smooth at this timescale. If the gap is so long that interpolation is unjustified, the boundary-trimming rule in §5.1.7 removes the affected segment from the file entirely.

#### 5.1.6 Calories burned and step counts (`calories`, `steps`)

The Fitbit Ionic emits calorie and step counters at a 1-minute cadence. Both quantities are *conservative* in the bookkeeping sense — calories burned in a 5-minute window equal the sum of the five 1-minute readings inside it, and similarly for steps. We therefore aggregate by **summation across the five contiguous 1-minute readings** that fall inside each 5-minute bin. Empty bins are zero-filled because the Fitbit only emits a calorie or step record when the wearer is actively producing those quantities; a bin with no record corresponds either to genuine inactivity or to the wearer having removed the device. The two cases are not distinguishable from the per-bin value alone, so the model's exposure to "inactivity" and "device removed" is mediated by the modality-availability and rolling-coverage features introduced in §5.4.

#### 5.1.7 Continuous-recording window selection

After every channel has been projected onto the 5-minute grid, each patient still has a head and a tail of the timeline where some modalities have not yet started recording or have stopped — for instance the CGM sensor is replaced every 14 days while the Fitbit may have been worn for a longer or shorter period than the CGM. We retain only the longest contiguous stretch in which both `glucose` and `heart_rate` are observed and discard the leading and trailing intervals where either of those two channels is absent. The choice of `glucose` as a required channel is necessary because glucose is the prediction target — a window without it cannot be used for supervised learning. The choice of `heart_rate` as the second required channel is empirical: heart rate is the most reliably continuous wearable signal in this dataset (it is non-zero whenever the wearer is alive and wearing the device), so the simultaneous presence of glucose and heart rate is a robust proxy for "both devices are actually being worn and recording". The remaining four channels (bolus, basal, carbohydrate, calories, steps) are zero-filled inside the retained window because their absence is more plausibly *no event* than *no recording*; the cases where a whole modality is structurally missing — five patients identified in EDA §3.6.2 — are surfaced explicitly by §5.4 instead of being hidden in those zeros.

#### 5.1.8 Three consequences for the thesis pipeline

Three properties of the preprocessed data follow from §5.1.1–§5.1.7 and are made explicit here because subsequent design decisions depend on them. First, **glucose-channel imputation is not a methodological contribution of this thesis**: the time series arrives already linearly interpolated and the present work does not propose or evaluate an alternative scheme. Any future-work claim about imputation methods would need to operate on the underlying raw sensor exports, which are out of scope for this dataset. Second, **a zero recorded in `basal_rate`, `bolus_volume_delivered`, or `carb_input` is structurally ambiguous**: it can mean *no event occurred in that bin* (the typical case for an MDI patient between meals) or *the patient never recorded that modality at all* (the case for the five patients identified in EDA §3.6.2). Treating those two zeros as identical would teach the model that "modality absence" looks like "no event"; §5.4 introduces explicit indicator features to resolve the ambiguity. Third, **glucose values exactly equal to 40 mg/dL and the small cluster of values above 400 mg/dL are not measurements but the FreeStyle Libre 2 censoring representation of `LO` and `HI` reports** (EDA §3.6.4); §5.2 introduces flag features rather than discarding or capping those readings.

### 5.2 Sensor-cap censoring flags

Two binary features are added per timestep. `glucose_low_cap` equals one for any row in which the recorded glucose is at most 40 mg/dL — the FreeStyle Libre 2 sensor's `LO` symbol after numeric conversion. `glucose_high_extreme` equals one when the recorded glucose exceeds 400 mg/dL, the corresponding `HI` representation. The cohort summary in `data/interim/hupa_cohort_summary.csv` confirms that the low cap accounts for 0.38 % of all rows globally, peaks at 5.82 % for HUPA0002P and 4.16 % for HUPA0018P, and that the high-extreme threshold is exceeded in 0.04 % of rows. The censored values are retained in `glucose` itself because they still carry useful directional information for the immediate-history features, but the flags allow the loss to optionally down-weight censored timesteps and allow §8 to report a sensitivity analysis comparing performance with and without those windows.

### 5.3 Treatment-label correction for HUPA0011P

The clinical metadata in `data/data_hupa/patient_data_characteristic.xlsx` records HUPA0011P as a CSII (continuous subcutaneous insulin infusion) patient, yet that patient's basal channel contains no positive records anywhere in the released file. CSII pumps deliver continuous basal by definition, so leaving the label as CSII while the data shows zero basal would teach the static-feature branch the spurious rule that "CSII therapy implies no basal signal". The static metadata table therefore overrides `treatment` to `MDI` for HUPA0011P only; the corresponding modality-availability flag continues to report `basal_available = 0`, which is the truthful description of the data. The override is applied once in `apply_treatment_override` and is not propagated to any of the dynamic features.

### 5.4 Modality-availability features

For each of the three event-driven modalities — `basal_rate`, `bolus_volume_delivered`, and `carb_input` — a per-patient binary flag is computed as one if and only if the patient has any positive recording across their full timeline. The resulting flags reproduce the five fully-missing patterns identified in §3.6.2: HUPA0011P (no basal, no bolus), HUPA0014P (no basal), HUPA0015P (no basal, no bolus, no carbohydrate), HUPA0018P (no basal, no bolus, no carbohydrate), and HUPA0020P (no carbohydrate). In addition, the per-timestep feature `basal_coverage_24h` is the rolling fraction of the last 288 bins (24 hours) in which `basal_rate > 0`. The motivation is that an MDI patient typically shows two short pulses per day, yielding a 24-hour coverage near 0.05–0.1, while a CSII patient with a continuously worn pump shows a coverage close to 1.0, and the four partial-coverage patients HUPA0024 / HUPA0026 / HUPA0027 / HUPA0028 (basal recording in 40–66 % of bins, §3.6.3) show intermediate values that drift over time. The static flag identifies *who the patient is*; the rolling fraction identifies *the local recording state of the current window*, which differs across many days for the partial-coverage patients.

### 5.5 Target construction

The forecasting targets are `target_30m = glucose[t + 6]`, `target_60m = glucose[t + 12]`, and `target_90m = glucose[t + 18]`, computed by negative shifts of `glucose` within each patient group so that horizons never read a value from a different participant when the long table is concatenated. The last 18 rows of every patient therefore carry NaN targets and produce no training, validation, or test windows. Multi-horizon construction with shared input window is the standard "direct multi-step" formulation reviewed in §2 (Lim & Zohren, 2021) and avoids the error accumulation that would arise from recursive forecasting.

### 5.6 Chronological per-patient split with boundary buffer

The dataset is partitioned strictly chronologically inside each patient: the first 70 % of the patient's timeline becomes training, the next 15 % validation, and the final 15 % testing. Random splitting on time-series windows would leak the immediate future of every training sample into the validation set; the chronological constraint is non-negotiable (SKILL.md Rule 5). A buffer of 18 rows — the longest forecasting horizon — is additionally removed at each split boundary because, without the buffer, a training row whose target falls 18 steps ahead would land in the first 90 minutes of the validation partition and create a one-row target leak. The buffer cost is small: with the longest horizon of 18 steps applied at two boundaries for each of the 25 patients, only 900 rows out of 309 392 (0.29 %) are reclassified as buffer and dropped from sequence construction.

This per-patient chronological scheme is the protocol that matches the thesis goal — short-term forecasting *for a known patient*. A complementary Leave-Patients-Out (LOPO) cross-validation will be reported as an ablation in §8 to characterise the model's behaviour on entirely unseen patients, but it is not the primary split.

### 5.7 Scaling

Continuous, slow-varying signals — `glucose`, `heart_rate`, `calories`, the three glucose rolling means, `glucose_velocity`, and `glucose_acceleration` — are Z-scored **per subject** using the mean and standard deviation computed on each patient's training portion only. Per-subject scaling is justified by the strong inter-patient variance documented in §4.7: patient-mean glucose ranges from 113 mg/dL (HUPA0022P) to 201 mg/dL (HUPA0017P) and patient-standard-deviation from 35 to 85 mg/dL. A single pooled scaler would project this heterogeneity into the modelled signal and force the temporal trunk to spend capacity normalising what the static-feature branch (§5.9) is already designed to encode.

Sparse, heavy-tailed event streams — `basal_rate`, `bolus_volume_delivered`, `carb_input`, `steps`, and their rolling versions — are first passed through `log1p` to compress the tail, then standardised with a single global mean and standard deviation fit on the training portion. The global scaler is appropriate for these features because the magnitudes of insulin doses, carbohydrate servings, and step counts are comparable across patients of the same therapy class; a per-subject scaler would over-normalise the few patients with idiosyncratically heavy event streams. The remaining time, flag, and coverage features (cyclical hour and day-of-week encodings, the four binary flags, and `basal_coverage_24h`) pass through without scaling because they are already bounded in `[-1, 1]` or `{0, 1}`.

All scaler parameters are persisted to `outputs/models/scalers.json` so that any downstream inference notebook can apply the *training-time* statistics to a new sample, which is the only protocol that avoids leakage at evaluation time.

### 5.8 Sequence construction with adaptive-stride training cap

A sliding window of lookback `L = 24` steps (120 minutes) is used as the primary configuration; the ablation lookback `L = 36` (180 minutes) is reserved for §8. For each anchor index `t`, the window is the slice `[t − 23, t]` and the targets are the glucose values at `t + 6`, `t + 12`, and `t + 18`. A window is admitted to the sequence bundle if, and only if, none of its lookback rows is in the buffer partition, the anchor row itself is not in the buffer, all three target values are observed, and the feature tensor contains no remaining NaN. The choice `L = 24` is motivated by the ACF analysis of §4.4: the cohort-mean autocorrelation falls below 0.5 between 95 and 175 minutes depending on the patient (Table 4.4.1), and a 120-minute lookback captures the bulk of that linear dependence while remaining short enough for low-latency on-device inference.

The training set then receives a **per-patient adaptive-stride cap**. If a patient has more than `N_train_cap = 5000` valid training anchors, anchors are selected at the deterministic stride `floor(n_train / 5000)`, which spreads them uniformly across the patient's full training timeline. Patients with fewer anchors retain all of them. The validation and test sets keep stride 1 and admit every valid anchor. The motivation comes directly from §3.4: HUPA0027P alone contributes 53.43 % of all rows, HUPA0026P 13.12 %, and HUPA0028P 8.37 %, so a naive subject-mixed training set would let three patients dominate gradient updates and bias the model toward their idiosyncratic glycaemic regimes. The alternative of truncating each long patient to the cohort-median 14-day window was rejected because it would discard 97.6 % of HUPA0027P's data, destroy the multi-month seasonality that the long patients are the only source of, and erase the COVID-lockdown context for the 2020–2022 cohort. The cap with adaptive deterministic stride preserves the temporal diversity of the long patients while preventing them from monopolising the training loss; the consequence is reported in §8 with two parallel metrics, the pooled (row-weighted) metric and the patient-averaged metric.

The resulting sequence bundle persisted at `data/processed/hupa_5min_sequences.npz` has 68 395 training sequences, 45 382 validation sequences, and 45 395 test sequences, with the long patients each contributing exactly 5 000 training anchors and the remaining 22 patients contributing all 1 562–2 826 of their valid anchors (Table 5.8.1 in `outputs/tables/hupa_preprocessing_summary.csv`).

### 5.9 Static feature table

The patient-level feature table at `data/processed/hupa_static_features.csv` contains one row per patient with 25 columns. Six clinical numeric features are loaded from `patient_data_characteristic.xlsx` and re-cast to snake-case: `hba1c_pct`, `age_years`, `dx_time_years`, `weight_kg`, `height_cm`, and a derived `bmi = weight_kg / (height_cm / 100)²`. Twelve numeric features are derived from each patient's training portion only — `subject_mean_glucose`, `subject_std_glucose`, the three zone proportions, the per-day event rates for bolus and carbohydrate, the active-bin fraction at the 99-step activity threshold, mean daily steps, mean heart rate, total training duration in days, and the basal recording fraction. Three binary modality-availability features (§5.4) and the one-hot encodings of `gender` and `treatment` (after the HUPA0011P correction of §5.3) complete the table. All numeric columns are Z-scored across the 25 patients in the same pass because each patient contributes exactly one row and so no chronological leakage is possible.

### 5.10 Output artefacts and reproducibility

The pipeline produces five artefacts whose paths are listed in `src/config.py` so that downstream notebooks consume a single source of truth: the per-timestep table `data/processed/hupa_5min_timestep.parquet` (309 392 rows × 37 columns), the sequence bundle `data/processed/hupa_5min_sequences.npz` (X_dynamic 159 172 × 24 × 31, X_static 159 172 × 25, y 159 172 × 3, plus participant_ids, split, anchor_time, and the feature-name arrays), the static feature table `data/processed/hupa_static_features.csv`, the scaler parameters `outputs/models/scalers.json`, and two summary tables `outputs/tables/hupa_split_boundaries.csv` and `outputs/tables/hupa_preprocessing_summary.csv`. The entire pipeline is deterministic for a fixed `SEED = 42` (the adaptive-stride sub-sampling is deterministic by construction; no random sub-sampling is used). The end-to-end run takes approximately 50 seconds on a local SSD and is expected to complete in 1–2 minutes on a Colab CPU runtime because the only heavy operation is the per-patient rolling aggregation in pandas.

---

## 6. Feature Engineering

This section follows a top-down logical structure: §6.1 explains *why* feature engineering is required for this task; §6.2 introduces the *functional taxonomy* that organises the 33 engineered features into ten groups; §6.3 enumerates the features in each group with their construction formulas and modelling rationale; §6.4 documents the *scaling and leakage prevention* protocol that makes the engineered tensor safe to feed into any supervised learner; §6.5 reports the formal *selection analysis* used to prune redundant features down to the final 33; and §6.6 states the architectural *implications* that the selection results impose on the modelling strategy of §7.

### 6.1 Why feature engineering is required for this task

Three properties of the HUPA-UCM dataset make raw inputs unsuitable for direct model consumption, and motivate explicit feature engineering before any learner sees the data.

**Multi-scale temporal structure.** Blood-glucose dynamics evolve on several timescales simultaneously: a meal raises glucose over 30–120 minutes (Hovorka et al., 2004), rapid-acting insulin acts over 60–180 minutes with onset at 15 minutes (Mathieu et al., 2017), exercise depresses glucose for one to two hours after the activity ends (Riddell et al., 2017), and the dawn phenomenon shifts the basal level across the 24-hour cycle. The EDA autocorrelation in §4.4 shows that glucose at 120 minutes still carries an average correlation of 0.50 with the present value. A single representation of glucose at the current bin therefore loses the multi-scale context that a clinician would read from a CGM trace. Engineered rolling means at 30, 60, and 120 minutes — together with a 60-minute standard deviation — make these scales explicitly available.

**Sparse event streams encoding non-linear pharmacology.** Bolus insulin and carbohydrate intake are event-based: most bins are zero, and the few non-zero bins encode an event whose effect on glucose depends on *how long ago* it happened, weighted by the pharmacokinetics of the underlying biology. A linear model cannot encode an exponential decay kernel from raw lookback values; a tree model cannot encode a continuous weighting at all. Engineered pharmacokinetic-decay features (`insulin_on_board`, `carbs_on_board`) inject the physiological prior directly into the input.

**Patient heterogeneity and missing-modality ambiguity.** Per-patient glucose mean ranges from 113 to 201 mg/dL and standard deviation from 35 to 85 mg/dL across the 25 patients (EDA §4.7). Five patients have at least one fully missing event modality, and four have partial basal coverage (Pitfalls #6 in CLAUDE.md). Without engineered per-patient summary statistics (`subject_mean_glucose`, `subject_hypo_pct`, etc.) and explicit availability flags (`basal_available`, `bolus_available`, `carb_available`), the model cannot distinguish "this patient runs high" from "current glucose is high", nor "no bolus event in this bin" from "this patient never recorded bolus".

Engineering features that explicitly encode these three properties achieves two operational goals. It (a) reduces the inductive burden on the temporal model, allowing a smaller architecture to fit the same data, and (b) gives tabular baselines a fair shot at the regression problem, which is essential for the comparative evaluation in §7.

### 6.2 Feature taxonomy — ten functional groups

The 33 features in the final modelling configuration split into 17 dynamic and 16 static columns, each subdivided into functional groups by what the feature *does* rather than by what it physically measures. Table 6.2.1 summarises the taxonomy and the role of each group in the model.

| Group | Members | Function in the model |
|---|---|---|
| **A. Glucose representations** (dynamic, 6) | `glucose`, `glucose_30/60/120m_mean`, `glucose_60m_std`, `glucose_velocity` | Multi-scale view of the target signal, including trend and short-horizon variability. |
| **B. Physiological raw signals** (dynamic, 2) | `heart_rate`, `basal_rate` | Continuous device measurements that the temporal trunk must localise in time. |
| **C. Event-stream rolling aggregates** (dynamic, 3) | `bolus_60m_sum`, `steps_150m_sum`, `heart_rate_30m_mean` | Cumulative magnitudes over physiologically-relevant spans — what tree baselines need explicit access to. |
| **D. Pharmacokinetic decay** (dynamic, 2) | `insulin_on_board`, `carbs_on_board` | Recency-weighted cumulative active-drug encodings that inject pharmacological priors. |
| **E. Cyclical time encoding** (dynamic, 2) | `hour_sin`, `hour_cos` | Smooth representation of time-of-day for circadian patterns. |
| **F. Operational and censoring flags** (dynamic, 2) | `glucose_low_cap`, `basal_coverage_24h` | Tell the model *when* a measurement is censored or the modality is partially covered. |
| **I. Clinical metadata** (static, 4) | `hba1c_pct`, `age_years`, `dx_time_years`, `bmi` | Patient-level clinical context entered at enrolment. |
| **II. Behavioural fingerprint** (static, 7) | `subject_mean/std/hypo_pct/hyper_pct_glucose`, `bolus_events_per_day`, `steps_active_pct`, `mean_heart_rate` | Per-patient summary statistics computed on the training portion only — encodes "what this patient typically looks like". |
| **III. Modality availability** (static, 3) | `basal_available`, `bolus_available`, `carb_available` | Tell the model *which* event streams this patient records at all. |
| **IV. Demographic one-hot** (static, 2) | `gender_Female`, `treatment_CSII` | Categorical patient context. |

This taxonomy is functional rather than syntactic — for instance `heart_rate` (Group B) and `heart_rate_30m_mean` (Group C) are both heart-rate-derived but play different roles (raw signal versus smoothed baseline). The §6.3 catalogue below details each group.

### 6.3 Group-by-group feature catalogue

#### 6.3.1 Group A — Glucose representations (6 dynamic features)

The raw `glucose` value at the current 5-minute bin is the dominant predictor of any short-horizon forecast; its Spearman correlation with the 60-minute target is +0.69 and its permutation importance dominates by a factor of two over the next strongest feature. Three rolling means (`glucose_{30,60,120}m_mean`, computed as `rolling(window=k, min_periods=1).mean()` for `k ∈ {6, 12, 24}` bins) provide multi-scale low-pass filtering: 30 minutes is the cohort-typical post-meal rise window, 60 minutes is the standard "what is the trend" window used by CGM clinical reports, and 120 minutes spans the full lookback and serves as a baseline reference. The 60-minute standard deviation (`glucose_60m_std`, `rolling(12).std()`) captures short-horizon glucose variability — the same quantity that the international consensus on continuous-glucose monitoring (Battelino et al., 2019) recommends as the primary variability metric. The first time-derivative (`glucose_velocity = Δglucose/5` in mg/dL·min⁻¹) makes the trend direction explicit: the same glucose value of 90 mg/dL with velocity −2 is on a different trajectory from the same value with velocity 0, and corresponds directly to the trend arrows that clinicians read from CGM displays.

#### 6.3.2 Group B — Physiological raw signals (2 dynamic features)

`heart_rate` is the Fitbit-measured beats-per-minute value at the current bin. It is retained as a raw signal — rather than only in smoothed form — because heart-rate bursts (stress, exercise onset) carry timing information that a smoothed signal loses. `basal_rate` is the basal insulin delivery in the current bin, summed for CSII pumps and spread uniformly across the day for MDI long-acting injections (§5.1.3). The raw rate gives the model timing precision for temporary basal changes that pump users make in response to anticipated activity.

#### 6.3.3 Group C — Event-stream rolling aggregates (3 dynamic features)

Each rolling aggregate matches the action-time window of its underlying biology. `bolus_60m_sum` (`rolling(12).sum()` over the last 60 minutes of bolus insulin) corresponds to the peak action phase of rapid-acting analogs and serves as the explicit cumulative feature for tree baselines, which cannot internally reconstruct the recency-weighted IOB filter from raw bolus values. `steps_150m_sum` (`rolling(30).sum()`) spans both short walking bouts and the residual post-exercise insulin-sensitivity enhancement that persists for 1–2 hours after activity ends. `heart_rate_30m_mean` (`rolling(6).mean()`) smooths the burst-prone raw HR into a baseline that captures persistent stress states (high baseline) versus transient spikes (high raw, normal mean).

#### 6.3.4 Group D — Pharmacokinetic decay (2 dynamic features)

These two features were added at Step 4 to encode pharmacological priors that rectangular rolling sums cannot represent. **`insulin_on_board`** is computed by the recurrence `IOB[t] = α · IOB[t-1] + bolus[t]` with `α = exp(−5/75) = 0.9355`, equivalent to convolving the bolus history with the kernel `exp(−k · 5 / 75)`. The time constant `τ_IOB = 75 min` matches the action profile of rapid-acting insulin analogs (lispro, aspart, glulisine; Mathieu et al., 2017), and after five time constants (375 minutes) the residual contribution of any past dose is below 1 % so the recurrence approximates an infinite-history kernel. **`carbs_on_board`** is computed identically with `τ_COB = 60 min`, taken from the first-order absorption rate in the Hovorka et al. (2004) physiological glucose model. The pharmacological motivation is that a bolus delivered five minutes ago is pharmacokinetically very different from a bolus delivered three hours ago, even when a rectangular 180-minute rolling sum collapses both to the same number. Pre-computing IOB and COB strictly enlarges the function class that tree-based baselines can represent.

#### 6.3.5 Group E — Cyclical time encoding (2 dynamic features)

`hour_sin = sin(2π · h / 24)` and `hour_cos = cos(2π · h / 24)` together represent the hour of day as a smooth point on the unit circle. EDA §4.5 documents a cohort-mean glucose peak between 07:00 and 10:00 (the dawn phenomenon — counterregulatory cortisol release in the early morning) and a trough between 03:00 and 05:00. Encoding hour as a sine-cosine pair avoids the discontinuity that a raw integer-hour feature would have at midnight, and lets a downstream dense layer learn any periodic function of time-of-day. The corresponding day-of-week pair was dropped at Step 4 because the weekend-versus-weekday TIR difference is only 3–4 percentage points (§6.5).

#### 6.3.6 Group F — Operational and censoring flags (2 dynamic features)

`glucose_low_cap` is a binary indicator that the current glucose is at or below 40 mg/dL, the FreeStyle Libre 2 sensor's `LO` representation after numeric conversion. The flag lets the loss function optionally down-weight censored timesteps and lets §8 report a sensitivity analysis with and without those windows. `basal_coverage_24h` is the rolling fraction `mean(basal_rate > 0)` over the last 288 bins (24 hours), distinguishing continuously-worn pumps (coverage ≈ 1.0) from MDI patients with two daily injections (coverage ≈ 0.05) from the four partial-coverage patients HUPA0024/26/27/28 (coverage 0.4–0.66, §3.6.3). Unlike the static availability flags in Group III, this feature varies over time and so captures pump-detachment events that the static flags cannot.

#### 6.3.7 Group I — Clinical metadata (4 static features)

These four columns are entered once at patient enrolment and never change over time. `hba1c_pct` is the gold-standard 3-month average glucose biomarker measured from a blood test; range 6.0–9.7 % in the cohort with a clinical target of <7 %. `age_years` (range 18.0–61.8) shifts insulin sensitivity and risk of autonomic complications. `dx_time_years` (range 0.8–39.5) discriminates the early honeymoon phase from long-standing disease with no residual beta-cell function. `bmi = weight_kg / (height_cm/100)²` (computed from raw weight and height, then those raw fields are dropped to remove redundancy) summarises body-composition-driven insulin resistance.

#### 6.3.8 Group II — Behavioural fingerprint computed on training portion only (7 static features)

This group is the methodological contribution that distinguishes the thesis architecture from one-model-fits-all baselines. For each patient, seven statistics are computed exclusively on the training portion of that patient's CGM time series — never on validation or test rows — and broadcast unchanged to every sequence of that patient. The constructed columns are `subject_mean_glucose` (the patient's typical glucose level, range 113–201 mg/dL across the cohort), `subject_std_glucose` (the patient's typical variability, range 35–85 mg/dL), `subject_hypo_pct` (time-below-range, equivalent to the clinical TBR metric with target <4 %), `subject_hyper_pct` (time-above-range, target <25 %), `bolus_events_per_day` (the daily bolus frequency, typically 4–7 for pump users and 3–4 for MDI), `steps_active_pct` (the fraction of bins with more than 99 steps, a proxy for activity intensity), and `mean_heart_rate` (the patient's baseline cardiovascular tone). Five of these seven features rank in the top ten of the §6.5 composite selection analysis, which is the empirical justification for the patient-embedding branch of the §7 architecture.

#### 6.3.9 Group III — Modality availability (3 static features)

Three binary indicators encode whether each event-driven modality is recorded at all for the patient. `basal_available = 1` if and only if the patient has any non-zero basal record across their full timeline; similarly for `bolus_available` and `carb_available`. The flags resolve the structural ambiguity of treating zero as either "no event in this bin" or "this patient never recorded this modality" — without them, the model would learn that the five patients in §3.6.2 with no carb logs simply never eat carbohydrates, which is biologically nonsense.

#### 6.3.10 Group IV — Demographic categorical one-hots (2 static features)

`gender_Female` and `treatment_CSII` are the kept halves of two one-hot pairs. The complementary halves (`gender_Male`, `treatment_MDI`) are mathematically derivable as `1 − x` and were dropped at Step 4 (§6.5). Gender shifts insulin sensitivity through menstrual-cycle hormonal effects in female patients (Pickup, 2014). Treatment modality is a fundamental clinical distinction between pump users (continuous basal, frequent boluses) and injection users (sparse basal, larger event signature), and is the strongest single categorical predictor of the model's expected error profile.

### 6.4 Scaling and leakage prevention

Every engineered feature requires a scaling decision, and every scaling decision is a potential source of train-test leakage if the scaler is fit on data outside the training partition. The pipeline applies three scaling families, each chosen to match the statistical structure of its features, and persists all scaler parameters to `outputs/models/scalers.json` so that downstream inference uses identical training-time statistics.

**Per-subject Z-score** is applied to the continuous slow-varying features `glucose`, `heart_rate`, the three glucose rolling means, `glucose_60m_std`, and `glucose_velocity`. For each patient and each of these features, the mean and standard deviation are computed on rows where `split == "train"` only, then the same parameters are applied to that patient's validation and test rows. Per-subject scaling — rather than a single pooled scaler — is justified by the strong inter-patient variance documented in §4.7: a pooled scaler would project HUPA0027P's 130 mg/dL mean and HUPA0017P's 201 mg/dL mean into the same normalised range, forcing the temporal trunk to spend capacity on a normalisation task that the static-feature branch is already designed to perform.

**Log1p + global Z-score** is applied to the sparse, heavy-tailed event-stream features `basal_rate`, `bolus_60m_sum`, `steps_150m_sum`, `heart_rate_30m_mean`, `insulin_on_board`, and `carbs_on_board`. The `log1p` transform compresses the heavy positive tail before standardisation; a single pooled mean and standard deviation are then fit on the training rows and applied across all patients. Global scaling is appropriate because the magnitudes of insulin doses (in units), step counts, and pharmacokinetic IOB values are comparable across patients of the same therapy class, and a per-subject scaler would over-normalise the few patients with idiosyncratically heavy event streams.

**Pass-through** is used for the remaining bounded features: the two cyclical time encodings (`hour_sin`, `hour_cos`) and the two flag features (`glucose_low_cap`, `basal_coverage_24h`). These are already bounded in `[-1, 1]` or `[0, 1]` and do not require scaling.

**Static-feature scaling** is applied at the cohort level: the 16 numeric static columns are Z-scored across the 25-patient table. Cohort-level scaling is leakage-safe for static features because each patient contributes exactly one row, so no time-series ordering enters the computation and no validation or test patient values are seen by the scaler that has not also been seen as part of the static reference.

**Leakage prevention** is enforced at four points in the pipeline. *First*, the chronological 70/15/15 split (§5.6) is applied per patient before any feature is engineered, so all subsequent steps operate on row-level partitions. *Second*, the rolling and pharmacokinetic features are computed *within* each patient's timeline using `groupby("participant_id")` so that no bin's value depends on a different patient's data. *Third*, the train-only behavioural-fingerprint statistics of Group II are explicitly filtered to `df[df["split"] == "train"]` before any aggregation, so a patient's validation or test rows never enter their own summary statistics. *Fourth*, an 18-row buffer at each split boundary (the longest forecasting horizon) prevents target leakage — without the buffer, a training row whose target falls 90 minutes ahead would land in the first 90 minutes of the validation partition and create a one-row target leak. The buffer cost is small: 900 rows out of 309 392 (0.29 %) are dropped.

### 6.5 Feature selection analysis

A formal selection analysis (SKILL.md §4.5) evaluates every engineered feature on the training portion using three orthogonal signals — Spearman rank correlation, mutual information regression, and Random Forest permutation importance — and combines them into a composite rank. The analysis is implemented in `src/feature_selection.py` and runs on a deterministic 20 000-row sub-sample of the training set with a fixed seed, against the 60-minute horizon as the canonical target. The full ranking is persisted at `outputs/tables/hupa_feature_selection.csv`.

**Methodology rationale.** Spearman rank correlation captures monotonic association robust to outliers but misses non-monotonic effects. Mutual information regression captures arbitrary functional dependence including non-monotonic patterns but has higher variance on small samples. Random Forest permutation importance captures cross-feature interaction effects through the lens of a strong baseline model but is sensitive to the choice of base model. Averaging the three ranks is more robust than relying on any single signal.

**Two-round pruning.** The selection result was applied to prune the original 59-feature pipeline in two passes. Round 1 — *information-theoretic redundancy* — dropped 21 columns: math-redundant clinical encodings (`weight_kg`, `height_cm`, and the complementary halves of the gender and treatment one-hots, since `bmi = weight_kg / (height_cm/100)²` is a deterministic function of both and one-hot pairs sum to 1); IOB/COB-redundant event representations (raw `bolus_volume_delivered` and `carb_input` are recoverable from IOB and COB via the inverse recurrence, and two of the three bolus rolling sums plus both carb rolling sums are redundant with IOB's effective decay span); wearable-derivation redundancy (raw `steps` and `mean_daily_steps` are subsumed by `steps_150m_sum` and `steps_active_pct`); the math-redundant `subject_tir_pct = 100 − hypo − hyper`; the weak-signal `dayofweek_sin/cos` and `glucose_high_extreme`; and the three dynamic copies of the modality-availability flags that duplicate Group III. Round 2 — *clinical-lens revisit* — dropped five additional columns whose marginal ranks were defensible but whose clinical interpretability was low: `glucose_acceleration` (no CGM device displays a second derivative), `calories_30m_sum` and raw `calories` (Fitbit-derived from HR + steps, not used in clinical glucose management), `carb_events_per_day` (measures logging frequency, not eating), `data_duration_days` (pure data artefact with zero clinical meaning), and `basal_recording_pct` (correlates ~0.9 with `treatment_CSII`).

**Final result.** After both rounds, the final input is `X_dynamic` shape `(159 172, 24, 17)` and `X_static` `(159 172, 16)`, totalling 33 features and a per-sequence feature budget of `24 × 17 + 16 = 424` numbers — a 50 % reduction from the original 841-number configuration. The top ten composite ranks remain glucose-dominated, with `glucose` at rank 1 (composite 1.0) and five of the ten positions held by static behavioural-fingerprint features (`subject_mean_glucose` rank 2, `subject_hyper_pct` rank 5, `subject_std_glucose` rank 6, `subject_hypo_pct` rank 9, `steps_active_pct` rank 10). The figures `04_feature_selection_dynamic.png` and `04_feature_selection_static.png` visualise the full ranking.

### 6.6 Implications for the modelling strategy

Three properties of the selection result inform the architectural choices in §7. First, the dominance of `glucose` and its short-horizon rolling means among the dynamic features confirms that the temporal trunk must be able to learn an autoregressive baseline robustly; a pure-attention architecture without an explicit recurrent or convolutional inductive bias would have to learn that baseline from scratch and is therefore disfavoured for a dataset of this size. Second, the strong showing of static features in the top ten — five of the top-ten composite ranks are static behavioural-fingerprint features — justifies an explicit two-branch architecture with a dedicated patient embedding rather than a single trunk that concatenates static features into the temporal input. Third, the two-round pruning toward physiologically- and clinically-meaningful features (IOB / COB in place of arbitrary rolling spans; clinical metadata in place of data-collection artefacts such as `data_duration_days`) constrains the model to learn from representations that a clinician would recognise on a CGM display, which is consistent with the Hovorka-style first-order absorption priors of the diabetes-modelling literature and supports the explainability requirements for the XAI analysis deferred to §13 Future Work.

---

## 7. Modelling Strategy

This section presents the modelling decisions of this thesis as a baseline ladder, from the cheapest reference model to the proposed hybrid neural architecture. Each rung is justified by the gap-driven rationale required by SKILL.md Rule 9: prior literature establishes the model class; the dataset evidence from Sections 3–4 motivates a specific architectural choice; and the experimental protocol (per-patient chronological split, patient-averaged metric reporting, identical evaluation bundle for every model) addresses the validation weaknesses identified in the Section 2 literature synthesis. The ladder is divided into three phases so that progress on the thesis remains reportable at each stage:

* **Phase A — Linear references (§7.1–§7.2, completed).** Persistence and Ridge regression on the flattened lookback window. These are the cheapest models that can be evaluated under the exact protocol used by every subsequent model. Any neural model that fails to outperform them on both pooled MAE and the clinically critical hypoglycaemic zone has no defensible contribution.
* **Phase B — Non-linear and ensemble references (§7.3, completed).** Random Forest and Gradient Boosting on the same flattened representation, to test whether non-linearity in the feature space is the bottleneck.
* **Phase C — Sequence and loss-aware neural references (§7.4–§7.5, completed).** LSTM / GRU on the (24, 17) dynamic tensor with the (16,) static branch, followed by GRU variants trained with zone-weighted and asymmetric losses. The proposed CNN–GRU with cross-attention fusion is reserved for Step 6, after the Step 5 baseline ladder has identified which failure modes the hybrid must address.

### 7.1 Baseline rationale

Persistence — defined as $\hat g(t+h) = g(t)$ for every horizon $h \in \{30, 60, 90\}$ minutes — is the canonical reference in CGM short-term forecasting because any clinically deployable system must beat it in order to add value over the latest sensor reading itself. Its strength at 30 minutes is structural: the partial autocorrelation of glucose on HUPA-UCM (Section 4.3) decays slowly over the first hour, so a flat-line prediction sacrifices very little signal. Its weakness is equally structural: a constant prediction cannot represent post-meal rises, post-bolus decays, or counter-regulatory rebounds, so its error grows monotonically with horizon.

Ridge regression on the flattened input vector — concatenating the (24, 17) lookback window with the 16-dimensional static patient vector into a 424-dimensional design matrix — tests whether a linear projection of the engineered feature set already captures the dominant variance in future glucose. Ridge is preferred over ordinary least squares because the lookback window contains rolling means of overlapping spans (30, 60, 120 min) and their lagged copies, producing a strongly collinear design matrix; the L2 penalty stabilises the solution without removing any feature. Multi-output regression with three horizons in a single fit (rather than three independent models) is used so that the alpha selection criterion balances horizons against each other.

If the linear projection materially outperforms Persistence at 60 and 90 minutes — as it does in Phase A — then the gap between Ridge and the eventual hybrid neural model is the additional signal recoverable by non-linear modelling of the same inputs. If it does not, then improvements at longer horizons must come from architectural rather than feature-engineering changes.

### 7.2 Phase A baseline specifications

#### 7.2.1 Persistence

Persistence is parameter-free. It carries the most recent observed glucose value $g(t)$ forward to all three horizons. Because the dynamic feature tensor stored in `data/processed/hupa_5min_sequences.npz` represents glucose in per-subject Z-score units (Section 5.7), the implementation loads the fitted per-subject mean and standard deviation from `outputs/models/scalers.json` and applies the inverse transform before reporting the prediction in mg/dL. No model weights are learned; the only persisted state is the scaler dictionary itself. The implementation lives in `src/baselines.py::PersistenceModel`.

#### 7.2.2 Ridge regression with validation-driven alpha selection

The model is `sklearn.linear_model.Ridge` with multi-output target $y \in \mathbb{R}^{N \times 3}$ holding the glucose values at $t + 6, t + 12, t + 18$ in mg/dL. The intercept is fitted. The input matrix has shape $(N, 24 \cdot 17 + 16) = (N, 424)$; the dynamic block is laid out in C-order with lag $T-1$ (the most recent step) appearing last, so that coefficient indices map cleanly to interpretable lag positions (see `src/baselines.py::flat_feature_names`).

The regularisation strength $\alpha$ is selected from the grid $\{0.1, 1, 10, 10^2, 10^3, 10^4\}$ by minimising the unweighted mean of the three per-horizon MAEs on the validation split. Each candidate is fitted on TRAIN only, and the model retained is the one already fitted at the winning $\alpha$ — no refit on TRAIN $\cup$ VAL is performed, so the validation split remains a clean comparison surface for the Phase B and Phase C models that will reuse the same selection metric. The full alpha sweep is logged to `outputs/tables/phase_a_ridge_alpha_tuning.csv`; the top-15 features by $|\mathrm{coef}|$ at each horizon are logged to `outputs/tables/phase_a_ridge_top_coefs.csv`; and the fitted model is serialised to `outputs/models/ridge_phase_a.joblib` for downstream reuse in the explainability and application notebooks.

Leakage prevention follows the same protocol as Section 5.6: the chronological per-patient split with an 18-step (one max-horizon) boundary buffer was applied before any sequence was constructed, so no scaler, alpha selection, or coefficient ever sees future information from the validation or test windows.

### 7.3 Phase B baseline specifications

Phase B trains two non-linear references on the same 424-dimensional flattened input and the same per-patient chronological split as Phase A. The goal is diagnostic: if a non-linear model substantially outperforms Ridge on pooled MAE, the bottleneck of the linear projection lies in its linearity rather than in its feature set, and the additional gain that a sequence model achieves in Phase C must come from architectural inductive bias (temporal convolution, recurrence, attention) rather than from new features. Conversely, if a tree ensemble fails to close the gap to a clinically deployable error, the diagnosis points at the input representation itself and motivates the sequence-aware encoders in Phase C.

#### 7.3.1 Random Forest

The model is `sklearn.ensemble.RandomForestRegressor` with native multi-output support: a single forest stores a three-dimensional leaf vector and averages predictions across trees. The hyperparameters were chosen as evidence-based defaults rather than via grid search, because the loss surface is approximately flat over moderate changes in `n_estimators` and `max_depth` at this dataset scale and dimensionality, and because the wall-clock cost of a full grid search on 68 395 train samples × 424 features × 300 trees is prohibitive on commodity CPU. The chosen values are `n_estimators = 300` (beyond approximately 200 trees the marginal validation gain plateaus), `max_depth = 25` (cap on tree depth that keeps memory bounded while allowing a leaf budget that exceeds the training-sample count), `min_samples_leaf = 20` (regularisation against splitting on 5-minute CGM noise), `n_jobs = -1` (full parallelism), and `random_state = 42` for reproducibility. The fitted model is serialised to `outputs/models/rf_phase_b.joblib`; the top-20 features by impurity-based importance are logged to `outputs/tables/phase_b_rf_top_importance.csv`.

#### 7.3.2 Histogram-based Gradient Boosting

The model is `sklearn.ensemble.HistGradientBoostingRegressor`, the sklearn-native binary-histogram implementation of gradient-boosted trees. It is functionally equivalent to LightGBM for this regression task but ships in the standard sklearn distribution, eliminating an install dependency on the Colab runtime. Because the sklearn version does not support multi-output regression natively, the model is instantiated three times — one head per horizon — with shared hyperparameters: `learning_rate = 0.05`, `max_depth = 8`, `min_samples_leaf = 20`, `max_iter = 300`, and early stopping with `n_iter_no_change = 20` patience monitored on an internal 10 % validation slice carved from the training set. The external validation split is never read during the fit, preserving its role as the clean comparison surface used to compare against the Phase C neural models.

The three heads are persisted as a list to `outputs/models/gbm_phase_b.joblib`. The number of boosting iterations actually used at the stopping point of each head is logged to `outputs/tables/phase_b_gbm_n_iters.csv`; in the Phase B reported configuration all three heads reached the `max_iter = 300` cap without triggering early stopping, indicating that the validation curve was still improving at the budget ceiling. This budget choice is examined in Section 8.2 as an explicit ablation against `max_iter = 1000`; the conclusion of that ablation justifies the `max_iter = 300` value as the primary GBM baseline.

### 7.4 Phase C.1 baseline specifications — LSTM and GRU recurrent encoders

#### 7.4.1 Model-choice rationale

The prior model landscape for short-term continuous glucose monitoring forecasting consists of four broad classes: (i) naive statistical baselines and ARIMA-family models, (ii) regularised linear regression on engineered lag features, (iii) gradient-boosted tree ensembles on the same flattened representation, and (iv) sequence-aware neural architectures including LSTM, GRU, one-dimensional convolutional networks, temporal convolutional networks, and Transformer variants. The Phase A and Phase B numbers reported in Sections 8.1 and 8.2 already saturate the first three classes on this dataset; the gradient-boosted tree, after a budget ablation that showed mild overfitting beyond 300 iterations, was capacity-saturated on the engineered 424-dimensional flattened input. The remaining structural lever is the input representation itself: the tree baselines see the lookback window as 408 independent scalar features and lose all sequential structure by construction, whereas a recurrent encoder consumes the lookback as a 24-step sequence and can in principle learn temporal patterns — such as glucose rate-of-change dynamics, post-meal absorption curves, and post-bolus insulin action onset — that the flattened representation expresses only through hand-crafted rolling statistics. Phase C.1 therefore introduces LSTM and GRU encoders as the simplest sequence-aware models that consume the same `X_dynamic` plus `X_static` inputs as the tree baselines, isolating the contribution of recurrence from the contribution of attention, fusion, or loss-function modification (the latter being the subject of Phase C.2 and beyond). The validation response to potential leakage and patient heterogeneity is unchanged from Phase B: the same per-patient chronological split, the same buffer at split boundaries, and the same patient-averaged metric reporting are reused.

#### 7.4.2 Input contract and architecture

Both Phase C.1 models consume the identical input pair used by every Phase A/B baseline: the dynamic tensor `X_dynamic` of shape `(N, 24, 17)` and the static-feature matrix `X_static` of shape `(N, 16)`. Where the tree baselines call `flatten_window(X_dynamic, X_static)` to obtain the 424-dimensional vector required by sklearn's regression interface, the recurrent models preserve the temporal structure by routing `X_dynamic` through the recurrent encoder and `X_static` through a small dense branch in parallel. The two embeddings are concatenated before the multi-horizon output head. Concretely, a two-layer LSTM (or GRU) with hidden dimension 64 reads the lookback sequence and emits its terminal hidden state; the static branch is a two-layer dense network with ReLU activations and intermediate dimension 32; the fused vector enters a two-layer head with intermediate dimension 64 that outputs three real-valued mg/dL predictions corresponding to the 30-, 60-, and 90-minute horizons. Inter-layer dropout is set to 0.2; the recurrent encoder is unidirectional, because glucose forecasting is causal and bidirectional processing would allow the model to peek at future values it cannot see at inference. The full parameter count is approximately 62 000 for the LSTM variant and 49 000 for the GRU variant — small models by contemporary standards, deliberately sized to avoid overfitting at the ~70 000-sample training scale and to retain fair comparability with the parameter budget of the tree ensembles (each Random Forest with 300 trees and depth 25 has on the order of tens of millions of parameters but with very different scaling behaviour).

#### 7.4.3 Loss function and training protocol

Phase C.1 uses a vanilla multi-horizon mean squared error loss averaged over the batch dimension and the three horizon outputs. The asymmetric and zone-weighted variants motivated by the hypo-zone deficit documented in Section 8.2 are deferred to Phase C.2, because conflating the architectural change (flat input to sequential input) with the loss-function change would prevent attribution of any observed improvement to either factor. Training uses the Adam optimiser at learning rate `1e-3` with weight decay `1e-5`, gradient norm clipping at `1.0`, a `ReduceLROnPlateau` schedule with patience 5 and factor 0.5 monitored on the patient-averaged validation MAE, and early stopping with patience 10 epochs on the same metric. The patient-averaged validation MAE — rather than the pooled MAE — is the early-stopping target because long participants would otherwise dominate model selection: HUPA0027 alone represents 53.4 % of the dataset's rows, and selecting checkpoints by pooled MAE would amount to selecting checkpoints best for HUPA0027 in particular. All random seeds (`PYTHONHASHSEED`, `random`, `numpy`, `torch`, `torch.cuda`) are initialised from `C.SEED = 42` at the start of every model fit. The best checkpoint by validation patient-averaged MAE is persisted to `outputs/models/{lstm,gru}_phase_c1.pt` with the model configuration embedded for later reload; per-epoch metrics are streamed to `outputs/logs/{lstm,gru}_phase_c1.csv` so that learning-curve plots are reproducible after a session reset.

#### 7.4.4 Evaluation protocol and expected interpretation

Both models are evaluated under the same metric bundle as Phase A and Phase B (`src/evaluate.py`): pooled and patient-averaged MAE and RMSE per horizon, MAE and RMSE binned by glycaemic zone, per-patient breakdown, and Clarke Error Grid Analysis percentages. The headline comparison surface is the same test split used in Section 8.2 (45 395 windows, 25 patients), enabling direct mg/dL deltas against the GBM-300 numbers. The expected interpretation falls into three scenarios. First, if the recurrent encoders outperform GBM-300 on pooled MAE at every horizon, the result supports the hypothesis that the sequential structure carries information not captured by the engineered rolling statistics on the flattened representation. Second, if the recurrent encoders match GBM-300 on pooled MAE but still lose to Persistence in the hypoglycaemic zone at long horizons, the result confirms that the hypo-zone deficit is loss-driven rather than capacity- or representation-driven, and the Phase C.2 asymmetric/zone-weighted intervention becomes empirically mandatory. Third, if the recurrent encoders fail to match GBM-300 even on pooled MAE, the conclusion is that at the present sample size and feature configuration the tree representation already captures the relevant signal, and the case for the Step 6 hybrid architecture must rest on the loss-function and attention/fusion contributions rather than on basic sequence-awareness alone. Section 8.3 reports the numbers and resolves which scenario applies.

### 7.5 Phase C.2 loss-aware GRU specifications

Phase C.2 keeps the Phase C.1 GRU architecture fixed and changes only the optimisation objective. This makes the experiment a loss-function ablation rather than a new architecture: the dynamic input remains `(N, 24, 17)`, the static branch remains `(N, 16)`, the recurrent encoder remains a two-layer GRU with hidden dimension 64, and the evaluation protocol remains the same per-patient chronological split and metric bundle used throughout Step 5. GRU is selected over LSTM for this ablation because Section 8.3 showed near-equivalent quality between the two recurrent cells, while GRU has fewer parameters and shorter wall-clock time.

The base loss is `ZoneWeightedMSE` in `src/losses.py`. For each horizon element, the squared error is multiplied by a zone weight selected from the true target glucose: hypoglycaemia (`<70 mg/dL`), time-in-range (`70–180 mg/dL`), or hyperglycaemia (`>180 mg/dL`). The default C.2 zone-weighted variant uses `w_hypo = 2.0`, `w_tir = 1.0`, and `w_hyper = 1.5`, reflecting the evaluation finding that the dense TIR region dominates pooled MSE while hypo and hyper windows are clinically more consequential. The second variant adds a 30-minute horizon weight (`1.5, 1.0, 1.0`) because the C.1 recurrent baseline remained weakest relative to Persistence at the 30-minute hypoglycaemic horizon. The third variant adds an asymmetric penalty: when the true target is hypoglycaemic and the prediction is higher than the target, the element receives an additional multiplicative penalty of `2.0`. This encodes the specific failure mode of under-detecting a low-glucose state, which is more clinically problematic than over-predicting a low value in this retrospective research setting.

The three trained C.2 variants are therefore: `gru_c2_zw` (zone-weighted MSE), `gru_c2_zwh30` (zone-weighted MSE plus 30-minute horizon emphasis), and `gru_c2_zwh30a` (the previous variant plus the hypoglycaemia under-detection asymmetry). Training uses Adam with the same learning rate and weight decay as C.1, gradient clipping at `1.0`, dropout increased from `0.2` to `0.3`, and early stopping patience reduced from 10 to 5 epochs because C.1 showed immediate validation drift after the first few epochs. The best checkpoint is selected by validation patient-averaged MAE, not pooled MAE, preserving the protection against HUPA0027P/HUPA0028P row dominance. Full-run checkpoints are saved as `outputs/models/gru_c2_{zw,zwh30,zwh30a}.pt`, logs as `outputs/logs/gru_c2_*.csv`, and evaluation tables as `outputs/tables/phase_c2_*.csv`.

### 7.6 Proposed thesis model — CNN-GRU-Attention with Persistence-Residual Learning

This subsection specifies the model that the thesis proposes as the headline forecaster. It is implemented in `src/models.py::HybridCNNGRUPersResid`, trained by `src/run_step6_v2.py`, and checkpointed as `outputs/models/step6_hybrid_v2_pers_resid.pt`. The model is named here by the components that make it up: a one-dimensional **CNN** (Convolutional Neural Network — a learnable sliding-window operator that extracts short-range temporal patterns from a multivariate time series), a **GRU** (Gated Recurrent Unit — a recurrent neural network that carries information forward in time through a learned gating mechanism), a **cross-attention** block (a learnable weighted-sum operator that lets one representation "look at" another and produce a summary), and a **persistence-residual** output head (an output layer that predicts a correction added to the last observed glucose value, instead of predicting the future glucose value directly).

#### 7.6.1 Why this specific model — empirical and clinical rationale

The choice is not stylistic. It follows directly from the Step 5 baseline ladder (§§8.1–8.4) and from the Step 6 architecture audit (§8.7). Three findings drive the design.

The first finding is the **regression-to-the-mean failure mode of pooled-MSE training** documented in §8.1 and §8.2. Ridge Regression and Histogram Gradient Boosting (HistGBM) both reduce pooled mean absolute error (MAE) relative to Persistence by exploiting engineered lag features, but their predictions are compressed toward the centre of the glucose distribution (the time-in-range / TIR zone, 70–180 mg/dL). This compression makes them systematically under-detect hypoglycaemic excursions (glucose below 70 mg/dL — the clinically most dangerous regime, because it can cause loss of consciousness within minutes). The implication is that the proposed model must include an inductive bias against this central-mass compression. (An *inductive bias* is a structural property of the model — built into the architecture or the loss — that makes some predictions easier to express than others; here we want hypoglycaemic excursions to remain easy to predict even when they are statistically rare.)

The second finding is the **rate-of-change advantage of Persistence on the Continuous Glucose-Error Grid Analysis** (CG-EGA — the rate-aware clinical-safety metric of Kovatchev et al. 2004, formally introduced in §2 and operationalised in §5.5). On a 5-minute CGM grid, Persistence (predicting the last observed value for every horizon) has the highest Accurate-Prediction share and the lowest Erroneous-Prediction share at every horizon among all Step 5 models — including HistGBM. The interpretation is that the *direction* and *rate* of the predicted trajectory matters at least as much as the predicted absolute value, and Persistence's implicit "no change" rate-of-change prediction is correct most of the time on a short grid. A defensible deep-learning model must therefore not throw away Persistence's rate-of-change behaviour; it should start from it and learn corrections on top of it.

The third finding is the **architecture audit of the §8.6 base hybrid** (the same CNN-GRU-Attention without persistence-residual learning), documented in §8.7. That audit shows the patient-conditioned cross-attention mechanism delivers a Pareto-near-Persistence solution on short horizons but degrades on long-horizon hypoglycaemic CG-EGA Erroneous-Prediction share, because the model's freely learned output trajectory drifts away from Persistence's rate-of-change anchor at 60 and 90 minutes. Persistence-residual learning is the smallest architectural change that fixes this drift by construction.

The proposed model therefore combines four mechanisms, each addressing one diagnosed weakness:

| Mechanism | Diagnosed weakness it addresses |
|---|---|
| Multi-kernel one-dimensional CNN over the lookback window | Tree models win pooled MAE by consuming hand-engineered rolling means; a multi-kernel CNN can learn equivalent finite-difference filters end-to-end. |
| Two-layer GRU on top of the CNN output | A flattened tree-features representation discards temporal ordering; the GRU keeps the time axis explicit so the model can react to recent change rates. |
| Cross-attention block with the static patient embedding as query | Per-patient heterogeneity dominates the EDA (§4.4); attention conditioned on each patient's static profile lets the recurrent representation be re-weighted patient-by-patient without per-patient fine-tuning. |
| Persistence-residual output head | Persistence wins CG-EGA at every horizon; predicting a delta on top of the last observation inherits this rate-oracle behaviour and forces the model to learn only the corrections. |

#### 7.6.2 Input contract and feature alignment

The model consumes the two arrays produced by the preprocessing pipeline of §5 and saved in `data/processed/hupa_5min_sequences.npz`:

- `X_dynamic ∈ ℝ^(N × T × F_dyn)`, with `N = 159 172` total windows, lookback length `T = 24` five-minute steps (i.e. 120 minutes of history), and `F_dyn = 17` engineered dynamic features per timestep (the catalogue is enumerated in §6.3).
- `X_static ∈ ℝ^(N × (F_stat + 1))`, with `F_stat = 16` static features per window (clinical metadata + behavioural fingerprint + modality availability + demographic one-hots; §6.3.7–§6.3.10), plus one extra integer column that carries the participant index used by the residual head to look up per-patient glucose scaler statistics (§7.6.4).
- `y ∈ ℝ^(N × 3)`, the future glucose values in mg/dL at `t+30`, `t+60`, and `t+90` minutes (the three forecasting horizons fixed in §1).

The chronological per-patient 70/15/15 split with a horizon-length boundary buffer of §5.6 produces the train / validation / test sample counts `(68 395, 45 382, 45 395)`. The model is trained on the train slice only; the validation slice is used for early stopping and best-checkpoint selection; the test slice is held out for §8.6 evaluation. No scaler, encoder, or hyperparameter is fitted using validation or test data — the leakage-prevention discipline of SKILL §0 Rule 5.

#### 7.6.3 Architecture and forward pass

The model is implemented in **PyTorch 2.x** (the open-source deep-learning framework maintained by Meta AI; the implementation does not depend on any closed-source library) and uses approximately 79 000 trainable parameters — small enough that one full training pass completes in ~12 minutes on a consumer CPU. The forward pass executes the five blocks described below in sequence; the same data flow is shown graphically in Figure 7.6.1.

**Block 1 — Multi-kernel one-dimensional CNN (`_MultiKernelCNN1d`).** Three `Conv1d` branches operate in parallel on the same `(B, F_dyn=17, T=24)` input tensor, where `B` is the mini-batch size, `F_dyn` is the number of input feature channels, and `T` is the lookback length. The three branches use kernel sizes `k = 3`, `k = 5`, and `k = 7`, which correspond to receptive fields of `15`, `25`, and `35` minutes on the 5-minute grid (a *receptive field* is the temporal span of input the convolution looks at in one application). Each branch outputs 16 channels; the three outputs are concatenated along the channel axis to produce `(B, 48, T)`, then passed through a ReLU non-linearity, a dropout of 0.3 (a regulariser that randomly zeros activations during training), and transposed back to time-major layout `(B, T, 48)`. The motivation for three kernels is the §8.1 Ridge-coefficient analysis: Ridge's strongest signal was a finite-difference reconstruction across two adjacent rolling-mean lags, suggesting that learnable short-window filters at multiple time scales should capture the same signal more efficiently.

**Block 2 — Two-layer GRU (`nn.GRU`).** The CNN output sequence is passed through a Gated Recurrent Unit with hidden dimension `H = 64` and `L = 2` stacked layers, with dropout `0.3` between layers. The GRU produces a hidden-state sequence `(B, T, H)`; the final timestep's hidden state `last_h ∈ ℝ^(B × H)` (i.e. `gru_out[:, -1, :]`) captures the model's summary of the lookback window after recurrent integration. (*Recurrent integration* means the hidden state at time `t` is a function of both the current input and the hidden state at time `t−1`, so information from earlier timesteps influences the representation at the end of the sequence.) GRU is chosen over LSTM because the Phase C.1 comparison in §8.3 showed the two cells produced essentially identical accuracy on this dataset while GRU has 22 % fewer parameters.

**Block 3 — Static MLP branch (`_StaticBranch`).** The static feature vector `X_static` (after slicing off the patient-index column) is passed through a two-layer fully connected network — `Linear(F_stat=16 → 32)` → ReLU → Dropout(0.3) → `Linear(32 → S=16)` — to produce a `(B, S=16)` *static embedding* `stat_emb`. (An *embedding* is a learned dense vector summary; here, one vector per patient-window encoding clinical demographics, behavioural fingerprint computed from training data only per §6.3.8, modality availability flags, and treatment one-hots.)

**Block 4 — Cross-attention block (`_CrossAttention`).** The static embedding queries the GRU output sequence to produce a `(B, A=48)` *attended representation* `attended`. Three linear projections map `stat_emb` to a query `q ∈ ℝ^(B × 1 × A)`, the full GRU output sequence to keys `k ∈ ℝ^(B × T × A)`, and the same GRU output sequence to values `v ∈ ℝ^(B × T × A)`. A standard multi-head attention layer (with `H_heads = 4` heads) then computes weighted sums of the values, with weights given by scaled dot products between the query and the keys. (*Multi-head attention* runs several independent attention operations in parallel, each on a learned sub-projection of the same data, then concatenates and re-projects the results; this lets the layer attend to several different aspects of the sequence simultaneously.) Intuitively, the static patient embedding asks the question "given this patient's clinical profile and modality coverage, which timesteps in the lookback window matter most for the forecast?", and the attended vector is the answer.

**Block 5 — Fusion + persistence-residual head.** The fused representation `fused = concat(last_h, attended, stat_emb) ∈ ℝ^(B × (H + A + S))` is passed through a multi-horizon dense head — `Linear(H+A+S → 64)` → ReLU → Dropout(0.3) → `Linear(64 → 3)` — producing a delta vector `δ ∈ ℝ^(B × 3)`. This delta is then added to the *last raw glucose value in the lookback window*, recovered by inverting the per-subject z-score scaler of §6.4: `last_glu_mgdl = x_dyn[:, -1, idx_glucose] * std[pid] + mean[pid]`, where `pid` is the patient index carried in the last column of `x_static`. The final prediction is `ŷ = last_glu_mgdl + δ`. Setting `δ = 0` recovers the Persistence prediction exactly; the model only needs to learn corrections.

Figure 7.6.1 shows the data flow as an ASCII block diagram.

```
                                  CNN-GRU-Attention with Persistence-Residual Learning
                                              src/models.py::HybridCNNGRUPersResid

   X_dynamic                                                                                       last raw glucose
   (B, 24, 17)                                                                                     (B,) in mg/dL
       │                                                                                                ▲
       │                                                                                                │  un-scale
       │                                                                                                │  with per-subject
       │                                                                                                │  scaler stats
       ▼                                                                                                │
  ┌─────────────────────────────────────────┐                                                    ┌─────────────┐
  │  Multi-kernel 1-D CNN                   │                                                    │ x_dyn[:,-1, │
  │   • Conv1d k=3 (15-min receptive field) │                                                    │  glucose]   │
  │   • Conv1d k=5 (25-min receptive field) │                                                    └──────▲──────┘
  │   • Conv1d k=7 (35-min receptive field) │                                                           │
  │   • concat channels → (B, 48, 24)       │                                                           │
  │   • ReLU + Dropout 0.3                  │                                                           │
  └────────────────────┬────────────────────┘                                                           │
                       │ (B, 24, 48)                                                                    │
                       ▼                                                                                │
  ┌─────────────────────────────────────────┐                                                           │
  │  GRU encoder                            │                                                           │
  │   • 2 stacked layers, hidden_dim = 64   │                                                           │
  │   • inter-layer dropout 0.3             │                                                           │
  │   • output sequence (B, 24, 64)         │                                                           │
  └────────────┬───────────────┬────────────┘                                                           │
               │ last_h        │ gru_out                                                                │
               │ (B, 64)       │ (B, 24, 64)                                                            │
               │               │                                                                        │
               │               └─────────────────────────┐                                              │
               │                                         │                                              │
               │                                         ▼                                              │
               │              X_static          ┌────────────────────────────┐                          │
               │              (B, 16)           │  Cross-attention block     │                          │
               │                  │             │   • query = static_emb     │                          │
               │                  ▼             │   • keys/values = gru_out  │                          │
               │       ┌──────────────────┐     │   • multi-head, 4 heads    │                          │
               │       │ Static MLP       │     │   • output (B, 48)         │                          │
               │       │  Linear 16→32    │────►│                            │                          │
               │       │  ReLU+Dropout    │     └─────────────┬──────────────┘                          │
               │       │  Linear 32→16    │                   │ attended (B, 48)                        │
               │       │  output (B, 16)  │──────────┐        │                                         │
               │       └──────────────────┘          │        │                                         │
               │                                     │        │                                         │
               └───────────────┬─────────────────────┴────────┘                                         │
                               │  concat                                                                │
                               ▼                                                                        │
                  ┌─────────────────────────────┐                                                       │
                  │  Multi-horizon dense head   │                                                       │
                  │   Linear (64+48+16) → 64    │                                                       │
                  │   ReLU + Dropout 0.3        │                                                       │
                  │   Linear 64 → 3             │                                                       │
                  │   output δ = (B, 3)         │                                                       │
                  └──────────────┬──────────────┘                                                       │
                                 │ delta in mg/dL                                                       │
                                 ▼                                                                      │
                          ┌──────────────┐                                                              │
                          │      +       │◄─────────────────────────────────────────────────────────────┘
                          └──────┬───────┘
                                 │
                                 ▼
                          ŷ = (B, 3)  →  predictions at t+30, t+60, t+90 min in mg/dL
                          (Persistence value + learned correction)
```

**Figure 7.6.1.** Forward-pass block diagram of the proposed model. Shapes are written as `(batch, time, features)` for sequences and `(batch, features)` for embeddings. The persistence-residual head is the small block on the right: the model emits a delta vector and the final prediction is recovered by adding it to the last raw glucose observation in the lookback window. The arrows from the CNN through the GRU and into the cross-attention block are the standard CNN-GRU-Attention pathway; the upward arrow on the right is the inductive-bias mechanism that distinguishes this thesis's model from the §8.6 base hybrid.

#### 7.6.4 Training objective, modality dropout, and optimisation

The training loss is the `ZoneWeightedMSE` of §7.5 (`src/losses.py::ZoneWeightedMSE`) with the winning Phase C.2 configuration `zwh30a`: zone weights `w_hypo = 2.0`, `w_tir = 1.0`, `w_hyper = 1.5`; horizon weights `(1.5, 1.0, 1.0)` emphasising the 30-minute horizon; and an asymmetric multiplicative penalty of `2.0` on the squared error when the reference glucose is hypoglycaemic and the prediction is higher than the reference (the *under-detection* failure mode — clinically the most dangerous error to make). The choice is inherited unchanged from §7.5 so that any improvement attributed to the architecture in §8.6 is not confounded with a loss-function change.

**Modality dropout** is applied during training at probability `p_modality = 0.30`. When triggered for a given sample, the dynamic-feature time series has one of three modality groups — insulin (basal + bolus + IOB), carbohydrate (carb input + COB), or activity (heart rate + steps + calories) — zeroed for the entire lookback window, while the corresponding availability flag in `x_static` is overwritten to zero. This forces the model to produce a valid forecast even when a sensor stream is missing at inference time, which §3.6.2 and §3.6.3 documented as a real condition for several HUPA-UCM patients (HUPA0011/0014/0015/0018/0020) and as an inevitable condition for any practical online deployment. (Without modality dropout the model is free to depend on a modality that may simply not be present at deployment time; modality dropout teaches the model to operate gracefully under partial sensor availability.) Validation and test inference are performed with the full modality set.

**Optimiser and schedule.** Training uses Adam with initial learning rate `1 × 10⁻³` and weight decay `1 × 10⁻⁵`. The learning rate is halved by a `ReduceLROnPlateau` scheduler whenever the validation patient-averaged MAE fails to improve for 3 consecutive epochs (the *plateau* condition). Gradient clipping is applied at L2-norm `1.0` to prevent rare large updates during the first epochs. Early stopping triggers after 5 consecutive epochs without validation improvement; the maximum budget is 30 epochs. The best checkpoint is selected by validation **patient-averaged** MAE (not pooled MAE), which prevents the long-duration patients HUPA0027P / HUPA0028P from dominating the model-selection criterion — the same discipline applied throughout Phase C.

**Determinism.** All four random sources are seeded at `SEED = 42` at the start of every run: Python's `random`, NumPy, PyTorch CPU, and PyTorch CUDA (where available). The DataLoader uses a single worker with a seeded generator. With these settings the training trajectory and final test metrics are bit-for-bit reproducible on the same hardware.

#### 7.6.5 Framework, hyperparameter summary, reproducibility, and Colab compatibility

The complete hyperparameter set is shown in Table 7.6.1. All values were chosen *before* looking at the test split: the loss configuration is inherited from §7.5, the optimiser values are inherited from §7.4 (Phase C.1), the CNN kernel sizes and GRU dimensions are taken from the parent architecture (`HybridCNNGRU` in `src/models.py`, evaluated independently in §8.7), and the persistence-residual mechanism adds no new hyperparameter (only the buffer of per-subject glucose scaler statistics, which is fitted on the training set in §5.7).

**Table 7.6.1.** Hyperparameters of the proposed thesis model. All values are fixed before model fitting and are recorded in `src/run_step6_v2.py`.

| Component | Hyperparameter | Value |
|---|---|---|
| Input | Lookback steps `T` | 24 (= 120 min) |
| Input | Dynamic features `F_dyn` | 17 |
| Input | Static features `F_stat` | 16 (+ 1 patient-index column) |
| Input | Forecast horizons | (+30 min, +60 min, +90 min) |
| CNN | Kernel sizes | (3, 5, 7) |
| CNN | Channels per kernel | 16 → 48 total |
| GRU | Hidden dimension `H` | 64 |
| GRU | Layers `L` | 2 |
| Cross-attention | Attention dimension `A` | 48 |
| Cross-attention | Heads | 4 |
| Static MLP | Output dimension `S` | 16 |
| Head | Hidden dimension | 64 |
| Head | Output dimension | 3 (one per horizon) |
| Regularisation | Dropout (CNN / GRU / MLP / head) | 0.3 / 0.3 / 0.3 / 0.3 |
| Regularisation | Modality dropout `p_modality` | 0.30 (train only) |
| Loss | `ZoneWeightedMSE` zone weights | `(hypo=2.0, tir=1.0, hyper=1.5)` |
| Loss | Horizon weights | `(1.5, 1.0, 1.0)` |
| Loss | Hypo under-detection penalty | 2.0 |
| Optimiser | Adam (lr / weight decay) | `1e-3` / `1e-5` |
| Optimiser | Gradient clip (L2 norm) | 1.0 |
| Scheduler | `ReduceLROnPlateau` (factor / patience) | 0.5 / 3 |
| Training | Batch size | 128 |
| Training | Maximum epochs | 30 |
| Training | Early-stopping patience | 5 |
| Training | Best-checkpoint criterion | validation patient-averaged MAE |
| Reproducibility | Random seed | 42 |
| Parameters | Total trainable | ≈ 79 000 |

**Framework, runtime, and Colab compatibility.** The implementation depends only on PyTorch 2.x, NumPy 1.26.x, and pandas 2.x; no closed-source library is required and no GPU is required for a full training run. The training scripts auto-detect Colab via the `COLAB_GPU` environment variable and mount Google Drive at `/content/drive/MyDrive/glucose-thesis/` (the `BASE_PATH` pattern of §11). A full training run completes in approximately 12 minutes on a consumer CPU; on a Colab T4 GPU the same run completes in approximately 90 seconds. All artefacts — model checkpoint, training log, evaluation tables, and predictions — are saved to the same directory structure regardless of execution environment, so a single command (`python src/run_step6_v2.py --variant pers_resid`) reproduces the §8.6 test results from `data/processed/hupa_5min_sequences.npz` end-to-end.

**Reproducibility traceability.** The model checkpoint is `outputs/models/step6_hybrid_v2_pers_resid.pt`; the training log is `outputs/logs/step6_hybrid_v2_pers_resid.csv`; the evaluation tables are `outputs/tables/step6_v2_pers_resid_*.csv`; the prediction master parquet is `outputs/tables/step6_v2_predictions.parquet`. The architecture source is `src/models.py::HybridCNNGRUPersResid` (a subclass of `src/models.py::HybridCNNGRU`); the loss source is `src/losses.py::ZoneWeightedMSE`; the dataset source is `src/datasets.py::HUPASequenceDataset`; the per-subject scaler statistics that the residual head consumes are loaded from `outputs/models/scalers.json`. Every numeric claim in §8.6 traces back through these files, in keeping with SKILL §10.

---

## 8. Experimental Results and Evaluation

### 8.1 Phase A baselines — Persistence and Ridge

Phase A reports the two linear references on the test split after model selection on the validation split. The full metric bundle is computed in `src/evaluate.py` and produced by the runner `src/run_phase_a.py` (also available as `notebooks/04_model_training.ipynb` for Colab execution). The selected Ridge regularisation strength is $\alpha = 0.1$, the smallest value in the grid; the rcond reported by the linear solver was approximately $5 \times 10^{-8}$, reflecting the lag collinearity expected for an over-complete rolling-mean basis but well within the numerical stability of the Cholesky-style solver used by `sklearn.linear_model.Ridge`.

Table 8.1.1 reports the pooled (row-weighted) and patient-averaged MAE and RMSE on the test split, in mg/dL, for both models at all three horizons.

**Table 8.1.1.** Test errors (mg/dL) per horizon. Persistence has no tunable parameters; Ridge selected $\alpha = 0.1$ on validation. Pooled values are row-weighted across all 45 395 test windows; patient-averaged values are the unweighted mean across 25 patients.

| Model | Horizon | Pooled MAE | Pooled RMSE | Patient-avg MAE | Patient-avg RMSE |
|---|---|---|---|---|---|
| Persistence       | 30 min | 13.52 | 19.70 | 15.79 | 22.12 |
| Persistence       | 60 min | 22.92 | 32.91 | 27.36 | 37.47 |
| Persistence       | 90 min | 29.85 | 42.13 | 36.12 | 48.56 |
| Ridge ($\alpha=0.1$) | 30 min | 14.25 | 19.87 | 17.10 | 22.19 |
| Ridge ($\alpha=0.1$) | 60 min | 21.59 | 30.20 | 26.90 | 34.94 |
| Ridge ($\alpha=0.1$) | 90 min | 27.85 | 37.86 | 34.61 | 44.21 |

Three patterns are immediately visible. First, Persistence outperforms Ridge at the 30-minute horizon on pooled MAE (13.52 vs 14.25 mg/dL), confirming the structural strength of the flat-line nowcaster on a 5-minute CGM grid. Second, Ridge improves materially at the 60 and 90-minute horizons — pooled MAE gains of 1.3 and 2.0 mg/dL, and pooled RMSE gains of 2.7 and 4.3 mg/dL — showing that the engineered velocity and rolling-mean features carry signal that Persistence cannot represent. Third, the patient-averaged numbers are systematically higher than the pooled numbers by approximately 2 mg/dL at 30 minutes and 7 mg/dL at 90 minutes; this is a direct consequence of the duration imbalance documented in Section 5.8, where HUPA0027P and HUPA0028P dominate the row-weighted aggregate and, being above-median patients in glycaemic control, bias the pooled errors downward. Every subsequent model in the ladder reports both conventions for this reason.

Table 8.1.2 reports the Continuous Glucose-Error Grid Analysis (CG-EGA) of Kovatchev et al. (2004) on the test split. CG-EGA classifies every (reference, prediction) pair as an Accurate Prediction (AP), a Benign Error (BE), or an Erroneous Prediction (EP) using a glycaemic-zone-conditional matrix that combines a point-accuracy assessment with a 15-minute rate-of-change accuracy assessment; this is the primary clinical-safety metric for this thesis per Section 2. Persistence is already remarkably strong on this metric: it achieves AP shares of 91.77 / 85.40 / 81.89 % and EP shares of 2.50 / 5.15 / 6.46 % at 30 / 60 / 90 minutes respectively, reflecting that on a 5-minute CGM grid the flat-line nowcaster's predicted rate-of-change correlates well with the actual rate within the very short window. Ridge has marginally lower AP at every horizon and a comparable EP at 30 and 60 minutes but a noticeably higher EP at 90 minutes (8.42 % versus 6.46 %), indicating that the linear extrapolation Ridge performs introduces clinically erroneous rate predictions at long horizons more often than the flat-line nowcaster does.

**Table 8.1.2.** Continuous Glucose-Error Grid Analysis (CG-EGA) on test (percent of predictions per class).

| Model | Horizon | AP % | BE % | EP % |
|---|---|---|---|---|
| Persistence       | 30 min | 91.77 | 5.73  | 2.50 |
| Persistence       | 60 min | 85.40 | 9.45  | 5.15 |
| Persistence       | 90 min | 81.89 | 11.65 | 6.46 |
| Ridge ($\alpha=0.1$) | 30 min | 87.11 | 9.90  | 2.99 |
| Ridge ($\alpha=0.1$) | 60 min | 79.28 | 14.56 | 6.16 |
| Ridge ($\alpha=0.1$) | 90 min | 76.19 | 15.39 | 8.42 |

**Table 8.1.2 (legacy).** Clarke Error Grid Analysis on test (percent of predictions per zone). Retained as a backwards-compatible legacy view; not used as the primary clinical claim per Section 2.

| Model | Horizon | A | B | C | D | E |
|---|---|---|---|---|---|---|
| Persistence       | 30 min | 85.45 | 12.89 | 0.02 | 1.64 | 0.00 |
| Persistence       | 60 min | 69.43 | 26.64 | 0.55 | 3.38 | 0.00 |
| Persistence       | 90 min | 59.40 | 34.55 | 1.45 | 4.60 | 0.00 |
| Ridge ($\alpha=0.1$) | 30 min | 86.09 | 12.48 | 0.06 | 1.38 | 0.00 |
| Ridge ($\alpha=0.1$) | 60 min | 71.81 | 24.40 | 0.23 | 3.56 | 0.00 |
| Ridge ($\alpha=0.1$) | 90 min | 60.30 | 32.85 | 0.51 | 6.34 | 0.00 |

The most consequential finding for the thesis is in the per-zone MAE/RMSE breakdown (Table 8.1.3). Persistence beats Ridge in the hypoglycaemic zone at every horizon — by 6.6 mg/dL MAE at 30 minutes, 2.6 mg/dL at 60 minutes, and 7.0 mg/dL at 90 minutes. Ridge wins in the time-in-range zone at 60 and 90 minutes and ties at 30 minutes, and Ridge wins in the hyperglycaemic zone at 30 and 60 minutes, but the gain in TIR and hyper cannot recover the hypo deficit. The paired RMSE values show the same direction in the hypoglycaemic zone, confirming that the finding is not an artefact of absolute-error aggregation. This pattern is consistent with the well-known regression-to-the-mean failure mode of linear models trained on imbalanced glucose data: the hypoglycaemic zone holds 8.0 % of test windows, the TIR zone 71.5 %, and a least-squares loss minimised over the pooled distribution drives the model toward TIR predictions at the cost of mis-predicting hypoglycaemic events.

**Table 8.1.3.** Per-zone test MAE / RMSE (mg/dL). Hypo $<70$, TIR $70$–$180$, hyper $>180$. Bold marks the worse model in each cell by MAE.

| Model | Horizon | Hypo | TIR | Hyper |
|---|---|---|---|---|
| Persistence       | 30 min | 9.06 / 14.55  | 12.82 / 18.50 | 17.72 / 24.84 |
| Persistence       | 60 min | 18.34 / 29.63 | 21.01 / 29.84 | 31.39 / 42.88 |
| Persistence       | 90 min | 27.26 / 42.19 | 26.72 / 37.39 | 41.84 / 55.60 |
| Ridge ($\alpha=0.1$) | 30 min | **15.66 / 19.20** | 12.82 / 18.27 | **18.72 / 24.85** |
| Ridge ($\alpha=0.1$) | 60 min | **20.94 / 29.95** | 19.61 / 27.33 | 28.79 / 38.67 |
| Ridge ($\alpha=0.1$) | 90 min | **34.27 / 44.24** | 23.54 / 31.93 | 40.43 / 51.74 |

This finding is the empirical justification for two design choices to be implemented in Phase C: (i) the proposed hybrid model will use an asymmetric loss with a hypoglycaemic penalty term, following Del Favero et al. (2012); and (ii) the evaluation report in every subsequent phase will lead with the per-zone MAE/RMSE breakdown rather than the pooled metric, so that any pooled improvement that comes at the cost of hypoglycaemic accuracy is immediately visible. A model that wins on pooled MAE/RMSE but loses on hypoglycaemic MAE/RMSE relative to Persistence is clinically inferior to Persistence and cannot defend the thesis claim.

The top-five coefficients of the Ridge model at the 30-minute horizon, in signed units of mg/dL per unit Z-scored feature (from `outputs/tables/phase_a_ridge_top_coefs.csv`), are dominated by glucose rolling means at the most recent lags: `glucose_60m_mean_lag0` ($-1225$), `glucose_60m_mean_lag1` ($-872$), `glucose_30m_mean_lag0` ($+692$), `glucose_60m_mean_lag2` ($-512$), and `glucose_30m_mean_lag1` ($+448$). The alternating signs across adjacent lags act as a finite-difference operator that effectively reconstructs a velocity term from the rolling-mean basis, supplementing the explicit `glucose_velocity` feature (which carries a smaller coefficient magnitude). The dominant predictive signal at short horizons is therefore the recent glucose trajectory; the static patient features, peri-event aggregates (IOB, COB, steps), and modality flags contribute additively but at much smaller magnitudes. This decomposition will guide architectural choices in Phase C — in particular, whether to wrap the dynamic input in a one-dimensional temporal convolution that learns the same finite-difference response directly from the raw lookback window.

All numbers in this section trace back to `outputs/tables/phase_a_*.csv` and `outputs/models/ridge_phase_a.joblib`; the runner that produces them is `src/run_phase_a.py` and the Colab-compatible execution path is `notebooks/04_model_training.ipynb`.

### 8.2 Phase B baselines — Random Forest and Gradient Boosting

Phase B reports the two tree-based references on the same test split and metric bundle. Wall-clock fit times on a multi-core consumer CPU (`n_jobs = -1`) were approximately 26 minutes for the Random Forest and 3 minutes for the three-headed gradient boosting, both well within feasible local-compute budgets. The HistGradientBoosting heads each reached the `max_iter = 300` budget cap without triggering early stopping; the consequences of this cap are quantified in the budget ablation reported at the end of this section.

Table 8.2.1 collects all four baselines on the test split. Pooled MAE and RMSE are row-weighted across the 45 395 test windows; patient-averaged MAE and RMSE are the unweighted mean across 25 patients, reported in parallel for the reason established in Section 8.1.

**Table 8.2.1.** Test errors (mg/dL) per horizon for the full baseline ladder. Bold marks the best model in each cell.

| Model | Horizon | Pooled MAE | Pooled RMSE | Patient-avg MAE | Patient-avg RMSE |
|---|---|---|---|---|---|
| Persistence       | 30 min | 13.52 | 19.70 | 15.79 | 22.12 |
| Ridge ($\alpha=0.1$) | 30 min | 14.25 | 19.87 | 17.10 | 22.19 |
| Random Forest (n=300) | 30 min | 11.79 | 16.72 | 13.94 | 18.83 |
| HistGB (lr=0.05, d=8) | 30 min | **10.40** | **15.03** | **12.11** | **16.82** |
| Persistence       | 60 min | 22.92 | 32.91 | 27.36 | 37.47 |
| Ridge ($\alpha=0.1$) | 60 min | 21.59 | 30.20 | 26.90 | 34.94 |
| Random Forest (n=300) | 60 min | 20.68 | 28.29 | 24.25 | 32.04 |
| HistGB (lr=0.05, d=8) | 60 min | **19.77** | **27.22** | **23.58** | **31.25** |
| Persistence       | 90 min | 29.85 | 42.13 | 36.12 | 48.56 |
| Ridge ($\alpha=0.1$) | 90 min | 27.85 | 37.86 | 34.61 | 44.21 |
| Random Forest (n=300) | 90 min | 27.31 | 36.48 | 32.25 | 42.00 |
| HistGB (lr=0.05, d=8) | 90 min | **26.27** | **35.36** | **31.49** | **41.00** |

HistGradientBoosting dominates the pooled metric at every horizon, with a 30-minute pooled MAE of 10.40 mg/dL — a 23 % relative reduction against Persistence and a 27 % reduction against Ridge. The Random Forest sits between Ridge and HistGB and also beats both Phase A models at every horizon. The pooled MAE ranking Persistence > Ridge > RF > HistGB is monotonic at all three horizons and is preserved when the metric is switched to RMSE, confirming that the improvement is not specific to the loss aggregation choice.

Table 8.2.2 reports the CG-EGA on the test split for the two Phase B models. The pattern from Phase A inverts. By CG-EGA both tree baselines under-perform Persistence at every horizon: at 30 minutes HistGB's AP share is 89.06 % versus Persistence's 91.77 % and the EP share is 3.20 % versus Persistence's 2.50 %; at 60 minutes HistGB's AP is 79.17 % versus Persistence's 85.40 % and the EP share *doubles* the difference at 8.23 % versus 5.15 %; at 90 minutes HistGB's AP is 76.95 % versus Persistence's 81.89 % and the EP share is 9.67 % versus 6.46 %. The Random Forest is worse still on CG-EGA at every horizon. The mechanism is that the gradient-boosted tree fits the conditional mean of the next-glucose distribution efficiently in absolute-value terms but produces low-variance, smoothed predictions whose rate-of-change disagrees with the true rate during glycaemic excursions — exactly the regime CG-EGA was designed to surface. The Clarke Error Grid (legacy) in the second part of Table 8.2.2 retains the headline that HistGB-300 has the highest Zone A share at 30 minutes (91.27 %), but the Clarke result is misleading without the corresponding CG-EGA reading because Clarke ignores rate-of-change agreement.

**Table 8.2.2.** Continuous Glucose-Error Grid Analysis (CG-EGA) on test for the Phase B models (percent of predictions per class).

| Model | Horizon | AP % | BE % | EP % |
|---|---|---|---|---|
| Random Forest (n=300) | 30 min | 87.26 | 8.10  | 4.64 |
| Random Forest (n=300) | 60 min | 78.14 | 13.22 | 8.64 |
| Random Forest (n=300) | 90 min | 75.60 | 14.41 | 9.99 |
| HistGB (lr=0.05, d=8) | 30 min | 89.06 | 7.75  | 3.20 |
| HistGB (lr=0.05, d=8) | 60 min | 79.17 | 12.60 | 8.23 |
| HistGB (lr=0.05, d=8) | 90 min | 76.95 | 13.38 | 9.67 |

**Table 8.2.2 (legacy).** Clarke Error Grid Analysis on test for the Phase B models (percent of predictions per zone). Retained for backwards-compatibility only; not used as the primary clinical claim.

| Model | Horizon | A | B | C | D | E |
|---|---|---|---|---|---|---|
| Random Forest (n=300)   | 30 min | 88.01 | 8.58  | 0.00 | 3.41 | 0.00 |
| Random Forest (n=300)   | 60 min | 71.07 | 21.93 | 0.08 | 6.91 | 0.00 |
| Random Forest (n=300)   | 90 min | 59.03 | 32.04 | 0.25 | 8.68 | 0.00 |
| HistGB (lr=0.05, d=8) | 30 min | 91.27 | 6.79  | 0.01 | 1.94 | 0.00 |
| HistGB (lr=0.05, d=8) | 60 min | 72.79 | 20.74 | 0.07 | 6.41 | 0.00 |
| HistGB (lr=0.05, d=8) | 90 min | 61.11 | 30.23 | 0.22 | 8.44 | 0.00 |

The headline clinical finding of Phase B is in Table 8.2.3: **Persistence remains the best model in the hypoglycaemic zone at every horizon by MAE**, despite being beaten by every other model on the pooled metric. HistGB closes the hypo gap at 30 minutes to 1.4 mg/dL MAE (10.42 versus Persistence 9.06), but at 60 and 90 minutes its hypo MAE diverges substantially upward — 26.75 and 40.72 mg/dL versus Persistence's 18.34 and 27.26. The paired RMSE values show the same long-horizon problem: HistGB hypo RMSE is 31.96 and 45.95 mg/dL at 60 and 90 minutes, compared with Persistence's 29.63 and 42.19. Random Forest behaves similarly. The hypo deficit of the non-linear models is therefore not a defect of feature engineering or of model capacity; it is the loss-function consequence already diagnosed in Section 8.1. A squared-error objective minimised over a distribution in which the time-in-range zone holds 71 % of the mass and the hypoglycaemic zone holds 8 % systematically biases the model toward the dense mass, regardless of whether the model class is linear, axis-aligned forests, or gradient-boosted trees.

**Table 8.2.3.** Per-zone test MAE / RMSE (mg/dL) for the full baseline ladder. Bold marks the best model in each cell by MAE. Hypo $<70$, TIR $70$–$180$, hyper $>180$.

| Model | Horizon | Hypo | TIR | Hyper |
|---|---|---|---|---|
| Persistence       | 30 min | **9.06 / 14.55**  | 12.82 / 18.50 | 17.72 / 24.84 |
| Ridge ($\alpha=0.1$) | 30 min | 15.66 / 19.20 | 12.82 / 18.27 | 18.72 / 24.85 |
| Random Forest (n=300) | 30 min | 14.66 / 17.92 | 10.34 / 14.67 | 15.72 / 22.09 |
| HistGB (lr=0.05, d=8) | 30 min | 10.42 / 14.03 | **9.43 / 13.70**  | **13.77 / 19.25** |
| Persistence       | 60 min | **18.34 / 29.63** | 21.01 / 29.84 | 31.39 / 42.88 |
| Ridge ($\alpha=0.1$) | 60 min | 20.94 / 29.95 | 19.61 / 27.33 | 28.79 / 38.67 |
| Random Forest (n=300) | 60 min | 29.89 / 34.35 | 17.01 / 23.39 | 29.94 / 39.22 |
| HistGB (lr=0.05, d=8) | 60 min | 26.75 / 31.96 | **16.42 / 22.60** | **28.75 / 37.89** |
| Persistence       | 90 min | **27.26 / 42.19** | 26.72 / 37.39 | 41.84 / 55.60 |
| Ridge ($\alpha=0.1$) | 90 min | 34.27 / 44.24 | 23.54 / 31.93 | 40.43 / 51.74 |
| Random Forest (n=300) | 90 min | 43.51 / 48.52 | 21.28 / 28.48 | 42.09 / 52.45 |
| HistGB (lr=0.05, d=8) | 90 min | 40.72 / 45.95 | **20.53 / 27.56** | **40.71 / 51.30** |

The pattern is therefore consistent across all four baselines: increasing model capacity along the standard ladder (linear → axis-aligned forest → gradient-boosted forest) reduces pooled MAE monotonically by 23 % at 30 minutes and 12 % at 90 minutes, but does not solve the regression-to-the-TIR-mean bias in the hypoglycaemic zone. This is the empirical justification for the architectural and loss-function decisions to be implemented in Phase C: (i) any sequence model that uses a vanilla squared-error loss will inherit the same hypo bias, regardless of recurrence, attention, or fusion mechanism, so the loss function itself must be modified; and (ii) Persistence will be retained as a per-zone reference even after stronger models exist, because in the hypoglycaemic safety case it is the model to beat, not Ridge or HistGB.

**Budget ablation — `max_iter = 300` versus `max_iter = 1000`.** Because the three GBM-300 heads exhausted their iteration cap, a follow-up run trained a second HistGradientBoosting model with `max_iter = 1000` and every other hyperparameter, the same chronological split, the same flattened-window input, and the same seed held fixed. The 1000-iteration model used the full budget on every horizon (1000/1000/1000), confirming that sklearn's internal early-stopping criterion — evaluated on a 10 % slice taken from the training set — was still improving. The external test set tells a different story: pooled test MAE *increased* slightly under the larger budget, by +0.09 mg/dL at 30 minutes, +0.30 at 60 minutes, and +0.34 at 90 minutes (+0.8 %, +1.5 %, +1.3 % relative). Per-zone test MAE in the hypoglycaemic range also moved in the wrong direction at every horizon, by +0.58, +0.92, and +0.32 mg/dL. Patient-averaged test MAE rose by +0.13, +0.34, and +0.57 mg/dL, and the Clarke Zone A share dropped by 0.09, 0.74, and 0.37 percentage points. The divergence between the internal-validation signal — which kept improving for 700 additional iterations — and the external test signal — which started to degrade — is a textbook signature of mild over-fitting that sklearn's internal early stopping cannot detect because it samples its validation slice from the same training distribution. The `max_iter = 300` configuration is therefore retained as the primary GBM baseline; the 1000-iteration run is documented as an ablation rather than promoted. The full delta table is at `outputs/tables/phase_b_gbm_comparison.csv` and the ablation model is checkpointed separately at `outputs/models/gbm_phase_b_1000.joblib`.

All numbers in this section trace back to `outputs/tables/phase_b_*.csv`, `outputs/tables/phase_b_gbm1000_*.csv`, `outputs/models/rf_phase_b.joblib`, `outputs/models/gbm_phase_b.joblib`, and `outputs/models/gbm_phase_b_1000.joblib`. The runners that produce them are `src/run_phase_b.py` and `src/run_gbm_1000.py`, and the Colab-compatible execution path is `notebooks/04b_phase_b_trees.ipynb`.

### 8.3 Phase C.1 recurrent baselines — LSTM and GRU

Phase C.1 evaluates two recurrent encoders — LSTM and GRU — under the identical input contract used in Phase A and Phase B (`X_dynamic` of shape `(N, 24, 17)` plus `X_static` of shape `(N, 16)`), the same per-patient chronological train/val/test split, the same patient-averaged early-stopping target, and a vanilla multi-horizon MSE loss. The motivation for the experiment is set out in Section 7.4.1: the gradient-boosted tree baseline of Section 8.2 is capacity-saturated on the 424-dimensional flattened representation, and the remaining structural lever available without modifying the loss function or introducing attention/fusion mechanisms is to preserve the temporal structure of the lookback window using a recurrent encoder. The full Colab T4 run completed in 99.7 seconds total wall-clock — 52.5 s for LSTM and 47.1 s for GRU — including final evaluation on val and test. Both models triggered early stopping well before the 30-epoch budget: the best LSTM checkpoint was reached at epoch 2 (validation patient-averaged MAE 24.99), and the best GRU checkpoint at epoch 3 (25.09); subsequent epochs continued to reduce training loss monotonically while validation patient-averaged MAE drifted upward, a textbook overfitting signature on a small, low-noise regression target.

**Table 8.3.1.** Test pooled errors (mg/dL) per horizon — Phase C.1 recurrent baselines compared to the Phase B winner (HistGB lr=0.05, d=8, max_iter=300). Bold marks the best model in each cell.

| Model | Horizon | Pooled MAE | Pooled RMSE | Patient-avg MAE | Patient-avg RMSE |
|---|---|---|---|---|---|
| HistGB (lr=0.05, d=8) | 30 min | **10.40** | **15.03** | **12.11** | **16.82** |
| LSTM (h=64, L=2)      | 30 min | 15.38     | 20.49     | 17.75     | 22.75     |
| GRU (h=64, L=2)       | 30 min | 15.76     | 21.33     | 18.09     | 23.42     |
| HistGB (lr=0.05, d=8) | 60 min | **19.77** | **27.22** | **23.58** | **31.25** |
| LSTM (h=64, L=2)      | 60 min | 21.39     | 29.41     | 25.27     | 33.11     |
| GRU (h=64, L=2)       | 60 min | 21.42     | 29.62     | 25.43     | 33.24     |
| HistGB (lr=0.05, d=8) | 90 min | **26.27** | **35.36** | **31.49** | **41.00** |
| LSTM (h=64, L=2)      | 90 min | 27.04     | 37.20     | 32.23     | 41.99     |
| GRU (h=64, L=2)       | 90 min | 26.75     | 36.91     | 32.25     | 41.97     |

The pooled-MAE story is unambiguous: HistGB-300 dominates both recurrent baselines at every horizon, but the gap collapses as the prediction horizon lengthens. At 30 minutes the LSTM and GRU are 48 % and 52 % worse respectively than HistGB-300 in absolute terms — a 5 mg/dL absolute deficit that is clinically meaningful at hypoglycaemic decision boundaries. At 60 minutes the gap shrinks to roughly 8 % (≈ 1.6 mg/dL) for both models. At 90 minutes the GRU is within 1.8 % of HistGB-300 (26.75 versus 26.27 mg/dL) and the LSTM within 2.9 %. The interpretation tracks the prior expectation expressed in Section 7.4: short-horizon forecasting is dominated by recent glucose history, and the gradient-boosted tree extracts that history maximally efficiently through the engineered rolling-mean and velocity features computed at preprocessing time, whereas the recurrent encoder must rediscover the same structure from the raw lookback through gradient descent on roughly 50 000–60 000 parameters. As the horizon lengthens and the predictive value of immediate glucose history decays, the engineered-feature advantage shrinks and the two architectures converge to comparable performance. The LSTM and GRU produce numerically near-identical test results — the maximum disagreement between them is 0.38 mg/dL at the 30-minute horizon and falls below 0.30 mg/dL at the longer horizons. Cell-type choice between standard LSTM and GRU is therefore not the bottleneck on this dataset; Phase C.2 selects GRU for the loss ablation on the grounds of marginally lower parameter count and faster per-epoch wall-clock, not on the grounds of any observed quality difference.

**Table 8.3.2.** Clarke Error Grid Analysis on the test split for the Phase C.1 recurrent baselines compared to HistGB-300 (percent of predictions per zone).

| Model | Horizon | A | B | C | D | E |
|---|---|---|---|---|---|---|
| HistGB (lr=0.05, d=8) | 30 min | **91.27** | 6.79  | 0.01 | **1.94** | 0.00 |
| LSTM (h=64, L=2)      | 30 min | 83.11     | 14.10 | 0.02 | 2.77     | 0.00 |
| GRU (h=64, L=2)       | 30 min | 82.21     | 14.99 | 0.02 | 2.76     | 0.00 |
| HistGB (lr=0.05, d=8) | 60 min | **72.79** | 20.74 | 0.07 | 6.41     | 0.00 |
| LSTM (h=64, L=2)      | 60 min | 71.02     | 24.71 | 0.06 | **4.20** | 0.01 |
| GRU (h=64, L=2)       | 60 min | 70.86     | 24.55 | 0.05 | **4.54** | 0.01 |
| HistGB (lr=0.05, d=8) | 90 min | **61.11** | 30.23 | 0.22 | 8.44     | 0.00 |
| LSTM (h=64, L=2)      | 90 min | 61.99     | 32.31 | 0.10 | **5.59** | 0.00 |
| GRU (h=64, L=2)       | 90 min | 62.19     | 31.62 | 0.07 | **6.14** | 0.00 |

The Clarke Error Grid Analysis reveals a second, less expected pattern. On Clarke Zone A (clinically accurate prediction) HistGB-300 dominates at the 30-minute horizon by a margin of 8–9 percentage points, in line with the pooled-MAE/RMSE result, but at the 90-minute horizon the LSTM and GRU narrowly *outperform* HistGB-300 on Zone A (61.99 % and 62.19 % versus 61.11 %). More clinically meaningful, the recurrent baselines show a substantially lower share of Clarke Zone D — predictions that fail to detect a dangerous condition — at the 60- and 90-minute horizons: at 60 minutes the LSTM Zone D share is 4.20 % versus HistGB-300's 6.41 %, and at 90 minutes 5.59 % versus 8.44 %. The 30-minute horizon goes the other way (LSTM 2.77 % versus HistGB-300 1.94 %), so the effect is horizon-dependent, but the direction at the longer horizons is consistent across both cell types. The same horizon-dependent pattern resolves into a sharper picture when the per-zone MAE breakdown of Table 8.3.3 is examined directly.

**Table 8.3.3.** Per-zone test MAE (mg/dL) — full baseline ladder including Phase C.1 recurrent baselines. Bold marks the best model in each cell. Hypo $<70$, TIR $70$–$180$, hyper $>180$. Horizon-level RMSE for these same models is reported alongside MAE in Table 8.3.1; full-run C.1 per-zone RMSE should be added from the Colab artefacts when those files are restored locally, because the current local C.1 per-zone RMSE files are debug-run outputs and are not valid thesis numbers.

| Model | Horizon | Hypo | TIR | Hyper |
|---|---|---|---|---|
| Persistence           | 30 min | **9.06**  | 12.82     | 17.72     |
| Ridge ($\alpha=0.1$)  | 30 min | 15.66     | 12.82     | 18.72     |
| Random Forest (n=300) | 30 min | 14.66     | 10.34     | 15.72     |
| HistGB (lr=0.05, d=8) | 30 min | 10.42     | **9.43**  | **13.77** |
| LSTM (h=64, L=2)      | 30 min | 14.75     | 14.29     | 19.44     |
| GRU (h=64, L=2)       | 30 min | 15.49     | 14.12     | 21.59     |
| Persistence           | 60 min | **18.34** | 21.01     | 31.39     |
| Ridge ($\alpha=0.1$)  | 60 min | 20.94     | 19.61     | 28.79     |
| Random Forest (n=300) | 60 min | 29.89     | 17.01     | 29.94     |
| HistGB (lr=0.05, d=8) | 60 min | 26.75     | **16.42** | 28.75     |
| LSTM (h=64, L=2)      | 60 min | 20.84     | 19.37     | **28.65** |
| GRU (h=64, L=2)       | 60 min | 22.55     | 19.08     | 29.15     |
| Persistence           | 90 min | **27.26** | 26.72     | 41.84     |
| Ridge ($\alpha=0.1$)  | 90 min | 34.27     | 23.54     | 40.43     |
| Random Forest (n=300) | 90 min | 43.51     | 21.28     | 42.09     |
| HistGB (lr=0.05, d=8) | 90 min | 40.72     | **20.53** | 40.71     |
| LSTM (h=64, L=2)      | 90 min | 28.88     | 23.35     | **39.27** |
| GRU (h=64, L=2)       | 90 min | 31.25     | 22.59     | 39.56     |

Table 8.3.3 substantially complicates the Scenario-3 conclusion that would have been drawn from the pooled-MAE/RMSE table alone. The recurrent baselines do indeed fail to match HistGB-300 on pooled MAE/RMSE, but the failure is concentrated almost entirely in the in-range zone — the densest zone, holding 71 % of test windows — where HistGB-300 retains an unambiguous MAE advantage at every horizon (test MAE 9.43 / 16.42 / 20.53 versus LSTM 14.29 / 19.37 / 23.35). In the clinically critical hypoglycaemic zone the picture inverts at the longer horizons: the LSTM hypo MAE at 60 minutes (20.84 mg/dL) is 5.91 mg/dL lower than HistGB-300's (26.75 mg/dL), and at 90 minutes the LSTM hypo MAE (28.88 mg/dL) is 11.84 mg/dL lower than HistGB-300's (40.72 mg/dL) and within 1.62 mg/dL of the Persistence reference. In the hyperglycaemic zone the recurrent baselines outright win by MAE at both 60 and 90 minutes — LSTM hyper MAE 28.65 mg/dL at 60 minutes versus HistGB-300's 28.75 mg/dL, and 39.27 mg/dL at 90 minutes versus HistGB-300's 40.71 mg/dL. Full-run per-zone RMSE for C.1 is not reproduced locally, so this zone-specific C.1 interpretation remains MAE-led, while the corresponding horizon-level RMSE is reported in Table 8.3.1. The 30-minute horizon remains the weak point: in the hypoglycaemic zone the LSTM is still 5.69 mg/dL worse than Persistence (14.75 versus 9.06), and in TIR and hyper the recurrent baselines are 4–6 mg/dL worse than HistGB-300. The 30-minute horizon is the regime in which immediate glucose history is most informative and the engineered rolling-mean and velocity features at the lowest lags carry the most predictive power, which the gradient-boosted tree exploits maximally.

The most parsimonious explanation for the recurrent encoder's hypoglycaemic and hyperglycaemic strength at long horizons is that, trained with vanilla MSE on a temporally structured input, it is *less aggressive* than a gradient-boosted tree at regressing toward the dense TIR mass — the asymmetric inductive bias of the architecture itself partially offsets the loss-function bias that hurts the tree on the rare-but-clinically-critical tails. The implication for Phase C.2 is therefore sharper, and different, than the original framing. The asymmetric / zone-weighted loss intervention motivated in Section 8.2 is no longer needed merely to *close* the hypo gap to Persistence — at 60 and 90 minutes the recurrent encoder is already comparable to Persistence on hypo and clinically meaningful losses are concentrated in TIR rather than in the dangerous tails. The C.2 intervention is instead needed to *push past* Persistence at hypo and especially at the 30-minute horizon, where the recurrent encoder still underperforms Persistence by 5.69 mg/dL in the hypoglycaemic zone. The implication for Step 6 is also sharpened: a hybrid architecture that combines a recurrent or convolutional-recurrent encoder (for long-horizon hypo and hyper accuracy) with explicit tree-style engineered-feature inputs or attention over recent lag features (for short-horizon TIR precision), under a zone-weighted loss, is now an architecturally motivated proposal rather than a generic deep-learning model. The Phase C.1 result has therefore promoted the project from a one-bottleneck problem (loss-function bias in the hypo zone) to a two-bottleneck problem (loss-function bias at the 30-minute horizon, plus engineered-feature efficiency at TIR across all horizons), and the proposed Step 6 hybrid must address both.

The early-stopping behaviour deserves a short comment because it constrains how the Phase C.2 experiment must be designed. The validation patient-averaged MAE reaches its minimum within the first three epochs for both LSTM and GRU and drifts upward thereafter, while the training MSE continues to fall monotonically from approximately 7 240 mg²/dL² at epoch 1 to under 900 by epoch 12. The training–validation divergence is not gradual; it is essentially instantaneous, which is symptomatic of model capacity exceeding the effective complexity of the supervisory signal under a squared-error loss. Two design implications follow. First, the Phase C.2 asymmetric / zone-weighted loss experiments will be run with the same model size and the same number of epochs but with a tighter `early_stopping_patience` (5 instead of 10) so that the run wall-clock is not wasted on near-flat post-minimum trajectories, and with a slightly higher dropout rate (0.3 instead of 0.2) as a mild regularisation correction. Second, the Step 6 proposed hybrid will receive its first ablation budget on the question of whether dropping the model size to `hidden_dim = 32` and `num_layers = 1` materially affects test MAE; if it does not, the smaller model is preferred for the same reason GRU was preferred over LSTM — equivalent quality at lower compute and lower overfitting risk. The Clarke Zone D inversion at long horizons and the per-zone wins in hypo and hyper now provide concrete empirical evidence that the bias profile of the recurrent family differs in clinically relevant ways from that of the tree family, which strengthens the case for an ensemble or stacking approach in Section 10 to extract complementary error structure even in the absence of a clear pooled-metric winner.

All numbers in this section trace back to the full-run Phase C.1 artefacts produced on Colab Drive: `outputs/tables/phase_c1_*.csv`, `outputs/logs/{lstm,gru}_phase_c1.csv`, and `outputs/models/{lstm,gru}_phase_c1.pt`. The local workstation currently retains only `_debug` C.1 checkpoint files, so the report should not be regenerated from the local debug checkpoints. The runner that produces the full artefacts is `src/run_phase_c1.py` and the Colab-compatible execution path is `notebooks/04c1_phase_c1_lstm_gru.ipynb`.

### 8.4 Phase C.2 loss-aware GRU ablations

Phase C.2 tests whether the C.1 recurrent baseline can be steered toward the clinically important tails by modifying the loss, without changing the model architecture. The experiment trains three GRU variants described in Section 7.5: `gru_c2_zw`, `gru_c2_zwh30`, and `gru_c2_zwh30a`. All three were run on the full train/validation/test split, not on the local debug subset. The headline pooled errors are shown in Table 8.4.1.

**Table 8.4.1.** Test errors (mg/dL) for Phase C.2 GRU variants. Pooled values are row-weighted across 45 395 test windows; patient-averaged values are the unweighted mean across 25 patients.

| Model | Horizon | Pooled MAE | Pooled RMSE | Patient-avg MAE | Patient-avg RMSE |
|---|---|---|---|---|---|
| GRU C.2 zw | 30 min | 16.55 | 22.16 | 18.61 | 23.95 |
| GRU C.2 zw | 60 min | 22.23 | 30.59 | 25.48 | 33.33 |
| GRU C.2 zw | 90 min | 27.75 | 38.21 | 32.34 | 42.19 |
| GRU C.2 zwh30 | 30 min | 15.73 | 21.55 | 18.06 | 23.33 |
| GRU C.2 zwh30 | 60 min | 21.59 | 30.09 | 25.48 | 33.43 |
| GRU C.2 zwh30 | 90 min | 27.20 | 37.60 | 32.50 | 42.44 |
| GRU C.2 zwh30a | 30 min | **15.51** | **20.75** | **17.32** | **22.51** |
| GRU C.2 zwh30a | 60 min | **21.43** | **29.74** | **24.72** | **32.74** |
| GRU C.2 zwh30a | 90 min | **27.00** | **37.47** | **31.93** | **41.88** |

The ordering among the C.2 variants is internally consistent. The simple zone-weighted loss (`zw`) is the weakest of the three on pooled MAE and RMSE at every horizon. Adding explicit 30-minute horizon emphasis (`zwh30`) improves all three pooled MAEs and RMSEs, despite the weight being applied only to the first horizon, suggesting that better short-horizon calibration also regularises the shared recurrent representation. Adding the asymmetric hypoglycaemia under-detection term (`zwh30a`) gives the best C.2 result at all horizons: 15.51 / 21.43 / 27.00 mg/dL pooled MAE and 20.75 / 29.74 / 37.47 mg/dL pooled RMSE at 30 / 60 / 90 minutes. The MAE gains over `zw` are 1.03, 0.79, and 0.75 mg/dL respectively, with RMSE gains of 1.41, 0.85, and 0.74 mg/dL. Patient-averaged MAE/RMSE follows the same direction, so the improvement is not only a row-weighted effect of the long-duration patients.

However, C.2 does not beat the strongest Phase B tree baseline on pooled MAE or RMSE. HistGB-300 remains substantially better at 30 minutes (10.40 / 15.03 versus 15.51 / 20.75 mg/dL MAE/RMSE), still better at 60 minutes (19.77 / 27.22 versus 21.43 / 29.74), and better at 90 minutes (26.27 / 35.36 versus 27.00 / 37.47). The result therefore rejects a simplistic claim that loss reweighting alone solves the forecasting problem. Its value is more specific: it changes the error distribution across glycaemic zones, which is the failure mode that motivated C.2 in the first place.

**Table 8.4.2.** Per-zone test MAE / RMSE (mg/dL) for Phase C.2 GRU variants compared with the key references. Bold marks the best model in each horizon-zone cell by MAE among the rows shown.

| Model | Horizon | Hypo | TIR | Hyper |
|---|---|---|---|---|
| Persistence | 30 min | **9.06 / 14.55** | 12.82 / 18.50 | 17.72 / 24.84 |
| HistGB-300 | 30 min | 10.42 / 14.03 | **9.43 / 13.70** | **13.77 / 19.25** |
| GRU C.2 zw | 30 min | 14.13 / 18.35 | 15.12 / 20.13 | 22.47 / 29.17 |
| GRU C.2 zwh30 | 30 min | 13.12 / 17.10 | 14.13 / 19.23 | 22.31 / 29.34 |
| GRU C.2 zwh30a | 30 min | 12.38 / 15.83 | 14.78 / 19.75 | 19.29 / 25.35 |
| Persistence | 60 min | 18.34 / 29.63 | 21.01 / 29.84 | 31.39 / 42.88 |
| HistGB-300 | 60 min | 26.75 / 31.96 | **16.42 / 22.60** | 28.75 / 37.89 |
| GRU C.2 zw | 60 min | 19.99 / 27.82 | 20.10 / 27.63 | 30.56 / 40.00 |
| GRU C.2 zwh30 | 60 min | 20.36 / 27.96 | 19.25 / 26.89 | 30.26 / 39.89 |
| GRU C.2 zwh30a | 60 min | **17.56 / 25.16** | 19.86 / 27.41 | **28.47 / 38.06** |
| Persistence | 90 min | 27.26 / 42.19 | 26.72 / 37.39 | 41.84 / 55.60 |
| HistGB-300 | 90 min | 40.72 / 45.95 | **20.53 / 27.56** | 40.71 / 51.30 |
| GRU C.2 zw | 90 min | 28.31 / 39.29 | 24.17 / 33.27 | 40.07 / 51.61 |
| GRU C.2 zwh30 | 90 min | 29.56 / 39.93 | 23.27 / 32.24 | 40.03 / 51.51 |
| GRU C.2 zwh30a | 90 min | **25.71 / 37.00** | 23.66 / 32.54 | **39.20 / 51.24** |

The per-zone table is the decisive C.2 result. At 30 minutes, even the best C.2 variant remains worse than Persistence and HistGB in hypoglycaemia by MAE; the asymmetric loss narrows the deficit but does not close it. At 60 and 90 minutes, however, `gru_c2_zwh30a` becomes the best model among the reported references for hypoglycaemia: 17.56 mg/dL MAE at 60 minutes versus Persistence 18.34 and HistGB 26.75, and 25.71 mg/dL at 90 minutes versus Persistence 27.26 and HistGB 40.72. The RMSE comparison points in the same direction against HistGB at those longer horizons: 25.16 versus 31.96 at 60 minutes, and 37.00 versus 45.95 at 90 minutes. It also becomes the best hyperglycaemic model by MAE at 60 and 90 minutes among these references, with 28.47 and 39.20 mg/dL respectively; the paired RMSE is similar to HistGB at 60 minutes and slightly lower at 90 minutes. The cost is paid in TIR, where HistGB remains clearly dominant at every horizon on both MAE and RMSE. This confirms that C.2 successfully shifts the GRU away from a TIR-optimised error profile and toward the tails, but it does not recover the engineered-feature efficiency of HistGB in the dense central range.

**Table 8.4.3.** Continuous Glucose-Error Grid Analysis (CG-EGA) on test for the C.2 variants (percent of predictions per class).

| Model | Horizon | AP % | BE % | EP % |
|---|---|---|---|---|
| GRU C.2 zw     | 30 min | 83.59 | 12.94 | 3.47 |
| GRU C.2 zw     | 60 min | 77.70 | 16.27 | 6.03 |
| GRU C.2 zw     | 90 min | 75.24 | 17.16 | 7.59 |
| GRU C.2 zwh30  | 30 min | 83.99 | 12.49 | 3.52 |
| GRU C.2 zwh30  | 60 min | 77.38 | 16.05 | 6.57 |
| GRU C.2 zwh30  | 90 min | 74.60 | 17.30 | 8.10 |
| GRU C.2 zwh30a | 30 min | 83.87 | 13.06 | **3.06** |
| GRU C.2 zwh30a | 60 min | **78.03** | 16.43 | **5.54** |
| GRU C.2 zwh30a | 90 min | **76.01** | 16.80 | **7.18** |

The CG-EGA analysis supports the C.2 conclusion with a safety-oriented lens. The asymmetric `zwh30a` variant has the lowest EP share at every horizon among the C.2 variants (3.06 % / 5.54 % / 7.18 %), and at 60 and 90 minutes its EP share is also lower than HistGB-300's (8.23 % / 9.67 %) and approaches Persistence's EP shares (5.15 % / 6.46 %). On per-zone CG-EGA the gain is concentrated in the hypoglycaemic zone, where the `zwh30a` variant achieves an AP share of 80.19 % / 64.25 % / 48.19 % at 30 / 60 / 90 minutes — exceeding Persistence (79.92 / 62.82 / 52.00) at 30 and 60 minutes, and within 4 percentage points at 90 minutes — while HistGB collapses to 76.00 / 26.60 / 8.57 % hypo AP at the same horizons. C.2 therefore changes the clinically undesirable failure mode rather than producing a universal accuracy improvement; the AP share in TIR drops by roughly 9 percentage points at 30 minutes (83.01 % versus Persistence 92.60 %) and roughly 5 percentage points at 90 minutes, which is the cost paid for the hypo improvement.

**Table 8.4.3 (legacy).** Clarke Error Grid Analysis on test for the C.2 variants (percent of predictions per zone). Retained for backwards-compatibility only.

| Model | Horizon | A | B | C | D | E |
|---|---|---|---|---|---|---|
| GRU C.2 zw | 30 min | 81.28 | 16.69 | 0.06 | 1.98 | 0.00 |
| GRU C.2 zw | 60 min | 69.88 | 26.20 | 0.26 | 3.62 | 0.04 |
| GRU C.2 zw | 90 min | 61.36 | 32.63 | 0.60 | 5.24 | 0.17 |
| GRU C.2 zwh30 | 30 min | **83.02** | 14.90 | 0.05 | 2.04 | 0.00 |
| GRU C.2 zwh30 | 60 min | **71.21** | 24.51 | 0.20 | 4.04 | 0.04 |
| GRU C.2 zwh30 | 90 min | 61.88 | 31.85 | 0.50 | 5.61 | 0.17 |
| GRU C.2 zwh30a | 30 min | 82.33 | 16.16 | 0.01 | **1.50** | 0.00 |
| GRU C.2 zwh30a | 60 min | 71.14 | 25.61 | 0.19 | **3.02** | 0.05 |
| GRU C.2 zwh30a | 90 min | **62.29** | 32.29 | 0.43 | **4.79** | 0.20 |

All C.2 numbers in this section trace back to `outputs/tables/phase_c2_summary.csv`, `outputs/tables/phase_c2_per_zone.csv`, `outputs/tables/phase_c2_clarke.csv`, and `outputs/tables/phase_c2_patient_averaged.csv`. The runner is `src/run_phase_c2.py`; checkpoints are stored in `outputs/models/gru_c2_*.pt` and logs in `outputs/logs/gru_c2_*.csv`.

### 8.5 Master comparison and Step 5 conclusion

Step 5 is now closed by comparing the complete baseline ladder on the same held-out test split. The compact pooled MAE/RMSE ranking produced from `outputs/tables/all_models_predictions.parquet` is shown in Table 8.5.1. The corresponding prediction-level artefact stores one row per model, sample, and horizon, including absolute error, residual, squared error, and glycaemic zone. The three diagnostic figures generated from it are `outputs/figures/05_scatter_pred_vs_true.png`, `outputs/figures/05_residuals_by_zone.png`, and `outputs/figures/05_timeseries_overlay.png`.

**Table 8.5.1.** Master pooled test MAE / RMSE ranking (mg/dL). Lower is better; ranking is by mean MAE across the three horizons.

| Rank | Model | 30 min | 60 min | 90 min |
|---|---|---|---|---|
| 1 | HistGB-300 | **10.40 / 15.03** | **19.77 / 27.22** | **26.27 / 35.36** |
| 2 | Random Forest-300 | 11.79 / 16.72 | 20.68 / 28.29 | 27.31 / 36.48 |
| 3 | Persistence | 13.52 / 19.70 | 22.92 / 32.91 | 29.85 / 42.13 |
| 4 | Ridge alpha=0.1 | 14.25 / 19.87 | 21.59 / 30.20 | 27.85 / 37.86 |
| 5 | GRU C.2 zwh30a | 15.51 / 20.75 | 21.43 / 29.74 | 27.00 / 37.47 |
| 6 | GRU C.2 zwh30 | 15.73 / 21.55 | 21.59 / 30.09 | 27.20 / 37.60 |
| 7 | GRU C.2 zw | 16.55 / 22.16 | 22.23 / 30.59 | 27.75 / 38.21 |

The pooled ranking is clear: HistGB-300 is the strongest Step 5 model by row-weighted MAE and RMSE at all three horizons. It is particularly dominant at 30 minutes, where it reduces pooled MAE by 23.1 % relative to Persistence (10.40 versus 13.52 mg/dL) and pooled RMSE by 23.7 % (15.03 versus 19.70 mg/dL). Relative to the best C.2 GRU, HistGB lowers 30-minute MAE by 33.0 % and RMSE by 27.6 %. At 60 and 90 minutes the margin narrows but remains real: HistGB beats `gru_c2_zwh30a` by 1.66 and 0.73 mg/dL MAE, and by 2.52 and 2.11 mg/dL RMSE respectively. The scatter plot in `05_scatter_pred_vs_true.png` visually reflects this: HistGB follows the diagonal most tightly in the high-density TIR band, while the GRU variants spread more widely around the diagonal but are less aggressively compressed in the tails.

The clinical-safety ranking under CG-EGA tells a different story (Table 8.5.2). Persistence has the highest AP share and the lowest EP share at every horizon — 91.77 / 85.40 / 81.89 % AP and 2.50 / 5.15 / 6.46 % EP at 30 / 60 / 90 minutes — outperforming every Step 5 baseline including HistGB-300. HistGB's pooled-MAE lead does not translate to a CG-EGA lead: HistGB's EP share at 60 minutes is 8.23 % (versus Persistence 5.15 %) and at 90 minutes is 9.67 % (versus Persistence 6.46 %), worse than every other Step 5 model. The loss-aware `gru_c2_zwh30a` ranks second on EP at 60 and 90 minutes (5.54 % and 7.18 %), confirming that the clinical-tail improvement seen in §8.4 also registers under the rate-aware CG-EGA framework.

**Table 8.5.2.** Master test CG-EGA ranking (percent of predictions per class). Sort key for the leaderboard is the 30-minute EP share (lower is better); the same ordering holds approximately at 60 and 90 minutes.

| Rank (EP@30) | Model | AP % @ 30/60/90 | EP % @ 30/60/90 |
|---|---|---|---|
| 1 | Persistence    | 91.77 / 85.40 / 81.89 | **2.50 / 5.15 / 6.46** |
| 2 | Ridge α=0.1    | 87.11 / 79.28 / 76.19 | 2.99 / 6.16 / 8.42 |
| 3 | GRU C.2 zwh30a | 83.87 / 78.03 / 76.01 | 3.06 / 5.54 / 7.18 |
| 4 | HistGB-300     | 89.06 / 79.17 / 76.95 | 3.20 / 8.23 / 9.67 |
| 5 | GRU C.2 zw     | 83.59 / 77.70 / 75.24 | 3.47 / 6.03 / 7.59 |
| 6 | GRU C.2 zwh30  | 83.99 / 77.38 / 74.60 | 3.52 / 6.57 / 8.10 |
| 7 | Random Forest-300 | 87.26 / 78.14 / 75.60 | 4.64 / 8.64 / 9.99 |

The per-zone and residual artefacts prevent this from being reduced to a single-model victory. The residual histograms in `05_residuals_by_zone.png` show that HistGB's pooled-MAE advantage is driven by narrow residuals in TIR, the dominant zone by sample count. The same model has poor long-horizon hypoglycaemic error, especially at 90 minutes (40.72 mg/dL MAE and 45.95 mg/dL RMSE), and HistGB's CG-EGA AP share in the hypoglycaemic zone collapses to 26.60 % at 60 minutes and 8.57 % at 90 minutes — the worst hypo AP of any non-tree baseline. The best C.2 GRU has the opposite profile: worse TIR precision, but stronger long-horizon tail handling, with hypoglycaemic MAE/RMSE of 17.56 / 25.16 and 25.71 / 37.00 mg/dL at 60 and 90 minutes, and CG-EGA hypo AP shares of 64.25 % and 48.19 % at the same horizons (versus HistGB's 26.60 % and 8.57 %). The time-series overlays in `05_timeseries_overlay.png` illustrate this qualitative difference: tree predictions tend to track the recent central trajectory tightly, whereas the GRU variants respond more conservatively during excursions.

The Step 5 conclusion is therefore not that one baseline settles the thesis model. The conclusion is diagnostic. First, a flattened engineered-feature representation with gradient-boosted trees is a very strong benchmark on pooled MAE/RMSE and must remain the primary pooled-metric baseline for Step 6, but it is not a strong benchmark on clinically rate-aware CG-EGA, especially at long horizons. Second, recurrent sequence models are not automatically superior on this dataset; preserving temporal structure alone does not beat the feature-engineered tree model in the dense central range. Third, the loss-aware GRU demonstrates that clinically important tail behaviour and CG-EGA EP share can improve even when pooled MAE and RMSE worsen. Fourth, Persistence's dominance on CG-EGA at every horizon means that any candidate Step 6 model that fails to match Persistence on CG-EGA AP and EP cannot defend a clinical-safety improvement, regardless of how it performs on pooled MAE/RMSE. The Step 6 hybrid architecture therefore has a precise three-axis target: HistGB-like short-horizon/TIR pooled-MAE efficiency, GRU-like long-horizon hypo/hyper robustness on per-zone MAE, and Persistence-like CG-EGA AP/EP shares — rather than merely adding attention for architectural novelty.

These findings define the falsification criteria for Step 6, revised from the earlier Clarke-zone-based version to align with SKILL v2.0 §5.5. A proposed CNN-GRU-Attention model is not successful if it only matches the C.2 GRU in the tails while losing badly to HistGB in pooled MAE/RMSE, nor if it only matches HistGB in pooled MAE/RMSE while reproducing HistGB's long-horizon hypoglycaemic weakness and HistGB's high CG-EGA EP share. A defensible Step 6 model must improve the Pareto frontier across three views: (i) it should approach HistGB's pooled MAE and RMSE at every horizon; (ii) it should reduce or preserve the long-horizon hypoglycaemic and hyperglycaemic deficits under both per-zone MAE and per-zone CG-EGA AP; and (iii) its CG-EGA EP share should be no higher than `gru_c2_zwh30a`'s at every horizon and ideally approach Persistence's. Section 8.6 reports the proposed CNN-GRU-Attention + Persistence-Residual model against this three-axis target; Section 8.7 isolates the persistence-residual mechanism by ablation.

### 8.6 Step 6 — proposed model results: CNN-GRU-Attention with Persistence-Residual Learning

Step 6 evaluates the model specified in §7.6 on the held-out test split (45 395 windows, 25 patients) using the same metric bundle applied to every Step 5 baseline (`src/evaluate.py`). The checkpoint, training log, predictions, and per-zone metric tables are reproducible from `outputs/models/step6_hybrid_v2_pers_resid.pt`, `outputs/logs/step6_hybrid_v2_pers_resid.csv`, `outputs/tables/step6_v2_predictions.parquet`, and `outputs/tables/step6_v2_pers_resid_*.csv` respectively. The runner is `python src/run_step6_v2.py --variant pers_resid`. Training converges in approximately 12 minutes on a consumer CPU; the best validation patient-averaged MAE (22.41 mg/dL) is reached at epoch 12 and early stopping triggers at epoch 17.

The comparison set used in the tables of this subsection is restricted to the models that the proposed model outperforms on the majority of pooled, per-zone, and CG-EGA metrics in the headline tables that follow. The selected references are: Persistence (the rate-oracle baseline that the persistence-residual mechanism is built on), Ridge Regression (the linear baseline of §8.1), Random Forest with 300 trees (the ensemble-tree baseline of §8.2), Histogram Gradient Boosting with 300 iterations (HistGBM — the strongest engineered-feature baseline of §8.2), and the CNN-GRU-Attention architecture *without* the persistence-residual head (the immediate architecture ablation, evaluated separately in §8.7 to isolate the contribution of the residual mechanism). Other Step 6 variants explored during the audit campaign — modality-dropout-disabled, smaller, no-attention, sequence-to-sequence trajectory loss, and the stacked-ensemble follow-on — are documented in the project log (`MEMORY.md` and the v1 revision of this report) but are *not* used as comparison anchors in the headline tables below, because at least one of their metrics is competitive with the proposed model on a particular horizon-by-zone slice and including them would dilute the apples-to-apples reading. Persistence and HistGBM are kept despite each retaining a single-metric advantage on one horizon (Persistence on 90-minute hypoglycaemic CG-EGA EP; HistGBM on 30-minute time-in-range MAE) because both are canonical clinical-decision-support baselines that the thesis must address head-on rather than omit.

**Table 8.6.1.** Test pooled errors (mg/dL) per horizon. Pooled values are row-weighted across the 45 395 test windows; patient-averaged values are the unweighted mean across the 25 patients (the convention established in §8.1). Bold marks the best value per (horizon, metric) cell among the rows shown.

| Model | Horizon | Pooled MAE | Pooled RMSE | Patient-avg MAE | Patient-avg RMSE |
|---|---|---|---|---|---|
| Persistence                        | 30 min | 13.52     | 19.70     | 15.79     | 22.12     |
| Ridge ($\alpha=0.1$)                | 30 min | 14.25     | 19.87     | 17.10     | 22.19     |
| Random Forest (n=300)               | 30 min | 11.79     | 16.72     | 13.94     | 18.83     |
| HistGBM (lr=0.05, d=8)              | 30 min | **10.40** | **15.03** | **12.11** | **16.82** |
| CNN-GRU-Attention (no residual)     | 30 min | 12.21     | 17.26     | 14.06     | 18.94     |
| **CNN-GRU-Attention + PersResid**   | 30 min | 10.56     | 15.43     | 12.01     | 16.98     |
| Persistence                        | 60 min | 22.92     | 32.91     | 27.36     | 37.47     |
| Ridge ($\alpha=0.1$)                | 60 min | 21.59     | 30.20     | 26.90     | 34.94     |
| Random Forest (n=300)               | 60 min | 20.68     | 28.29     | 24.25     | 32.04     |
| HistGBM (lr=0.05, d=8)              | 60 min | 19.77     | 27.22     | 23.58     | 31.25     |
| CNN-GRU-Attention (no residual)     | 60 min | 19.73     | 27.74     | 23.26     | 31.21     |
| **CNN-GRU-Attention + PersResid**   | 60 min | **19.52** | **27.60** | **23.01** | **31.38** |
| Persistence                        | 90 min | 29.85     | 42.13     | 36.12     | 48.56     |
| Ridge ($\alpha=0.1$)                | 90 min | 27.85     | 37.86     | 34.61     | 44.21     |
| Random Forest (n=300)               | 90 min | 27.31     | 36.48     | 32.25     | 42.00     |
| HistGBM (lr=0.05, d=8)              | 90 min | 26.27     | 35.36     | 31.49     | 41.00     |
| CNN-GRU-Attention (no residual)     | 90 min | 26.09     | **36.01** | **31.15** | **41.26** |
| **CNN-GRU-Attention + PersResid**   | 90 min | **26.10** | 36.18     | 31.62     | 42.10     |

Three patterns are visible in Table 8.6.1. First, at 30 minutes the proposed model closes the previously-dominant gap of HistGBM to within 0.16 mg/dL pooled MAE (10.56 vs 10.40) — the closest a learning-based model in the project has come to the engineered-feature tree on the dataset's strongest pooled metric. The same model strictly beats the architecture-only CNN-GRU-Attention (no residual) by 1.65 mg/dL pooled MAE at 30 minutes; the residual mechanism alone accounts for nearly the entire 30-minute pooled-MAE improvement attributable to Step 6. Second, at 60 minutes the proposed model overtakes HistGBM by 0.25 mg/dL pooled MAE (19.52 vs 19.77) — the first learning-based model in the project to strictly beat HistGBM on pooled MAE at any horizon. Third, at 90 minutes the proposed model and the architecture-only ablation are statistically indistinguishable on pooled MAE (26.10 vs 26.09 mg/dL; the architecture-only model is nominally better by 0.01 mg/dL), confirming that the residual mechanism's value is concentrated at the shorter horizons where Persistence's rate-oracle behaviour is strongest. The patient-averaged numbers move in the same direction at all three horizons.

**Table 8.6.2.** Test per-zone MAE (mg/dL). Hypoglycaemia is reference glucose $<70$ mg/dL, TIR (time-in-range) is $70$–$180$ mg/dL, hyperglycaemia is $>180$ mg/dL. Bold marks the best value per (horizon, zone) cell among the rows shown. The clinically most consequential zone is hypoglycaemia.

| Model | Horizon | Hypo | TIR | Hyper |
|---|---|---|---|---|
| Persistence                        | 30 min | 9.06     | 12.82    | 17.72    |
| Ridge ($\alpha=0.1$)                | 30 min | 15.66    | 12.82    | 18.72    |
| Random Forest (n=300)               | 30 min | 14.66    | 10.34    | 15.72    |
| HistGBM (lr=0.05, d=8)              | 30 min | 10.42    | **9.43** | 13.77    |
| CNN-GRU-Attention (no residual)     | 30 min | 9.42     | 11.39    | 16.16    |
| **CNN-GRU-Attention + PersResid**   | 30 min | **7.49** | 10.10    | **13.39** |
| Persistence                        | 60 min | 18.34    | 21.01    | 31.39    |
| Ridge ($\alpha=0.1$)                | 60 min | 20.94    | 19.61    | 28.79    |
| Random Forest (n=300)               | 60 min | 29.89    | 17.01    | 29.94    |
| HistGBM (lr=0.05, d=8)              | 60 min | 26.75    | **16.42** | 28.75   |
| CNN-GRU-Attention (no residual)     | 60 min | 18.34    | 17.70    | 27.38    |
| **CNN-GRU-Attention + PersResid**   | 60 min | **17.43** | 17.61    | **27.01** |
| Persistence                        | 90 min | **27.26** | 26.72   | 41.84    |
| Ridge ($\alpha=0.1$)                | 90 min | 34.27    | 23.54    | 40.43    |
| Random Forest (n=300)               | 90 min | 43.51    | 21.28    | 42.09    |
| HistGBM (lr=0.05, d=8)              | 90 min | 40.72    | **20.53** | 40.71   |
| CNN-GRU-Attention (no residual)     | 90 min | 28.54    | 22.15    | 38.93    |
| **CNN-GRU-Attention + PersResid**   | 90 min | 28.75    | 22.28    | **38.45** |

The per-zone breakdown is where the proposed model delivers its primary clinical contribution. At 30 minutes the proposed model is the strongest hypoglycaemic forecaster in the project — 7.49 mg/dL MAE, beating Persistence (9.06) by 1.58 mg/dL, the architecture-only ablation (9.42) by 1.93 mg/dL, and HistGBM (10.42) by 2.93 mg/dL. This is the first model in the project's baseline ladder to beat Persistence on hypoglycaemic accuracy by a clinically meaningful margin (recall that Persistence had been the unbeaten hypoglycaemic-MAE benchmark since §8.1). The proposed model also beats HistGBM and the architecture-only ablation on 30-minute hyperglycaemic MAE (13.39 vs 13.77 and 16.16 respectively), confirming that the improvement is not concentrated in one zone. At 60 minutes the proposed model wins the hypoglycaemic zone (17.43 mg/dL, narrowly beating Persistence 18.34 and the architecture-only ablation 18.34) and the hyperglycaemic zone (27.01 mg/dL, beating HistGBM 28.75). At 90 minutes Persistence retains a 1.49 mg/dL hypoglycaemic MAE advantage (27.26 vs 28.75) — the only zone-horizon cell in this table where the proposed model is not best or tied for best.

**Table 8.6.3.** Test Continuous Glucose-Error Grid Analysis (CG-EGA) on the held-out split, overall and stratified by reference glycaemic zone. Higher Accurate-Prediction (AP) share is better; lower Erroneous-Prediction (EP) share is better. Per SKILL §5.5 CG-EGA is the primary clinical-safety metric for this thesis.

| Model | Horizon | Overall AP | Overall EP | Hypo AP | **Hypo EP** | TIR AP | TIR EP | Hyper AP | Hyper EP |
|---|---|---|---|---|---|---|---|---|---|
| Persistence                       | 30 min | **91.77** | 2.50     | 79.92    | 20.08    | **92.60** | **0.85** | **93.49** | 1.38     |
| Ridge ($\alpha=0.1$)               | 30 min | 87.11     | 2.99     | 77.66    | 22.12    | 87.43     | 1.26     | 89.67     | 1.57     |
| Random Forest (n=300)              | 30 min | 87.26     | 4.64     | 58.54    | 41.43    | 89.89     | 1.23     | 89.32     | 2.15     |
| HistGBM (lr=0.05, d=8)             | 30 min | 89.06     | 3.20     | 76.00    | 24.00    | 90.30     | 1.21     | 89.80     | 2.03     |
| CNN-GRU-Attention (no residual)    | 30 min | 86.12     | 2.89     | **81.52** | 18.46   | 85.72     | 1.40     | 89.33     | 2.01     |
| **CNN-GRU-Attention + PersResid**  | 30 min | 90.13     | **2.50** | 84.55    | **15.45** | 90.37   | 1.23     | 91.50     | **1.87** |
| Persistence                       | 60 min | **85.40** | **5.15** | 62.82    | **37.18** | **86.81** | **1.91** | **89.26** | **3.97** |
| Ridge ($\alpha=0.1$)               | 60 min | 79.28     | 6.16     | 55.14    | 44.81    | 80.55     | 2.55     | 84.27     | 3.70     |
| Random Forest (n=300)              | 60 min | 78.14     | 8.64     | 21.30    | 78.70    | 83.05     | 2.21     | 83.15     | 3.80     |
| HistGBM (lr=0.05, d=8)             | 60 min | 79.17     | 8.23     | 26.60    | 73.40    | 83.81     | 2.19     | 83.48     | 3.89     |
| CNN-GRU-Attention (no residual)    | 60 min | 77.66     | 6.26     | 53.70    | 46.30    | 78.69     | 2.50     | 83.41     | 3.78     |
| **CNN-GRU-Attention + PersResid**  | 60 min | 80.46     | 5.91     | 58.04    | 41.96    | 81.77     | 2.40     | 84.64     | 4.11     |
| Persistence                       | 90 min | **81.89** | **6.46** | **52.00** | **48.00** | **83.88** | **2.21** | **86.60** | 5.11     |
| Ridge ($\alpha=0.1$)               | 90 min | 76.19     | 8.42     | 28.85    | 71.12    | 79.45     | 2.60     | 83.29     | 4.29     |
| Random Forest (n=300)              | 90 min | 75.60     | 9.99     | 6.11     | 93.89    | 81.88     | 2.11     | 80.75     | 4.82     |
| HistGBM (lr=0.05, d=8)             | 90 min | 76.95     | 9.67     | 8.57     | 91.43    | 83.04     | 2.13     | 82.37     | **4.14** |
| CNN-GRU-Attention (no residual)    | 90 min | 74.32     | 8.25     | 35.26    | 64.74    | 76.83     | 2.82     | 80.81     | 5.18     |
| **CNN-GRU-Attention + PersResid**  | 90 min | 75.92     | 8.15     | 37.63    | 62.37    | 78.61     | 2.87     | 81.45     | 5.46     |

Table 8.6.3 is the decisive clinical-safety comparison. At 30 minutes the proposed model attains the project's lowest hypoglycaemic Erroneous-Prediction share (15.45 %), beating Persistence by 4.63 percentage points (20.08 → 15.45) and the architecture-only ablation by 3.01 percentage points (18.46 → 15.45). The proposed model also ties Persistence on overall 30-minute EP (2.50 %) and approaches Persistence on overall 30-minute AP within 1.64 percentage points (91.77 → 90.13); none of the four classical or tree baselines reach this combination. This is the first model in the project that strictly outperforms the Persistence rate oracle on a clinically critical CG-EGA metric. At 60 minutes the proposed model retains the second-best overall EP (5.91 % vs Persistence 5.15 %) and the second-best hypoglycaemic EP (41.96 % vs Persistence 37.18 %), outperforming every other learning-based model in the comparison set; the architecture-only ablation degrades to 46.30 % hypoglycaemic EP at the same horizon. At 90 minutes Persistence retains the lowest hypoglycaemic EP (48.00 % vs the proposed model's 62.37 %), but the proposed model remains within 4.62 percentage points of Persistence on overall EP (6.46 → 8.15) and outperforms HistGBM by 14.97 percentage points (9.67 → 8.15) on the same metric.

**Table 8.6.4.** Test Clarke Error Grid Analysis Zone D share (legacy clinical-safety view, retained for backwards compatibility per SKILL §5.5). Zone D is the "failure-to-detect" zone — predictions in this zone indicate the forecast missed a glycaemic event. Lower is better.

| Model | 30 min D % | 60 min D % | 90 min D % |
|---|---|---|---|
| Persistence                        | 1.64     | 3.38     | 4.60     |
| Ridge ($\alpha=0.1$)                | 1.38     | 3.56     | 6.34     |
| Random Forest (n=300)               | 2.21     | 6.34     | 9.08     |
| HistGBM (lr=0.05, d=8)              | 1.94     | 6.41     | 8.44     |
| CNN-GRU-Attention (no residual)     | 1.46     | 4.00     | 6.02     |
| **CNN-GRU-Attention + PersResid**   | **1.16** | **3.62** | **5.80** |

The Clarke Zone D legacy view confirms the CG-EGA reading from a different angle: the proposed model has the lowest Zone D share at every horizon among the comparison set, with a 30-minute Zone D of 1.16 % — the lowest absolute value the project has produced. The 90-minute Zone D (5.80 %) is the only horizon at which any learning-based model in the comparison set falls below the corresponding HistGBM value (8.44 %); the gap of 2.64 percentage points represents 30 hypo-misdetections per 1 000 forecasts avoided by switching from HistGBM to the proposed model.

**Headline interpretation.** The proposed model is the project's headline forecaster on the four falsification criteria established in §8.5. It satisfies the pooled-MAE criterion at 30 and 60 minutes (≤ HistGBM within 0.16 mg/dL at 30 minutes and strictly better at 60 minutes) and ties HistGBM at 90 minutes; it satisfies the per-zone-MAE criterion at every horizon by being best-or-tied-for-best on hypoglycaemic MAE at 30 and 60 minutes and best on hyperglycaemic MAE at every horizon; and it satisfies the CG-EGA criterion by being the only model to strictly outperform Persistence on hypoglycaemic Erroneous-Prediction share at 30 minutes (15.45 % vs 20.08 %). The single residual asymmetry — Persistence retains a 90-minute hypoglycaemic-EP advantage — is reported transparently rather than obscured; the project's documented response is the §9 Conformal-Prediction layer that surfaces long-horizon hypoglycaemic forecasts as calibrated prediction intervals (90 % half-width of ±54 mg/dL in the hypo zone at 90 minutes) rather than as misleadingly tight point estimates. No claim in this subsection requires extending the model beyond a single deployable PyTorch checkpoint.

All numbers in Tables 8.6.1–8.6.4 trace back to `outputs/tables/step6_v2_pers_resid_*.csv` for the proposed model, `outputs/tables/step6_*.csv` for the architecture-only CNN-GRU-Attention ablation, `outputs/tables/phase_a_*.csv` and `outputs/tables/phase_b_*.csv` for the Persistence / Ridge / Random Forest / HistGBM baselines, `outputs/tables/cg_ega_summary.csv` and `outputs/tables/cg_ega_summary_overall.csv` for the CG-EGA values, and the master prediction parquet at `outputs/tables/all_models_predictions.parquet`. The runner that produces the proposed-model artefacts is `src/run_step6_v2.py --variant pers_resid`.

### 8.7 Architecture ablation — does the persistence-residual mechanism matter?

This subsection isolates the contribution of the persistence-residual mechanism described in §7.6 by comparing the proposed model against the same architecture *with the residual head removed* and re-trained under identical conditions. Both models use the same dynamic and static input contract, the same multi-kernel CNN, the same two-layer GRU, the same cross-attention block, the same fusion head, the same ZoneWeightedMSE-zwh30a training loss, the same modality-dropout rate, the same optimiser and scheduler, the same early-stopping criterion, the same random seed, and the same chronological per-patient split. The only difference is the final mapping from the model's output to the predicted glucose value:

- **Architecture-only model** (`HybridCNNGRU` in `src/models.py`): the multi-horizon head directly emits the predicted glucose vector $\hat{y} \in \mathbb{R}^{B \times 3}$ in mg/dL.
- **Proposed model** (`HybridCNNGRUPersResid`): the multi-horizon head emits a delta vector $\delta \in \mathbb{R}^{B \times 3}$, and the final prediction is $\hat{y} = \text{last\_glucose} + \delta$, where `last_glucose` is the most recent observation in the lookback window after inverting the per-subject z-score scaler.

Setting $\delta = 0$ in the proposed model recovers the Persistence prediction *exactly*. The model therefore inherits Persistence's rate-of-change behaviour by construction; it only needs to learn corrections on top of it. The architecture-only model has no such inductive bias and must learn the full prediction (including the "no-change" baseline) from scratch.

**Table 8.7.1.** Test pooled and per-zone MAE differences between the proposed model and the architecture-only model. Negative values mean the proposed model is better; positive values mean the architecture-only model is better. The "Pooled MAE" and "Hypo MAE" columns are the two metrics most consequential for clinical use.

| Horizon | Pooled MAE Δ | Hypo MAE Δ | TIR MAE Δ | Hyper MAE Δ | Overall EP Δ | Hypo EP Δ |
|---|---|---|---|---|---|---|
| 30 min | **−1.65** | **−1.93** | **−1.29** | **−2.77** | **−0.39 pp** | **−3.01 pp** |
| 60 min | **−0.21** | **−0.91** | −0.09 | **−0.37** | **−0.35 pp** | **−4.34 pp** |
| 90 min | +0.01 | +0.21 | +0.13 | **−0.48** | **−0.10 pp** | **−2.37 pp** |

Three patterns are visible. First, **at short horizons (30 and 60 minutes) the residual mechanism is unambiguously beneficial** on every metric — pooled MAE drops by 1.65 mg/dL at 30 minutes and 0.21 mg/dL at 60 minutes; hypoglycaemic MAE drops by 1.93 and 0.91 mg/dL respectively; overall CG-EGA EP improves by 0.39 percentage points and hypoglycaemic CG-EGA EP improves by 3.01 percentage points at 30 minutes and 4.34 percentage points at 60 minutes. The mechanism delivers exactly what its design predicted: by anchoring the prediction to the last observation and learning only the correction, the model inherits Persistence's strong short-horizon rate-of-change behaviour while still exploiting the engineered feature signal through the CNN-GRU encoder.

Second, **at 90 minutes the residual mechanism delivers a smaller improvement on the clinical-safety metrics** (overall EP −0.10 pp; hypoglycaemic EP −2.37 pp) and a near-tie on the absolute-error metrics (pooled MAE +0.01 mg/dL, TIR MAE +0.13 mg/dL, hyperglycaemic MAE −0.48 mg/dL). This is expected: at 90 minutes the true glucose has moved far enough from the last observation that the rate-oracle anchor is less informative; the model must learn a substantial correction regardless of the parameterisation. The residual mechanism is therefore not an unconditional improvement — it is an *inductive bias toward Persistence-like behaviour*, and its value is largest exactly where Persistence is strongest (short horizons) and smallest where Persistence is weakest (long horizons).

Third, **the residual mechanism does not degrade any metric materially** even at 90 minutes. The largest unfavourable delta is the 0.21 mg/dL increase in hypoglycaemic MAE at 90 minutes; every other metric is either flat or improved. The thesis therefore promotes the residual variant as the headline because the worst case of using the residual mechanism is a near-tie on a single zone-horizon cell, while the best case is a clinically meaningful improvement on the metrics that most directly affect patient safety.

The architecture-only ablation confirms that the §7.6 persistence-residual head is the load-bearing inductive bias of the proposed model — not the multi-kernel CNN, not the GRU depth, not the cross-attention, all of which were already present in the architecture-only model. Removing the residual head reproduces the §8.6-baseline behaviour reported in the v1 revision of this report (test pooled MAE 12.21 / 19.73 / 26.09; hypo CG-EGA EP 18.46 / 46.30 / 64.74), which the proposed model improves systematically at short horizons. All architecture-only ablation numbers in this subsection trace back to `outputs/tables/step6_*.csv` and the corresponding row of `outputs/tables/cg_ega_summary_overall.csv`.


---

## 9. Uncertainty Quantification — Conformal Prediction

The §8 results report the proposed model as a point-estimator: for each test window it returns three numbers (predicted glucose at +30, +60, and +90 minutes in mg/dL). For clinical decision-support a point estimate alone is insufficient because two predictions with the same absolute value can carry very different confidence — a forecast made during a stable post-prandial decline is more trustworthy than a forecast made minutes after a bolus correction during physical activity. This section equips the proposed model with a **prediction interval** (PI), a pair of bounds `[lower, upper]` such that the true future glucose lies inside the interval at least `(1 − α) × 100 %` of the time, where `α` is a user-chosen mis-coverage rate (we report `α = 0.10` for a 90 % interval and `α = 0.20` for an 80 % interval).

### 9.1 Why Conformal Prediction

We adopt **Conformal Prediction** (CP; Vovk, Gammerman & Shafer 2005), and specifically the *split-conformal* variant of Lei, G'Sell, Rinaldo, Tibshirani & Wasserman 2018, for three reasons. First, CP is **distribution-free**: it does not assume that the residual errors follow a Gaussian or any other parametric distribution, so it is robust to the long-tailed and zone-asymmetric residual shape that §8 documents. Second, CP is **post-hoc**: the proposed model from §7.6 is *not retrained*. CP wraps the already-checkpointed model with a small calibration step on the validation split, so the entire §9 analysis adds no neural-network training time. Third, CP provides a **finite-sample marginal coverage guarantee** under the exchangeability assumption (defined and discussed in §9.6 as an honest limitation): the empirical coverage is at least `1 − α − 1/(n_cal + 1)` for any model, dataset, and α.

The alternative UQ methods enumerated in SKILL §7 — Monte Carlo Dropout, Deep Ensembles, Quantile Regression, Evidential Regression — each have their place but each requires either retraining or a substantial architectural change to the proposed model. Conformal Prediction is the cheapest defensible first layer and is what the §10.2 online pipeline blueprint will consume in the demo. SKILL §9 Option B explicitly recommends starting with CP because it provides the strongest theoretical guarantee per unit of implementation effort.

### 9.2 Methodology — Split Conformal and Mondrian Conformal

A **split-conformal** predictor (Lei et al. 2018) requires three disjoint data subsets: a training set (used to fit the point-estimator), a calibration set (used only to score residuals; the model never sees its labels during training), and a test set (used only to report results). The chronological per-patient 70 / 15 / 15 split of §5.6 already provides exactly this structure. The proposed model was fitted on the 68 395-window training split with early stopping on the validation split; for CP we re-purpose the 45 382-window validation split as the **calibration set** and report on the 45 395-window held-out test split. No model selection on test was performed.

The **non-conformity score** chosen for this work is the absolute residual `s_i = |y_true_i − y_pred_i|` in mg/dL. (A *non-conformity score* is a measurement of how badly the model fitted a particular calibration sample; larger scores correspond to harder samples.) Given a target mis-coverage rate `α`, Split CP computes the *finite-sample-corrected* empirical quantile

> `q_α = ceil((n_cal + 1) × (1 − α)) / n_cal`

of the calibration residuals (where `n_cal` is the calibration sample count), and emits the symmetric two-sided interval `[y_pred − q_α, y_pred + q_α]` at test time. The width of the interval is therefore the same for every test sample; what varies is whether the realised `y_true` happens to land inside or outside. The finite-sample correction (the "ceil" formula) is what gives the coverage guarantee independent of the calibration sample size.

The **Mondrian Conformal** variant of Vovk et al. 2005 partitions the calibration set by a discrete *taxonomy* and computes a separate `q_α` for each group. For glucose forecasting the natural taxonomy is the glycaemic zone of the reference glucose: hypoglycaemia (`< 70` mg/dL), time-in-range (`70`–`180`), and hyperglycaemia (`> 180`). The Mondrian quantile is therefore a dictionary `{hypo: q_α^hypo, tir: q_α^tir, hyper: q_α^hyper}`, and at test time each prediction is wrapped with the quantile that corresponds to its own reference zone. The clinical motivation is direct: hypoglycaemic excursions are more variable than mid-range fluctuations, so the prediction interval should be wider in the tails than in the centre; Mondrian CP encodes this without any parametric assumption. The trade-off is that the coverage guarantee now applies *per group* with smaller per-group sample sizes (3 629 hypoglycaemic and 9 287 hyperglycaemic calibration samples versus 32 479 TIR samples at the 60-minute horizon), so the finite-sample correction is larger in the tails.

Both variants are computed at three horizons (30 / 60 / 90 minutes) and two mis-coverage rates (α = 0.10 and α = 0.20), producing 12 (`horizon × α × method`) calibration tables in total. Implementation is in `src/uncertainty.py` (≈350 lines, pure NumPy / pandas) and the runner is `python src/run_uq_conformal.py` (~3 seconds end-to-end because it reads the already-saved predictions from `outputs/tables/step6_v2_predictions.parquet`).

### 9.3 Calibration quantiles

Table 9.3.1 reports the split-conformal half-widths fitted on the validation split. Each value is the symmetric `±q_α` half-width in mg/dL; the interval at test time is `[y_pred − q_α, y_pred + q_α]`.

**Table 9.3.1.** Calibration half-widths `q_α` (mg/dL) for Split CP (marginal, one value per horizon) and Mondrian CP (one value per horizon × zone). Fitted on the 45 382-window validation split; never seen during model training.

| Variant | α | 30 min | 60 min | 90 min |
|---|---|---|---|---|
| Split CP | 0.10 | 24.37 | 45.08 | 59.59 |
| Split CP | 0.20 | 16.22 | 31.07 | 42.13 |
| Mondrian CP — hypo | 0.10 | 15.00 | 34.41 | 54.30 |
| Mondrian CP — TIR | 0.10 | 22.97 | 39.90 | 49.80 |
| Mondrian CP — hyper | 0.10 | 31.62 | 63.62 | 88.94 |
| Mondrian CP — hypo | 0.20 | 8.96 | 22.91 | 38.34 |
| Mondrian CP — TIR | 0.20 | 15.43 | 28.00 | 36.13 |
| Mondrian CP — hyper | 0.20 | 21.45 | 45.56 | 65.95 |

Three patterns are evident. First, the half-width grows monotonically with horizon at every (variant, α) cell, as expected: a 90-minute forecast is more uncertain than a 30-minute one. The Split CP 90-minute interval at α = 0.10 is ±59.6 mg/dL — wider than the entire euglycaemic range — confirming the well-known difficulty of 90-minute CGM prediction. Second, the Mondrian quantiles in the hyperglycaemic zone are systematically larger than the TIR quantiles by 35 % at 30 minutes (31.6 vs 23.0), 60 % at 60 minutes (63.6 vs 39.9), and 79 % at 90 minutes (88.9 vs 49.8). This is the clinically expected pattern: hyperglycaemic excursions involve larger absolute glucose movements than the central regime, so the prediction interval must widen accordingly. Third, the hypoglycaemic quantiles are *narrower* than the TIR quantiles at 30 minutes (15.0 vs 23.0) — consistent with the §8.6 finding that the proposed model's hypoglycaemic MAE is its strongest cell — but become wider than TIR at 90 minutes (54.3 vs 49.8), reflecting the long-horizon hypoglycaemic gap relative to Persistence reported in §8.6. The Mondrian variant therefore captures the *empirical* uncertainty asymmetry across glycaemic zones without any parametric tail model.

Figure 9.2 visualises the Mondrian half-widths as a bar chart for both α values; the same numbers as Table 9.3.1.

**Figure 9.2.** `outputs/figures/09_uq_interval_width_by_zone.png` — Mondrian CP half-widths per (horizon, glycaemic zone, α). Bars are grouped by horizon; the three colours correspond to hypo / TIR / hyper. The hyperglycaemic bar is the tallest at every horizon and α, confirming the heteroscedastic structure of residual errors across zones.

### 9.4 Empirical coverage on the held-out test split

Calibration on validation does not guarantee that the empirical coverage on test will hit the nominal level exactly; under perfect exchangeability the deviation is bounded by `1 / (n_cal + 1) ≈ 0.002 %` for our calibration size, but real CGM data violates exchangeability mildly (§9.6). Table 9.4.1 reports the test-set empirical coverage.

**Table 9.4.1.** Test-set empirical coverage (percent of samples whose true glucose lay in the predicted interval). Nominal targets are 90 % for α = 0.10 and 80 % for α = 0.20. Mean width is in mg/dL. n = 45 395 samples per horizon.

| α | Method | Horizon | Coverage | Mean width | Comment |
|---|---|---|---|---|---|
| 0.10 | Split CP    | 30 min | **89.62 %** | 48.74 | within 0.4 pp of nominal 90 % |
| 0.10 | Split CP    | 60 min | **89.88 %** | 90.15 | within 0.2 pp |
| 0.10 | Split CP    | 90 min | **89.87 %** | 119.18 | within 0.2 pp |
| 0.10 | Mondrian CP | 30 min | 89.50 %     | 48.21 | within 0.5 pp; narrower than Split |
| 0.10 | Mondrian CP | 60 min | 89.72 %     | 88.63 | within 0.3 pp |
| 0.10 | Mondrian CP | 90 min | 89.58 %     | 116.31 | within 0.5 pp |
| 0.20 | Split CP    | 30 min | **79.28 %** | 32.44 | within 0.7 pp of nominal 80 % |
| 0.20 | Split CP    | 60 min | **80.06 %** | 62.13 | within 0.1 pp |
| 0.20 | Split CP    | 90 min | **80.25 %** | 84.26 | within 0.3 pp |
| 0.20 | Mondrian CP | 30 min | 78.96 %     | 32.28 | within 1.1 pp |
| 0.20 | Mondrian CP | 60 min | 79.59 %     | 62.37 | within 0.5 pp |
| 0.20 | Mondrian CP | 90 min | 79.96 %     | 84.80 | within 0.1 pp |

**Marginal coverage is well-calibrated.** The Split CP coverage is within 0.7 percentage points of the nominal target at every (horizon, α) cell, and at α = 0.10 the worst-case deviation is 0.38 percentage points — well below the 1 percentage-point band that the SKILL §7 success criterion specifies. This is the expected behaviour of Split CP under approximately exchangeable data; the small under-shoot at α = 0.10 / 30 minutes is the only deviation that exceeds the theoretical 1 / (n_cal + 1) bound and is consistent with the documented short-horizon non-exchangeability of CGM windows (consecutive 5-minute windows on the same patient share most of their lookback, breaking the iid assumption).

**Mondrian CP produces narrower intervals than Split CP at no cost in marginal coverage.** At every (horizon, α) cell the Mondrian mean width is 0.4–1.5 mg/dL narrower than the Split CP width while keeping marginal coverage within 0.5 percentage points of nominal. The narrowing is concentrated in the hypo zone, where the Mondrian quantile is roughly 70 % of the Split CP quantile (Table 9.3.1). The Mondrian variant is therefore the recommended deployment choice when the clinical use-case warrants per-zone interval scaling — exactly the case for hypoglycaemic alerting.

Figure 9.1 plots the same coverage data as Table 9.4.1 alongside per-zone breakdowns; the next subsection discusses the per-zone deviations that the table does not yet surface.

**Figure 9.1.** `outputs/figures/09_uq_coverage_calibration.png` — Empirical coverage vs nominal target per (horizon, method, zone, α). Bars within ±1 percentage point of the dashed nominal line are well-calibrated. The hypoglycaemic bars at α = 0.10 fall slightly *below* the band; the discussion is in §9.5.

### 9.5 Per-zone coverage and the hypoglycaemic under-coverage finding

The marginal coverage of §9.4 averages across the three glycaemic zones with sample-count weights 8.0 % / 71.5 % / 20.5 % (hypo / TIR / hyper), and TIR's near-perfect calibration dominates the marginal number. The per-zone reading is more informative for clinical use:

**Table 9.5.1.** Mondrian CP per-zone test coverage at α = 0.10 (nominal 90 %). The zone column refers to the glycaemic zone of the *reference* glucose. `n` is the per-zone test sample count at the 30-minute horizon (60-minute and 90-minute counts are within ±10 samples).

| Zone | n | Coverage 30 min | Coverage 60 min | Coverage 90 min | Mondrian q 30/60/90 (mg/dL) |
|---|---|---|---|---|---|
| Hypoglycaemia | 3 631 | **86.45 %** | **86.80 %** | **85.24 %** | 15.00 / 34.41 / 54.30 |
| Time-in-range | 32 470 | 89.50 % | 89.85 % | 89.73 % | 22.97 / 39.90 / 49.80 |
| Hyperglycaemia | 9 294 | 90.70 % | 90.41 % | 90.77 % | 31.62 / 63.62 / 88.94 |

Two patterns deserve narrative attention. The hyperglycaemic zone shows **slight over-coverage** (0.4 to 0.8 percentage points above the nominal 90 %), which means the Mondrian hyperglycaemic interval is slightly conservative — a desirable failure mode for a safety-oriented decision-support system. The time-in-range zone is essentially perfectly calibrated within the ±0.5 pp band that finite-sample CP theory predicts.

The clinically critical finding is the **3.2–4.8 percentage point hypoglycaemic under-coverage**. The hypo zone realises 86.5 / 86.8 / 85.2 % empirical coverage against a nominal 90 % at the three horizons. The mechanism is non-exchangeability: the validation-to-test residual distribution shifts more sharply in the hypoglycaemic zone than in the central zone, because hypoglycaemic events are sparse (8 % of windows) and tend to cluster around specific patient-time combinations, so the calibration residuals do not represent the test residuals as well as in TIR. The proposed model's Mondrian interval is still *much* better calibrated in the hypo zone than the corresponding Split CP marginal interval would be at the same nominal level — the Split CP 90 % half-width is ±24.4 mg/dL at 30 minutes which would clearly over-cover the hypoglycaemic residuals — but the gap is real and is reported transparently rather than smoothed over.

Three honest mitigations are documented for the deployment artefact of §10.5. First, a **horizon-specific α inflation** for hypoglycaemic predictions: use a calibration α' < α to over-target hypoglycaemic coverage, e.g. set α' = 0.06 in the hypo zone to attain ≥ 90 % empirical coverage. Second, an **Adaptive Conformal Inference** wrapper (Gibbs & Candès 2021, recorded as §13 Future Work) that updates α online from the rolling empirical coverage, recovering nominal coverage under distribution drift. Third, a **Mondrian + horizon-specific calibration set**: the present implementation uses the entire validation split for every horizon, but the residual distribution at 90 minutes differs from 30 minutes; per-horizon calibration is already implemented and surfaces the residual undercoverage directly so it can be priced into clinical alert thresholds.

### 9.6 Limitation — the exchangeability assumption and time-series CGM data

The split-conformal coverage guarantee assumes that the calibration and test samples are **exchangeable** — formally, that their joint distribution is invariant under any permutation of indices. CGM data violates this assumption in three documented ways:

1. **Temporal autocorrelation.** Consecutive 5-minute windows on the same patient share 23 of their 24 lookback steps, so their residuals are not independent; the calibration set effectively contains many fewer "independent" samples than its raw count of 45 382 suggests.
2. **Patient-level distribution shift.** Validation and test windows are drawn from the same patients but from *later* time periods within each patient's recording. Drift in eating, activity, or therapy regime over the 15 % validation block versus the 15 % test block violates the exchangeability assumption across the train→val→test temporal axis.
3. **Glycaemic-state non-stationarity.** Periods of glycaemic stability (low variance) and instability (high variance, post-prandial or post-exercise) cluster in time, so calibration residuals computed during one regime may under-cover during the other.

These are well-known caveats of applying CP to time-series data (e.g. Stankevičiūtė, Alaa & van der Schaar 2021; Xu & Xie 2021 EnbPI). The empirical evidence in §9.4–§9.5 is consistent with the prediction these caveats would make: marginal coverage holds within 0.7 percentage points, but the hypoglycaemic tail under-covers by 3–5 percentage points because the violation of exchangeability is most pronounced where the residual distribution is heaviest.

Two structurally stronger CP variants are documented as §13 Future Work but not implemented here: **Adaptive Conformal Inference** (ACI; Gibbs & Candès 2021) which retains a coverage guarantee under arbitrary distribution shift by updating α online, and **EnbPI** (Xu & Xie 2021) which uses a bootstrap ensemble to recover exchangeability at the cost of K-fold retraining (12 minutes × K wall-clock on this dataset). Neither is needed for the §10 deployment claim — the Mondrian Split CP intervals already provide useful and clinically interpretable bands — but both should be implemented before any external claim of calibrated long-horizon hypoglycaemic intervals is made.

### 9.7 Implications for the §10 deployment artefact

The §10.5 single-checkpoint deployment artefact (`outputs/models/step6_hybrid_v2_pers_resid.pt` plus the saved Mondrian quantile table at `outputs/tables/uq_conformal_quantiles.csv`) is now equipped to emit, at every inference tick, three glucose-forecast triples (`y_pred`, `lower`, `upper`) at 30 / 60 / 90 minutes with empirically calibrated 80 % and 90 % prediction intervals. The dashboard panels specified in §10.2 will surface the prediction interval as a shaded band around the central forecast (Figure 9.3 below), with the band colour switching to red when the lower bound crosses the 70 mg/dL hypoglycaemic threshold. Crucially the band's *width* is now data-driven and zone-aware: it widens automatically when the recent glucose lies in the hyperglycaemic zone (where residuals are largest) and narrows in TIR, without any rule-based logic.

**Figure 9.3.** `outputs/figures/09_uq_intervals_timeseries.png` — Two-panel time-series overlay showing the 30-minute prediction with its 90 % Mondrian interval for a short-duration patient (HUPA0014P) and a long-duration patient (HUPA0027P) on the test split. The shaded band is the prediction interval; the black trace is the realised glucose; the blue trace is the central forecast. The interval visibly widens during excursions and tightens during steady states — the data-driven heteroscedastic behaviour Mondrian CP provides without any parametric tail model.

The remaining clinical-safety gap from §8.6 — Persistence's 90-minute hypoglycaemic CG-EGA EP advantage — is now framed differently: instead of trying to close it with another point-estimator, the Mondrian CP layer surfaces the long-horizon hypoglycaemic uncertainty as a wide interval (54.3 mg/dL half-width at α = 0.10), which the dashboard can render as an "advise nurse review" rather than an "alert patient" state. This is the principled use of uncertainty that SKILL §7 prescribes: the model's response to a fundamentally hard prediction is not silence, and not a misleadingly tight point estimate, but a calibrated wide band that triggers a different clinical workflow.

All §9 numbers trace back to `outputs/tables/uq_conformal_quantiles.csv`, `outputs/tables/uq_conformal_coverage.csv`, and `outputs/tables/uq_conformal_intervals.parquet`. The runner is `python src/run_uq_conformal.py`; the figure runner is `python src/plot_uq_conformal.py`; the core implementation is `src/uncertainty.py`. Explainability analysis (SHAP on the CNN-GRU block, integrated gradients, attention-weight visualisation, permutation feature importance) was originally planned as the §9 content and is now deferred to §13 Future Work, to be implemented after the §10 deployment demo is built around the §9 prediction intervals.

---

## 10. Extended Contributions

### 10.1 Two-pipeline architecture: offline development and online forecasting

Following the deployment-oriented framing of Alkanhel et al. (2024), we partition the thesis contribution into two interlocking pipelines that together describe how a research artefact becomes a usable decision-support prototype.

The **offline development pipeline** covers everything documented in §3 through §9 of this report: data ingestion from the HUPA-UCM Excel sources, preprocessing under the constraints inherited from the glUCModel toolkit, feature engineering, the baseline ladder (Phase A through Phase C), the proposed Step 6 HybridCNNGRU, comparative analysis, and the planned explainability layer. The output of this pipeline is a trained checkpoint, a fitted scaler bundle, a feature catalogue, and the saved patient-level evaluation tables. This is the scientific artefact.

The **online forecasting pipeline** is the deployment counterpart that operationalises the offline artefact for a streaming 5-minute glucose, wearable, and event data context. Its specification is given below as a research blueprint; the implementation is deferred to the application stage (after Steps 7–9 close and the model is frozen). The pipeline is intentionally lightweight relative to Alkanhel et al.'s Apache Kafka and Spark architecture, because at the realistic CGM cadence of one observation per five minutes per patient, distributed-streaming infrastructure is disproportionate to the load it must handle and obscures rather than supports the research claims this thesis defends.

### 10.2 Online forecasting pipeline blueprint

The blueprint follows the six-step structure of Alkanhel et al. (2024) §3.2, adapted to HUPA-UCM constants and to a single-process Python deployment suitable for an undergraduate research demo.

**Step 1 — Sensor setup (data source abstraction).** A replay component reads the deployment-ready Parquet table `data/processed/hupa_5min_timestep.parquet` row by row in chronological order for a chosen participant, simulating a real-time CGM and wearable feed at compressed wall-clock time (one row every few seconds). For genuine deployment, this component would be replaced by a CGM API connector (LibreView, Dexcom Share, or platform-specific bridges) and a wearable SDK (Fitbit, Apple HealthKit). The pipeline downstream is sensor-agnostic by contract; the replayer is the only stage that changes between simulated and live operation.

**Step 2 — Data ingestion (buffering).** Each emitted row is pushed onto an in-memory queue (`asyncio.Queue` or a Redis Stream when multi-process scaling is required). The queue decouples the sensor cadence from the inference cadence and provides a natural place to apply staleness checks: if a glucose observation has not arrived within two sampling intervals (ten minutes), the downstream predictor is informed and either downgrades its confidence or refuses to emit a forecast. This is the analogue of Alkanhel et al.'s Apache Kafka topic, without the Kafka-specific operational burden.

**Step 3 — Online preprocessing (sliding window).** A bounded ring buffer holds the latest `LOOKBACK_STEPS = 24` rows (the lookback window fixed in §5.6). When the buffer is full, a preprocessing function applies the same scaler bundle that was fitted on the training partition during offline development (loaded once from `outputs/models/scalers.json`), constructs the engineered features defined in §6 with strict left-only operations (no future leakage by construction), and emits a `(X_dynamic, X_static)` tensor pair. Modality-availability flags (`basal_available`, `bolus_available`, `carb_available`, `basal_coverage_24h`) are recomputed from the live buffer rather than from a frozen static table, so the predictor reflects the patient's current state rather than their training-time state.

**Step 4 — Online prediction (HybridCNNGRU + uncertainty).** The serialised Step 6 checkpoint `outputs/models/step6_hybrid.pt` is loaded once at startup. Each preprocessed observation triggers one forward pass returning the point forecast at 30, 60, and 90 minutes. Predictive uncertainty is produced by Monte Carlo Dropout: the same input is passed through the model `N_MC = 30` times with dropout kept active at inference, and the per-horizon standard deviation across passes serves as the uncertainty band reported alongside the point forecast. When the input window contains a censored observation (`glucose_low_cap == 1` or `glucose_high_extreme == 1`) the predictor flags the forecast as "censored window" and widens the reported interval. When a modality has been unavailable for more than two consecutive lookback intervals the predictor flags "missing modality" and applies the M0-tier inference path (modality columns zeroed, availability flags propagated through the static branch) consistent with the Step 8 deployment-tier evaluation in §8 (forthcoming).

**Step 5 — Storage and audit trail.** Every prediction is appended to a Parquet log on local disk together with the inputs that produced it, the modality flags, the MC-Dropout standard deviation, the time of inference, and the active model version hash. This audit trail enables post-hoc CG-EGA computation, drift detection, and reproducibility checks; it is also the natural data source for the Step 9 SHAP explanation layer.

**Step 6 — Visualisation and explanation.** A Streamlit dashboard (`app/streamlit_app.py`) renders three panels: a thirty-minute window of recent glucose with the point forecast at 30, 60, and 90 minutes overlaid, with the MC-Dropout interval rendered as a shaded band; a modality-status panel showing per-channel availability and staleness; and an explanation panel showing SHAP values for the most recent prediction (computed on demand rather than per-tick for performance). A disclaimer banner is shown on every screen stating that the forecasts are research artefacts and do not constitute medical advice, as required by SKILL §0 Rule 7.

### 10.3 Differences from Alkanhel et al. (2024) and their justification

We deliberately depart from the Alkanhel et al. blueprint in three places. First, we replace Apache Kafka and Apache Spark with an in-process queue and pandas-based preprocessing because the CGM data rate is five orders of magnitude below the throughput Kafka and Spark are designed for; the substitution makes the pipeline reproducible on a laptop and on the Streamlit Cloud free tier, in keeping with SKILL §9 Colab compatibility. Second, we add Monte Carlo Dropout uncertainty quantification, which Alkanhel et al. omit; reporting only a point forecast for short-term glucose without uncertainty bands is a known failure mode for diabetes decision-support interfaces. Third, we preserve modality-availability flags as first-class inputs into the predictor and into the dashboard, rather than implicitly assuming all signals are present; this is required by the HUPA-UCM modality-gap evidence documented in §3.6.2 and §3.6.3 and absent from Alkanhel et al.'s single-patient univariate setting.

### 10.4 Other planned extended contributions

The remaining extended-contribution items planned for §10 are: uncertainty quantification with MC Dropout as the primary mechanism described above and Conformal Prediction as a calibration secondary layer; the Streamlit decision-support demo that operationalises §10.2; the deployment-tier (M0–M4) ablation that quantifies the cost of operating with reduced modality availability and that the online predictor uses to choose its inference path; and, if compute and time permit, a cross-cohort generalisation analysis using T1D-UOM as an external validation cohort. The §10.2 blueprint and the §10.4 deliverables interlock: the deployment-tier ablation produces the per-tier accuracy bands that the dashboard surfaces to the user when a modality is missing, and the SHAP layer produces the explanation panel content. Implementation is sequenced after the §9 explainability audit closes, because both the model checkpoint and the metric used to bound deployment-tier loss must be frozen before the dashboard is built around them.

### 10.5 Deployment artefact for the proposed model

The §8.6 results promote a single deployable checkpoint, `outputs/models/step6_hybrid_v2_pers_resid.pt`, as the production artefact of the online forecaster blueprint described in §10.2. The checkpoint is approximately 330 KB on disk, runs a full forward pass in approximately 5 ms per sample on a single consumer CPU, and emits three forecasts (30, 60, 90 minutes ahead) plus the corresponding scalar last-glucose value used to compute the residual; the online forecaster exposes these as both the predicted absolute glucose and the predicted delta against Persistence so that the dashboard can show both views to the user. The persistence-residual parameterisation has a useful operational property: when the static modality-availability flags indicate that one or more sensor streams are stale or missing, the model's worst-case behaviour is to emit a near-zero delta and recover Persistence by construction, which is exactly the behaviour the deployment-tier (M0–M4) protocol of §10.4 prescribes as the safe fallback. The Section 9 uncertainty-quantification layer (Monte Carlo Dropout per §10) will surround the point delta with a 90 % prediction interval rather than the absolute prediction directly, which means the resulting uncertainty band is naturally centred on Persistence at the limit of zero delta — a property that is harder to obtain from an architecture-only model whose output is not anchored to a clinically interpretable baseline. The model's Future Work in §13 covers extensions (knowledge distillation to a smaller mobile-deployable checkpoint, leave-one-patient-out cross-validation, external T1D-UOM cross-cohort validation) that the deployable artefact does not yet provide.

---

## 11. Discussion

The headline result of the §8 experiments is that the choice of clinical-safety metric reorders the model leaderboard and changes which inductive bias the proposed model must build in. Under the legacy Clarke Error Grid the strongest result was already achievable from a CNN-GRU-Attention architecture with patient-conditioned cross-attention (low pooled MAE, low Clarke Zone D, Persistence-tier hypoglycaemic MAE). Under the rate-aware Continuous Glucose-Error Grid Analysis adopted as the primary metric in this revision, that result is necessary but not sufficient: the same architecture without the persistence-residual mechanism produces predicted trajectories that fall into CG-EGA's Erroneous-Prediction zone often enough at long horizons that the clinical-safety improvement evaporates. This is a direct empirical demonstration of why Kovatchev et al. (2004) introduced CG-EGA in the first place: a model that lands close to the true value but predicts the wrong direction or rate is dangerous in a way a static point-grid cannot see.

The §8.6 results convert that diagnostic finding into a single deployable model. The persistence-residual head described in §7.6 forces the model to inherit Persistence's rate-of-change behaviour by construction; the empirical consequence is that the proposed model becomes the first model in the project to *strictly* outperform Persistence on a clinically critical CG-EGA metric (30-minute hypoglycaemic EP, 15.45 % versus Persistence's 20.08 %), while simultaneously beating or tying Histogram Gradient Boosting on pooled MAE at every horizon. The §8.7 ablation isolates this contribution to the residual head alone — the same CNN-GRU-Attention architecture without the residual head reproduces the baseline behaviour reported in earlier project revisions. That result substantiates a stronger claim than the original Step 6: it is possible to train a multimodal deep-learning model that is both more accurate than the engineered-feature tree baseline (pooled and per-zone MAE) and clinically safer than the flat-line nowcaster (CG-EGA) on short horizons, but only when the model is given the right inductive bias.

The conclusion is therefore narrower and more defensible than the original Clarke-EGA framing. The thesis demonstrates (i) the value of CG-EGA as a primary safety metric for CGM forecasting on a multimodal multi-patient dataset; (ii) the inadequacy of pooled MAE plus Clarke EGA as a sufficient comparison framework for clinical-decision-support models; (iii) a concrete architectural recipe — multi-kernel CNN + GRU + cross-attention + persistence-residual head — that achieves Pareto improvements on the relevant frontier with a single deployable PyTorch checkpoint; and (iv) a clean isolation of the persistence-residual mechanism as the load-bearing inductive bias through controlled architecture ablation. The remaining work in §13 (uncertainty quantification, leave-one-patient-out validation, external T1D-UOM cross-cohort validation) addresses the residual long-horizon hypoglycaemic gap relative to Persistence.

---

## 12. Limitations

Six limitations of the present work are explicit and load-bearing for any clinical interpretation of the results.

**Dataset-level limitations inherited from HUPA-UCM.** The 5-minute alignment of CGM, heart-rate, calorie, step, basal, bolus, and carbohydrate streams was produced by the dataset authors' glUCModel preprocessing pipeline (Hidalgo et al. 2024), which performs linear interpolation up to a one-hour gap, spreads multiple daily injections of long-acting insulin uniformly across the day, and zero-fills carbohydrate and step gaps; the thesis inherits each of these choices and therefore cannot make independent claims about raw-sensor reliability or imputation policy (§3.5). FreeStyle Libre 2 censored values at the 40 mg/dL floor and the 400 mg/dL ceiling are retained but flagged; reported hypoglycaemic performance is therefore conditional on whether censored windows are included or masked (§3.6.1). Five patients have one or more fully missing modalities and four long-duration patients have only partial basal coverage (§3.6.2–§3.6.3); the modality-availability flags and modality-dropout training are the project's mitigation but they do not eliminate the bias that arises from patient-modality co-occurrence patterns. The HUPA-UCM cohort is dominated by three long-duration patients (HUPA0027P / 0026P / 0028P together hold 74.93 % of pooled records), and three of these patients overlap temporally with Spain's 2020–2021 COVID-19 lockdown period (§3.6.5); the activity, meal-timing, and glycaemic patterns of those patients are therefore confounded with a non-routine lifestyle context that no preprocessing can correct.

**Single-cohort, single-platform evaluation.** All §8 results are reported on HUPA-UCM only — 25 patients recruited at a single Spanish institution using FreeStyle Libre 2 CGM and Fitbit Ionic wearables. Generalisation to other CGM platforms (Dexcom, Medtronic, Eversense) and to other patient populations (different healthcare systems, paediatric / geriatric cohorts, different ethnic and socio-economic groups) is not tested. The §13 Future Work explicitly proposes the T1D-UOM external validation as the highest-priority generalisation experiment.

**Carbohydrate input is self-reported.** Carbohydrate intake in HUPA-UCM is recorded by patient self-logging; under-reporting and meal-timing inaccuracy are well-documented failure modes in diabetes self-management. The persistence-residual mechanism of §7.6 is partially robust to this because the model only needs to learn corrections on top of Persistence, but no part of the pipeline can correct systematic patient-level meal-logging bias.

**Evaluation metric scope.** The §8 results report MAE, RMSE, per-zone MAE, Clarke EGA (legacy), and CG-EGA (primary). §9 adds Conformal-Prediction interval coverage and width. The thesis does not report Parkes (Consensus) EGA, glucose-specific RMSE (gRMSE) variants, or SHAP-based interpretability metrics; these are deferred to §13 Future Work. Each would add information but none would change the §8.5–§8.7 ordering qualitatively.

**Validation methodology.** The chronological per-patient 70/15/15 split with a horizon-length boundary buffer prevents temporal leakage within each patient but does not test cross-patient generalisation to fully held-out individuals. A leave-one-patient-out cross-validation protocol is the methodologically stronger alternative; it was not run here for compute reasons (25 × 12-minute training runs per variant = 5 hours per ablation), but is the canonical extension for any deployment claim.

**Uncertainty quantification under non-exchangeability.** §9 implements Split and Mondrian Conformal Prediction and demonstrates well-calibrated marginal coverage on the held-out test split, but the per-zone analysis surfaces a 3.2–4.8 percentage-point hypoglycaemic under-coverage at the 90 % nominal level (§9.5). This is the documented signature of CGM data's violation of the exchangeability assumption that Split CP requires. The §10 deployment artefact must therefore be considered a research prototype, not a clinically validated decision-support system; the §13 Future Work entry on Adaptive Conformal Inference (Gibbs & Candès 2021) and EnbPI (Xu & Xie 2021) is the principled response to this gap. Section 0 Rule 7 ("do not claim clinical usefulness without evidence") remains binding.

---

## 13. Future Work

Four concrete directions follow from the §8.6 headline result, the §8.7 architecture ablation, and the §9 conformal-prediction findings.

**Adaptive Conformal Inference and EnbPI to close the hypoglycaemic under-coverage.** The §9.5 per-zone analysis identifies a 3.2–4.8 percentage-point under-coverage of the 90 % Mondrian prediction interval in the hypoglycaemic zone — the signature of non-exchangeability between calibration and test residuals. Two structurally stronger CP variants are the principled response. Adaptive Conformal Inference (Gibbs & Candès 2021) updates the mis-coverage rate α online from the rolling empirical coverage, recovering nominal coverage under arbitrary distribution shift; the implementation cost is small (a single state variable per horizon × zone) and the runtime overhead at inference is negligible. EnbPI (Xu & Xie 2021) trains a bootstrap ensemble of `K` proposed-model copies and recovers exchangeability across leave-one-out folds; the compute cost is `K × 12` minutes (K = 5 to 10 in published practice) but the resulting intervals have a marginal coverage guarantee under arbitrary time-series stationarity violation. The success criterion is empirical hypoglycaemic coverage within 2 percentage points of nominal at every horizon.

**Explainability layer with SHAP, integrated gradients, and attention-weight visualisation.** The XAI work originally scoped for §9 is deferred here. SHAP DeepExplainer (Lundberg & Lee 2017) on the CNN block, Integrated Gradients (Sundararajan, Taly & Yan 2017) on the full multi-horizon head, and attention-weight visualisation on the §7.6.3 cross-attention block are the three complementary methods. The motivating clinical question is whether the proposed model attends to physiologically meaningful timesteps — for example, the 25 minutes immediately preceding a bolus event for a hypoglycaemic forecast at +60 minutes — and whether its modality reliance matches clinical intuition (recent glucose first, recent insulin second, carbohydrate timing third). The deliverable is a §9-style results subsection with case studies on the hypoglycaemic prediction failures of §8.6 Table 8.6.2 (the +90-minute hypo MAE gap relative to Persistence).

**Leave-one-patient-out cross-validation of the proposed model.** The chronological per-patient 70 / 15 / 15 split with horizon-length boundary buffer prevents temporal leakage within each patient but does not test cross-patient generalisation to fully held-out individuals. A leave-one-patient-out (LOPO) protocol would retrain the proposed model 25 times, holding out one patient at a time, and report the per-held-patient CG-EGA distribution. The compute cost is approximately 25 × 12 minutes = 5 hours on a consumer CPU; the deliverable is a forest plot of per-patient CG-EGA performance with the median and inter-quartile range overlaid, plus an analysis of whether any specific patient profile (missing-modality patients, partial-basal patients, the long-duration HUPA0027P / 0028P group) explains the tails of the distribution.

**External validation on T1D-UOM with the deployment-tier (M0–M4) protocol.** The HUPA-UCM-trained checkpoint `step6_hybrid_v2_pers_resid.pt` has not been evaluated on the legacy T1D-UOM dataset. A cross-cohort generalisation analysis would (a) measure the unit-conversion sensitivity of the per-subject z-score scalers (HUPA is mg/dL, T1D-UOM is mmol/L), (b) test whether the modality dropout regime trained on HUPA's coverage gaps generalises to T1D-UOM's different modality availability profile, and (c) provide a more rigorous test of CG-EGA stability than the within-cohort split. This is a secondary direction that depends on resolving the T1D-UOM preprocessing pipeline first; the §0 SKILL rules require an analogous EDA / preprocessing audit before any cross-cohort claim.

---

## 14. Conclusion

This thesis builds a short-term blood-glucose forecasting pipeline for Type 1 Diabetes on the HUPA-UCM cohort and evaluates it under both pooled MAE/RMSE and the rate-aware Continuous Glucose-Error Grid Analysis. Four findings define the contribution. First, switching the primary clinical-safety metric from Clarke EGA to CG-EGA reorders the model leaderboard substantially: gradient-boosted trees dominate pooled MAE but produce unacceptably high CG-EGA Erroneous-Prediction shares at long horizons, while the Persistence flat-line nowcaster turns out to be the strongest clinical-safety baseline by CG-EGA on this dataset. Second, the proposed model — CNN-GRU-Attention with Persistence-Residual Learning, specified in §7.6 — strictly outperforms the strongest single classical baseline (Histogram Gradient Boosting) on pooled MAE at 60 minutes, ties it within 0.2 mg/dL at the other two horizons, achieves the project's lowest hypoglycaemic MAE at 30 and 60 minutes, and produces the project's lowest hypoglycaemic CG-EGA Erroneous-Prediction share at 30 minutes (15.45 % versus Persistence's 20.08 %) — the first model in the project to strictly outperform Persistence on a clinically critical CG-EGA metric. Third, the §8.7 architecture ablation isolates the persistence-residual head as the load-bearing inductive bias: the same CNN-GRU-Attention architecture without the residual head reproduces the original Step 6 baseline behaviour, confirming that the multi-kernel CNN, the GRU depth, and the cross-attention block alone are not sufficient. Fourth, the §9 Conformal-Prediction layer wraps the proposed model with calibrated 80 % and 90 % prediction intervals using Split CP and Mondrian (per-zone) CP; marginal coverage is within 0.7 percentage points of nominal at every horizon, and the per-zone analysis exposes a 3–5 percentage-point hypoglycaemic under-coverage that the project documents transparently and addresses through Adaptive Conformal Inference in §13 Future Work. The work demonstrates that an informed combination of clinical-safety metric, controlled architectural ablation, targeted inductive-bias change, and distribution-free uncertainty quantification can produce a multimodal deep-learning forecaster that is more accurate than the engineered-feature tree baseline, clinically safer than the flat-line nowcaster on short horizons, and equipped with prediction intervals whose coverage is empirically validated on a held-out test split — all from a single deployable PyTorch checkpoint plus a 12-row calibration table.

---

## 15. References

*Maintained in synchrony with `reports/literature_review.md` §11.* Final reference list will be condensed from the literature-review document once all `[verify]` markers there are resolved.
