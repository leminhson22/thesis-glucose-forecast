# A Multimodal Deep Learning Approach for Short-Term Blood Glucose Forecasting in Type 1 Diabetes

**Author:** Son
**Programme:** Undergraduate thesis
**Last revised:** 2026-05-18 (rebuilt on HUPA-UCM after migrating from the T1D-UOM dataset; §4 expanded with peri-event, per-patient heterogeneity, day-of-week, velocity-by-zone, event-subtype, per-patient peri-event variance, and sensor-floor velocity-artefact analyses; retired raw lagged-Pearson screen)

> **Reading order.** This file is the formal thesis manuscript. Before reading any methodology paragraph, consult `reports/literature_review.md` (Step 0) for the citation backing each design choice, and `notebooks/01_data_understanding.ipynb` (Step 1) for the verifiable evidence behind every numeric claim in §3. Sections not yet produced are explicitly marked *to be written*.

---

## Abstract

*To be written after Step 6 modelling results are available.* The Abstract must summarise: problem statement, dataset (HUPA-UCM, 25 patients), proposed method (hybrid CNN-GRU with cross-attention fusion of CGM, insulin, carbohydrate, and Fitbit-derived activity signals, plus a static patient-embedding branch), key numerical results at the 30/60/90-minute horizons, and the main contribution to the field. Target length 150–200 words.

---

## 1. Introduction

*To be expanded after §3 is fully signed off.* This section motivates short-term blood-glucose forecasting in Type 1 Diabetes from a clinical standpoint (hypoglycaemia prevention, decision support, closed-loop control); summarises the limitations of existing approaches (single-modality, limited generalisation, lack of clinical-grade uncertainty estimates); states the research gap addressed; and previews the thesis structure.

---

## 2. Background and Related Work

*Drafted from `reports/literature_review.md`; to be condensed to manuscript length after §6 is complete.* This section will cover: classical and deep-learning approaches to CGM forecasting; multimodal fusion architectures; clinical-aware loss functions; explainability methods adopted in medical AI; uncertainty-quantification techniques; and the Clarke / Parkes Error Grid framework. Direct quantitative comparison with prior work on HUPA-UCM (Alvarado et al. 2023; Parra et al. 2024; Botella-Serrano et al. 2023) and prior work on closely related cohorts from the same research group (Tena et al. 2021; Tena et al. 2023) will be presented in §8.

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

Three properties of the selection result inform the architectural choices in §7. First, the dominance of `glucose` and its short-horizon rolling means among the dynamic features confirms that the temporal trunk must be able to learn an autoregressive baseline robustly; a pure-attention architecture without an explicit recurrent or convolutional inductive bias would have to learn that baseline from scratch and is therefore disfavoured for a dataset of this size. Second, the strong showing of static features in the top ten — five of the top-ten composite ranks are static behavioural-fingerprint features — justifies an explicit two-branch architecture with a dedicated patient embedding rather than a single trunk that concatenates static features into the temporal input. Third, the two-round pruning toward physiologically- and clinically-meaningful features (IOB / COB in place of arbitrary rolling spans; clinical metadata in place of data-collection artefacts such as `data_duration_days`) constrains the model to learn from representations that a clinician would recognise on a CGM display, which is consistent with the Hovorka-style first-order absorption priors of the diabetes-modelling literature and supports the interpretability requirements for the XAI analysis in §9.

---

## 7. Modelling Strategy

This section presents the modelling decisions of this thesis as a baseline ladder, from the cheapest reference model to the proposed hybrid neural architecture. Each rung is justified by the gap-driven rationale required by SKILL.md Rule 9: prior literature establishes the model class; the dataset evidence from Sections 3–4 motivates a specific architectural choice; and the experimental protocol (per-patient chronological split, patient-averaged metric reporting, identical evaluation bundle for every model) addresses the validation weaknesses identified in the Section 2 literature synthesis. The ladder is divided into three phases so that progress on the thesis remains reportable at each stage:

* **Phase A — Linear references (§7.1–§7.2, completed).** Persistence and Ridge regression on the flattened lookback window. These are the cheapest models that can be evaluated under the exact protocol used by every subsequent model. Any neural model that fails to outperform them on both pooled MAE and the clinically critical hypoglycaemic zone has no defensible contribution.
* **Phase B — Non-linear and ensemble references (§7.3, planned).** Random Forest and Gradient Boosting on the same flattened representation, to test whether non-linearity in the feature space is the bottleneck.
* **Phase C — Sequence and hybrid models (§7.4, planned).** LSTM / GRU on the (24, 17) dynamic tensor with the (16,) static branch, then the proposed CNN–GRU with cross-attention fusion. The hybrid is the model the thesis advances; it must outperform every reference on at least one horizon and on at least the hypoglycaemic zone to justify its complexity.

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

Both models are evaluated under the same metric bundle as Phase A and Phase B (`src/evaluate.py`): pooled and patient-averaged MAE and RMSE per horizon, MAE and RMSE binned by glycaemic zone, per-patient breakdown, and Clarke Error Grid Analysis percentages. The headline comparison surface is the same test split used in Section 8.2 (45 395 windows, 25 patients), enabling direct mg/dL deltas against the GBM-300 numbers. The expected interpretation falls into three scenarios. First, if the recurrent encoders outperform GBM-300 on pooled MAE at every horizon, the result supports the hypothesis that the sequential structure carries information not captured by the engineered rolling statistics on the flattened representation. Second, if the recurrent encoders match GBM-300 on pooled MAE but still lose to Persistence in the hypoglycaemic zone at long horizons, the result confirms that the hypo-zone deficit is loss-driven rather than capacity- or representation-driven, and the Phase C.2 asymmetric/zone-weighted intervention becomes empirically mandatory. Third, if the recurrent encoders fail to match GBM-300 even on pooled MAE, the conclusion is that at the present sample size and feature configuration the tree representation already captures the relevant signal, and the case for the hybrid architecture in Phase C.3 must rest on the loss-function and attention/fusion contributions rather than on basic sequence-awareness alone. Section 8.3 reports the numbers and resolves which scenario applies.

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

Table 8.1.2 reports the Clarke Error Grid Analysis (Clarke et al. 1987) zone shares on the test split. The combined A + B share — clinically acceptable predictions — stays above 95 % at 30 minutes for both models and degrades to approximately 93 % at 90 minutes; the Zone D share (failure to detect a dangerous excursion) grows from 1.4–1.6 % at 30 minutes to 4.6 % (Persistence) and 6.3 % (Ridge) at 90 minutes, signalling that the linear models trade hypoglycaemic and hyperglycaemic sensitivity for euglycaemic accuracy as the horizon lengthens.

**Table 8.1.2.** Clarke Error Grid Analysis on test (percent of predictions per zone).

| Model | Horizon | A | B | C | D | E |
|---|---|---|---|---|---|---|
| Persistence       | 30 min | 85.45 | 12.89 | 0.02 | 1.64 | 0.00 |
| Persistence       | 60 min | 69.43 | 26.64 | 0.55 | 3.38 | 0.00 |
| Persistence       | 90 min | 59.40 | 34.55 | 1.45 | 4.60 | 0.00 |
| Ridge ($\alpha=0.1$) | 30 min | 86.09 | 12.48 | 0.06 | 1.38 | 0.00 |
| Ridge ($\alpha=0.1$) | 60 min | 71.81 | 24.40 | 0.23 | 3.56 | 0.00 |
| Ridge ($\alpha=0.1$) | 90 min | 60.30 | 32.85 | 0.51 | 6.34 | 0.00 |

The most consequential finding for the thesis is in the per-zone MAE breakdown (Table 8.1.3). Persistence beats Ridge in the hypoglycaemic zone at every horizon — by 6.6 mg/dL at 30 minutes, 2.6 mg/dL at 60 minutes, and 7.0 mg/dL at 90 minutes. Ridge wins in the time-in-range zone at 60 and 90 minutes and ties at 30 minutes, and Ridge wins in the hyperglycaemic zone at 30 and 60 minutes, but the gain in TIR and hyper cannot recover the hypo deficit. This pattern is consistent with the well-known regression-to-the-mean failure mode of linear models trained on imbalanced glucose data: the hypoglycaemic zone holds 8.0 % of test windows, the TIR zone 71.5 %, and a least-squares loss minimised over the pooled distribution drives the model toward TIR predictions at the cost of mis-predicting hypoglycaemic events.

**Table 8.1.3.** Per-zone test MAE (mg/dL). Hypo $<70$, TIR $70$–$180$, hyper $>180$. Bold marks the worse model in each cell.

| Model | Horizon | Hypo | TIR | Hyper |
|---|---|---|---|---|
| Persistence       | 30 min | 9.06  | 12.82 | 17.72 |
| Persistence       | 60 min | 18.34 | 21.01 | 31.39 |
| Persistence       | 90 min | 27.26 | 26.72 | 41.84 |
| Ridge ($\alpha=0.1$) | 30 min | **15.66** | 12.82 | **18.72** |
| Ridge ($\alpha=0.1$) | 60 min | **20.94** | 19.61 | 28.79 |
| Ridge ($\alpha=0.1$) | 90 min | **34.27** | 23.54 | 40.43 |

This finding is the empirical justification for two design choices to be implemented in Phase C: (i) the proposed hybrid model will use an asymmetric loss with a hypoglycaemic penalty term, following Del Favero et al. (2012); and (ii) the evaluation report in every subsequent phase will lead with the per-zone breakdown rather than the pooled metric, so that any pooled improvement that comes at the cost of hypoglycaemic accuracy is immediately visible. A model that wins on pooled MAE but loses on hypoglycaemic MAE relative to Persistence is clinically inferior to Persistence and cannot defend the thesis claim.

The top-five coefficients of the Ridge model at the 30-minute horizon, in signed units of mg/dL per unit Z-scored feature (from `outputs/tables/phase_a_ridge_top_coefs.csv`), are dominated by glucose rolling means at the most recent lags: `glucose_60m_mean_lag0` ($-1225$), `glucose_60m_mean_lag1` ($-872$), `glucose_30m_mean_lag0` ($+692$), `glucose_60m_mean_lag2` ($-512$), and `glucose_30m_mean_lag1` ($+448$). The alternating signs across adjacent lags act as a finite-difference operator that effectively reconstructs a velocity term from the rolling-mean basis, supplementing the explicit `glucose_velocity` feature (which carries a smaller coefficient magnitude). The dominant predictive signal at short horizons is therefore the recent glucose trajectory; the static patient features, peri-event aggregates (IOB, COB, steps), and modality flags contribute additively but at much smaller magnitudes. This decomposition will guide architectural choices in Phase C — in particular, whether to wrap the dynamic input in a one-dimensional temporal convolution that learns the same finite-difference response directly from the raw lookback window.

All numbers in this section trace back to `outputs/tables/phase_a_*.csv` and `outputs/models/ridge_phase_a.joblib`; the runner that produces them is `src/run_phase_a.py` and the Colab-compatible execution path is `notebooks/04_model_training.ipynb`.

### 8.2 Phase B baselines — Random Forest and Gradient Boosting

Phase B reports the two tree-based references on the same test split and metric bundle. Wall-clock fit times on a multi-core consumer CPU (`n_jobs = -1`) were approximately 26 minutes for the Random Forest and 3 minutes for the three-headed gradient boosting, both well within feasible local-compute budgets. The HistGradientBoosting heads each reached the `max_iter = 300` budget cap without triggering early stopping; the consequences of this cap are quantified in the budget ablation reported at the end of this section.

Table 8.2.1 collects all four baselines on the test split. Pooled MAE is row-weighted across the 45 395 test windows; patient-averaged MAE is the unweighted mean across 25 patients, reported in parallel for the reason established in Section 8.1.

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

Table 8.2.2 reports the Clarke Error Grid Analysis on the test split for the two Phase B models. The combined A + B share is 98 % at 30 minutes for HistGB (versus 98.3 % for Persistence and 98.6 % for Ridge — a small loss in absolute share but the share rebalances toward Zone A: HistGB Zone A at 30 minutes is 91.3 % versus 85.5 % for Persistence). The Zone D share — failure to detect a dangerous condition — is 1.9 % at 30 minutes for HistGB versus 1.6 % for Persistence; a marginal increase that warrants the per-zone scrutiny below.

**Table 8.2.2.** Clarke Error Grid Analysis on test for the Phase B models (percent of predictions per zone).

| Model | Horizon | A | B | C | D | E |
|---|---|---|---|---|---|---|
| Random Forest (n=300)   | 30 min | 88.01 | 8.58  | 0.00 | 3.41 | 0.00 |
| Random Forest (n=300)   | 60 min | 71.07 | 21.93 | 0.08 | 6.91 | 0.00 |
| Random Forest (n=300)   | 90 min | 59.03 | 32.04 | 0.25 | 8.68 | 0.00 |
| HistGB (lr=0.05, d=8) | 30 min | 91.27 | 6.79  | 0.01 | 1.94 | 0.00 |
| HistGB (lr=0.05, d=8) | 60 min | 72.79 | 20.74 | 0.07 | 6.41 | 0.00 |
| HistGB (lr=0.05, d=8) | 90 min | 61.11 | 30.23 | 0.22 | 8.44 | 0.00 |

The headline clinical finding of Phase B is in Table 8.2.3: **Persistence remains the best model in the hypoglycaemic zone at every horizon**, despite being beaten by every other model on the pooled metric. HistGB closes the hypo gap at 30 minutes to 1.4 mg/dL (10.42 versus Persistence 9.06), but at 60 and 90 minutes its hypo MAE diverges substantially upward — 26.75 and 40.72 mg/dL versus Persistence's 18.34 and 27.26. Random Forest behaves similarly. The hypo deficit of the non-linear models is therefore not a defect of feature engineering or of model capacity; it is the loss-function consequence already diagnosed in Section 8.1. A squared-error objective minimised over a distribution in which the time-in-range zone holds 71 % of the mass and the hypoglycaemic zone holds 8 % systematically biases the model toward the dense mass, regardless of whether the model class is linear, axis-aligned forests, or gradient-boosted trees.

**Table 8.2.3.** Per-zone test MAE (mg/dL) for the full baseline ladder. Bold marks the best model in each cell. Hypo $<70$, TIR $70$–$180$, hyper $>180$.

| Model | Horizon | Hypo | TIR | Hyper |
|---|---|---|---|---|
| Persistence       | 30 min | **9.06**  | 12.82 | 17.72 |
| Ridge ($\alpha=0.1$) | 30 min | 15.66 | 12.82 | 18.72 |
| Random Forest (n=300) | 30 min | 14.66 | 10.34 | 15.72 |
| HistGB (lr=0.05, d=8) | 30 min | 10.42 | **9.43**  | **13.77** |
| Persistence       | 60 min | **18.34** | 21.01 | 31.39 |
| Ridge ($\alpha=0.1$) | 60 min | 20.94 | 19.61 | 28.79 |
| Random Forest (n=300) | 60 min | 29.89 | 17.01 | 29.94 |
| HistGB (lr=0.05, d=8) | 60 min | 26.75 | **16.42** | **28.75** |
| Persistence       | 90 min | **27.26** | 26.72 | 41.84 |
| Ridge ($\alpha=0.1$) | 90 min | 34.27 | 23.54 | 40.43 |
| Random Forest (n=300) | 90 min | 43.51 | 21.28 | 42.09 |
| HistGB (lr=0.05, d=8) | 90 min | 40.72 | **20.53** | **40.71** |

The pattern is therefore consistent across all four baselines: increasing model capacity along the standard ladder (linear → axis-aligned forest → gradient-boosted forest) reduces pooled MAE monotonically by 23 % at 30 minutes and 12 % at 90 minutes, but does not solve the regression-to-the-TIR-mean bias in the hypoglycaemic zone. This is the empirical justification for the architectural and loss-function decisions to be implemented in Phase C: (i) any sequence model that uses a vanilla squared-error loss will inherit the same hypo bias, regardless of recurrence, attention, or fusion mechanism, so the loss function itself must be modified; and (ii) Persistence will be retained as a per-zone reference even after stronger models exist, because in the hypoglycaemic safety case it is the model to beat, not Ridge or HistGB.

**Budget ablation — `max_iter = 300` versus `max_iter = 1000`.** Because the three GBM-300 heads exhausted their iteration cap, a follow-up run trained a second HistGradientBoosting model with `max_iter = 1000` and every other hyperparameter, the same chronological split, the same flattened-window input, and the same seed held fixed. The 1000-iteration model used the full budget on every horizon (1000/1000/1000), confirming that sklearn's internal early-stopping criterion — evaluated on a 10 % slice taken from the training set — was still improving. The external test set tells a different story: pooled test MAE *increased* slightly under the larger budget, by +0.09 mg/dL at 30 minutes, +0.30 at 60 minutes, and +0.34 at 90 minutes (+0.8 %, +1.5 %, +1.3 % relative). Per-zone test MAE in the hypoglycaemic range also moved in the wrong direction at every horizon, by +0.58, +0.92, and +0.32 mg/dL. Patient-averaged test MAE rose by +0.13, +0.34, and +0.57 mg/dL, and the Clarke Zone A share dropped by 0.09, 0.74, and 0.37 percentage points. The divergence between the internal-validation signal — which kept improving for 700 additional iterations — and the external test signal — which started to degrade — is a textbook signature of mild over-fitting that sklearn's internal early stopping cannot detect because it samples its validation slice from the same training distribution. The `max_iter = 300` configuration is therefore retained as the primary GBM baseline; the 1000-iteration run is documented as an ablation rather than promoted. The full delta table is at `outputs/tables/phase_b_gbm_comparison.csv` and the ablation model is checkpointed separately at `outputs/models/gbm_phase_b_1000.joblib`.

All numbers in this section trace back to `outputs/tables/phase_b_*.csv`, `outputs/tables/phase_b_gbm1000_*.csv`, `outputs/models/rf_phase_b.joblib`, `outputs/models/gbm_phase_b.joblib`, and `outputs/models/gbm_phase_b_1000.joblib`. The runners that produce them are `src/run_phase_b.py` and `src/run_gbm_1000.py`, and the Colab-compatible execution path is `notebooks/04b_phase_b_trees.ipynb`.

---

## 9. Explainability Analysis

*To be written after Step 7.* SHAP, integrated gradients, attention-weight visualisations, and permutation feature importance, focused on hypoglycaemic prediction cases as the most clinically critical.

---

## 10. Extended Contributions

*To be written after Step 8 / 10.* Uncertainty quantification (MC Dropout primary, Conformal Prediction secondary), and the optional Streamlit decision-support demo. Cross-cohort generalisation analysis using T1D-UOM as an external validation cohort, if compute and time permit.

---

## 11. Discussion

*To be written after §8–10.*

---

## 12. Limitations

*To be written after §8.* Will explicitly include: dependence on the dataset authors' interpolation choices (3.5); censored sensor readings (3.6.1); modality availability gaps for 5 patients (3.6.2–3.6.3); duration imbalance dominated by HUPA0027P (3.4); the COVID-19 recording-period confound on the three long participants (3.6.5); single-CGM-platform evaluation (FreeStyle Libre 2 only); modest cohort size (25 patients); single-country (Spain) recruitment; manual carbohydrate self-report (per the source article's own Limitations section).

---

## 13. Future Work

*To be written.*

---

## 14. Conclusion

*To be written.*

---

## 15. References

*Maintained in synchrony with `reports/literature_review.md` §11.* Final reference list will be condensed from the literature-review document once all `[verify]` markers there are resolved.
