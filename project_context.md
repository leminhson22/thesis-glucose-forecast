---
name: project-context
description: "Son's thesis on HUPA-UCM 25-patient blood-glucose forecasting; as of 2026-05-19 Step 0 has a draft literature_review.md with known gaps and Step 1 is complete with verified artefacts; ready to start Step 2 EDA review"
metadata:
  node_type: memory
  type: project
  originSessionId: 4ba1dc89-29fd-4de6-b7b0-1c89b677ad41
---

**Thesis:** "A Multimodal Deep Learning Approach for Short-Term Blood Glucose Forecasting in Type 1 Diabetes"

**Dataset decision (2026-05-17):** Switched **from T1D-UOM to HUPA-UCM** as primary modelling dataset. T1D-UOM archived to `archive/uom_phase/`. Rationale: HUPA-UCM is pre-aligned 5-min grid, mg/dL native, 25 patients with full clinical metadata, multiple prior baselines from the same research group.

**Current dataset: HUPA-UCM** (Hidalgo et al., Data in Brief 2024, DOI 10.17632/3hbcscwz44.1)
- 25 patients, FreeStyle Libre 2 CGM + Fitbit Ionic.
- 309,392 rows total, ~1,074 patient-days, 5-min aligned grid, glucose in mg/dL (range 40-444).
- 8 columns per file: time, glucose, calories, heart_rate, steps, basal_rate, bolus_volume_delivered, carb_input.
- Treatment split: 14 CSII / 11 MDI.
- Full static metadata in `patient_data_characteristic.xlsx`.

**Critical quirks (verified against data 2026-05-19, do not re-derive):**
- **Duration imbalance:** HUPA0027P = 53.43% of rows (574 days), HUPA0026P = 13.12% (141 days), HUPA0028P = 8.37% (90 days); top-3 combined = 74.93%. Median duration = 13.3 days. Use patient-level CV, never subject-mixed split.
- **Missing-modality patients (5 fully missing, 5.89% rows):** HUPA0011 (CSII, no basal+bolus — anomaly, treat as MDI in `treatment` feature), HUPA0014 (MDI, no basal), HUPA0015 (MDI, no basal+bolus+carb), HUPA0018 (MDI, no basal+bolus+carb), HUPA0020 (MDI, no carb).
- **Partial basal coverage (4 long-duration MDI patients):** HUPA0024 (59.9%), HUPA0026 (66.1%), HUPA0027 (59.2%), HUPA0028 (40.0%). Combined with full-missing patients, modality dropout + availability flags are mandatory.
- **Glucose sensor caps:** value `==40` mg/dL accounts for 0.38% of rows globally (HUPA0002P 5.82%, HUPA0018P 4.16%, HUPA0022P 2.61%); value `>400` is 0.04%. Censored — retain with `glucose_low_cap` / `glucose_high_extreme` flags, never as ordinary observations.
- **Time index:** strictly monotonic, exactly 300-second step everywhere. Dataset already resampled by glUCModel; the thesis does not contribute on interpolation.
- **Recording span: 2018-06-13 to 2022-05-18 (~4 years, 9 recruitment waves).** HUPA0026/0027/0028 are temporally co-located with the COVID-19 lockdown period in Spain (state of alarm 14-Mar-2020 onwards) — a confound that biases lifestyle, activity, and meal-timing patterns. Documented in `report.md §3.6.5` and §12 Limitations.

**Pipeline status (as of 2026-05-19):**
- ✅ Step 0 (literature review): `reports/literature_review.md` exists with draft content. **Has known gaps** — see [[literature-review-gaps]] memory before continuing Step 0.
- ✅ Step 1 (data understanding): complete. `notebooks/01_data_understanding.ipynb` runs end-to-end; produces `data/interim/hupa_cohort_summary.csv` and `outputs/figures/01_data_understanding_overview.png`. `reports/report.md §3` (Dataset Description) is fully drafted and scrubbed for consistency with notebook outputs (BMI numbers corrected, weighting convention disambiguated, COVID confound documented, "1-hour gap" claim marked `[verify]`).
- ✅ Step 2 (EDA): two review passes. Pass 1 (2026-05-17) added peri-event/per-patient ACF/day-of-week/velocity-by-zone and corrected a mislabeled glucose-distribution figure. Pass 2 (2026-05-18) added bolus/carb subtypes (meal vs correction/solo), per-patient peri-event variance (mean±SD across patients), sensor-floor velocity artefact check, and **retired the lagged-Pearson screen entirely** (CSV + figure deleted) because it is structurally useless on sparse event streams. Key new findings: correction bolus alone drops glucose −30 mg/dL at 120 min (2× the pooled estimate); solo carb alone rises +26 mg/dL at 120 min; cross-patient SD of peri-event response is ~2× the mean magnitude → cross-attention/gated modality fusion is empirically justified (not stylistic). All artefacts under `outputs/tables/hupa_eda_*.csv` (15 tables) and `outputs/figures/02_eda_*.png` (7 figures). `report.md §4` has 9 subsections including 4.6.1 subtype + 4.6.2 per-patient + filtered velocity in 4.7. Notebook `02_eda.ipynb` has 36 cells (24 markdown + 12 code).
- ✅ Step 3 (preprocessing): complete (2026-05-18). `src/config.py` + `src/preprocessing.py` (modular, Colab-compatible, deterministic at SEED=42) + `notebooks/03_preprocessing_feature_engineering.ipynb` (16 cells). Outputs in `data/processed/`: `hupa_5min_timestep.parquet` + `hupa_5min_sequences.npz` + `hupa_static_features.csv`. Scalers in `outputs/models/scalers.json`. Splits per-patient chronological 70/15/15 with 18-row buffer at boundaries (0.29% rows dropped). Long-patient handling: deterministic adaptive-stride cap at N_TRAIN_CAP=5000 per patient on TRAIN only (val/test stride=1) — see [[long-patient-strategy]]. P11 treatment overridden to MDI in static table. End-to-end runtime: ~50s local. `report.md §5` has 10 subsections (5.1–5.10) covering inherited glUCModel pipeline + thesis preprocessing.
- ✅ Step 4 (feature engineering + selection): complete with TWO-ROUND pruning (2026-05-19). First pass added 3 features (glucose_60m_std, IOB, COB) → 59. **Round 1 (info-theoretic, drop 21)**: math-redundant (weight/height/Male/MDI), IOB/COB-redundant (raw bolus/carb + 4 rolling sums), wearable-redundant (raw calories/steps + mean_daily_steps), statistical (subject_tir_pct), weak-signal (dayofweek/glucose_high_extreme/3 dyn availability flags). **Round 2 (clinical-lens, drop 5)**: glucose_acceleration (no CGM device displays 2nd derivative), calories_30m_sum (Fitbit-derived from HR+steps, not clinical for glucose), carb_events_per_day (measures log frequency not eating), data_duration_days (pure data artifact, deployment-misleading), basal_recording_pct (correlates ~0.9 with treatment_CSII). **FINAL: 17 dynamic + 16 static = 33 features**, X_dynamic shape (159172, 24, 17), X_static (159172, 16). Feature budget per sample dropped 49.6% (841→424). Feature-selection re-run on 33: top-10 still glucose + 5/10 static derived. Artefacts: `data/processed/hupa_5min_sequences.npz`, `outputs/tables/hupa_feature_selection.csv`, `outputs/figures/04_feature_selection_{dynamic,static}.png`. Word docs: `feature_catalogue.docx` (44KB), `feature_selection_report.docx` (142KB), `bao_cao_tong_quan.*.docx` (overview). `report.md §6.5` has both rounds documented with 5+5 categories.
- ⏳ Steps 5-13: not started. Next: Step 5 baseline ladder (persistence → Ridge → RF/XGBoost → LSTM/GRU).

**Active design preferences (carry into Step 5+):**
- Feature-engineering write-ups follow Why→Taxonomy→Per-group→Scaling→Selection→Implications structure — see [[feature-engineering-writeup-structure]].
- Step 6 will implement modality dropout (30 % random branch zero-out during training) and Step 8 will report M0-M4 tier evaluation — see [[deployment-tier-strategy]].
- User favours aggressive feature pruning grounded in clinical interpretability over "keep all and ablate later". 26 features pruned across two rounds gave the final 33; further pruning candidates are exhausted.

**File state on disk (verified 2026-05-19):**
- `data/processed/hupa_5min_sequences.npz` — X_dynamic (159172, 24, 17), X_static (159172, 16), y (159172, 3). Train/val/test = 68395/45382/45395.
- `data/processed/hupa_5min_timestep.parquet` and `hupa_static_features.csv` regenerated for the 33-feature config.
- `reports/report.md` §1-§6.6 written (English). §6 follows the new Why→Taxonomy→Per-group→Scaling→Selection logic.
- `reports/bao_cao_tong_quan.docx` (833 KB) — Vietnamese overview, Phần 10 follows the same new logic.
- `reports/feature_catalogue.docx` (45 KB), `reports/feature_selection_report.docx` (146 KB) — Word references.
- `outputs/figures/04_feature_selection_{dynamic,static}.png` — refreshed for 33-feature ranking.

**Important nuance on prior HUPA work (still requires per-paper verification):**
`SKILL.md` Step 0 lists 6 papers as "HUPA-UCM baselines" (Tena 2021, Alvarado 2023, Parra 2024, Ingelse 2023, Botella-Serrano 2023, Tena 2023 FPGA). However an earlier note in this memory claimed Tena 2021 used OhioT1DM (not HUPA) and Ingelse used an earlier 10-patient cohort. Conflict not yet resolved — the literature review currently treats Tena 2021 and the Tan & McBeth 2026 preprint as `[verify]`. Reconciliation task tracked in [[literature-review-gaps]].

**How to apply:**
- Always read `CLAUDE.md` first — it has the current HUPA-specific preprocessing rules and 9 Known Pitfalls.
- Read `skills/SKILL.md` for the 14-step methodology.
- For numeric claims about the cohort, check `data/interim/hupa_cohort_summary.csv` rather than re-loading raw Excel files.
- For UOM-related questions, see `archive/uom_phase/` — historical only.
- Excel lock files `~$HUPA0001P.xlsx` and `~$HUPA0011P.xlsx` were observed in `data/data_hupa/Preprocessed/` — user has these files open in Excel. Ask user to close before pushing or running on Colab.

**Why:** Undergraduate thesis. Research goal: hybrid DL model (CNN-GRU + cross-attention) predicting glucose at 30/60/90-min horizons, beating baselines, with XAI (SHAP + attention + integrated gradients) and uncertainty quantification (MC Dropout or Conformal).
