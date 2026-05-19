---
name: blood-glucose-forecasting
version: 2.0.0
description: >
  Research skill for Son's undergraduate thesis titled "A Multimodal Deep Learning
  Approach for Short-Term Blood Glucose Forecasting in Type 1 Diabetes". Governs
  the full research pipeline: dataset understanding, meaningful EDA, preprocessing,
  feature engineering, hybrid deep learning model design, baseline comparison,
  fine-tuning, XAI, application development, Google Colab compatibility, and
  academic report writing in scientific style.
domain: medical-ai
language: python
author: Son
tags:
  - glucose-forecasting
  - blood-glucose-prediction
  - deep-learning
  - multimodal
  - time-series
  - XAI
  - type1-diabetes
  - CGM
  - hybrid-model
  - undergraduate-thesis
triggers:
  - glucose forecasting
  - blood glucose prediction
  - CGM data
  - diabetes deep learning
  - hybrid model diabetes
  - T1D forecasting
  - glucose thesis
  - multimodal diabetes model
colab_compatible: true
report_required: true
language_preference: "Vietnamese for explanations, academic English for report sections"
---

# SKILL: Blood Glucose Forecasting — Undergraduate Thesis Assistant

## Project Identity

**Thesis Title:** A Multimodal Deep Learning Approach for Short-Term Blood Glucose Forecasting in Type 1 Diabetes

**Author:** Son

**Research Goal:** Design and evaluate a hybrid deep learning model that predicts future blood glucose values at multiple short-term horizons (e.g., 30, 60, 90 minutes) for patients with Type 1 Diabetes. Demonstrate its superiority over baseline models only if experiments support that claim. Extend the contribution through stacked hybrid ensembling, XAI, uncertainty quantification, LLM integration, and/or a deployable web/mobile research application.

**Language Behavior:**
- Explain concepts and decisions to the user in **Vietnamese**.
- Write all `report.md` sections in **formal academic English** unless the user explicitly requests Vietnamese.
- Write all code comments in **clear English**.

---

## 0. MANDATORY PRE-CONDITIONS (Non-Negotiable Rules)

These rules apply at every stage of the project without exception.

**Rule 1 — Never speculate before inspecting data.**
Before choosing any preprocessing strategy, model architecture, feature set, or evaluation metric, always inspect the active HUPA-UCM dataset in `data/data_hupa/` and its documentation in `data/data_hupa/Article_data.md`. Summarize what the HUPA files actually contain before proposing anything. Do not inspect or use `data/raw/` for the main pipeline unless the user explicitly asks for T1D-UOM external validation or historical comparison.

**Rule 1b — Use only HUPA-UCM for the main thesis pipeline.**
For EDA, preprocessing, feature engineering, model training, evaluation, XAI, and app demos, use `data/data_hupa/` as the only modelling dataset. Treat `data/raw/` as the old T1D-UOM dataset and exclude it from all main outputs, splits, scalers, processed tables, and training scripts by default. Never mix HUPA-UCM and T1D-UOM in the same processed training table unless the user explicitly requests a separate cross-dataset experiment.

**Rule 2 — Every decision must be evidence-based.**
All methodological choices — data cleaning, imputation, resampling frequency, normalization, feature engineering, model architecture, loss function, evaluation metrics, XAI method — must be explicitly justified by either empirical observations from the data or by citation of relevant peer-reviewed literature (preferably published after 2020). Never make choices by default or convention alone.

**Rule 3 — Always read recent literature before major decisions.**
Before finalizing any significant methodological choice, search for and read recent papers on: short-term blood glucose prediction, CGM-based forecasting, multimodal diabetes models, LSTM/GRU/Transformer/CNN for time series, XAI in healthcare, and clinical evaluation metrics for glucose prediction. Mark claims that need citations with `[citation needed]`. Never invent citations.

**Rule 4 — Do not apply preprocessing blindly.**
Evaluate each transformation or cleaning step against what the data actually requires. Justify both inclusions and exclusions. Do not use all available modalities without checking their completeness, reliability, and relevance.

**Rule 5 — Never cause data leakage.**
In time-series forecasting, the target value must always occur strictly after the input window. Never use random splitting for time-series data unless there is a clearly documented reason. Never fit scalers, encoders, or imputers on validation or test data.

**Rule 6 — All code must run on Google Colab without errors.**
Design every notebook and script with Colab constraints: GPU memory limits, session resets, Drive mounting, dependency installation cells, and lightweight debug configurations. Test with a fresh Colab runtime before finalizing.

**Rule 7 — Do not claim clinical usefulness without evidence.**
This project is research and decision-support only. Never provide or generate medical advice. Any LLM-generated text must clearly state it is for research purposes only and must not replace professional medical judgment.

**Rule 8 — Produce a complete `report.md`.**
After each major stage (EDA, preprocessing, modeling, evaluation), contribute to the running `report.md`. The final version must follow the full academic format defined in Section 11. Write in coherent paragraphs, not disconnected bullet points.

**Rule 9 — Model selection must be gap-driven and justified.**
Whenever selecting a model family or final architecture, explicitly explain: (1) which model classes have already been used in prior literature or project baselines, (2) what limitations remain in those models or their validation protocols, (3) what dataset evidence from EDA supports the proposed choice, and (4) why the selected model is more suitable for this thesis goal than plausible alternatives. Never choose CNN, GRU, LSTM, Transformer, attention, multimodal fusion, or static embeddings because they are fashionable; the choice must address a documented gap such as leakage risk, missing modalities, patient heterogeneity, low-data constraints, deployment constraints, horizon degradation, or lack of clinical interpretability.

---

## 1. Step 1 — Read Project Structure and Dataset Documentation

Before writing a single line of modeling code, the assistant must inspect the project structure completely.

**Primary dataset location for this project (post-2026-05-17 dataset switch):**
- `data/data_hupa/` — HUPA-UCM, 25 patients, primary modelling dataset.
- `data/data_hupa/Article_data.md` — source paper (Hidalgo et al., Data in Brief 2024) describing collection and preprocessing.
- `data/data_hupa/Preprocessed/HUPA*.xlsx` — per-patient 5-min aligned files (CGM, calories, HR, steps, basal, bolus, carb).
- `data/data_hupa/patient_data_characteristic.xlsx` — clinical static metadata (HbA1c, age, gender, weight, height, DX time, treatment).
- `data/data_hupa/Raw_Data/` — per-patient raw sensor streams (Fitbit, FreeStyle, Medtronic) for reference only; not used for modelling.

**Historical / archived dataset:**
- `data/raw/` — old T1D-UOM dataset. Do not use it for main EDA, preprocessing, feature engineering, training, evaluation, XAI, or demos. Use it only if the user explicitly requests external validation, historical comparison, or archived UOM discussion.

### 1.1 What to Check
- The folder and file structure of the project root.
- The dataset description file (`Article_data.md` for HUPA) — read in full.
- All files inside `data/data_hupa/` — formats, sizes, naming conventions, and whether pre-processing has already been applied.
- Available modalities and what each represents: CGM glucose, bolus insulin, basal insulin, meal/carbohydrate intake, physical activity, sleep, heart rate, demographics, or other.
- Patient identifiers and whether data is single-patient or multi-patient.
- Timestamp columns, their format, and timezone.
- Sampling frequency of the primary CGM glucose signal.
- Target glucose units (mg/dL or mmol/L).
- Approximate number of patients, records per patient, and total time span.
- **Coverage gaps**: per-patient counts of fully missing or partially missing modalities (basal, bolus, carb). For HUPA this has been audited in `outputs/hupa_missing_modality_audit.txt`.
- **Censored values**: sensor floor/ceiling caps that may distort training signal. For HUPA: glucose==40 mg/dL (LOW) and glucose>400 mg/dL (HIGH).

### 1.2 What to Produce from This Step
A written summary covering: dataset name and source, collection protocol, CGM device, available modalities, number of subjects, time range, known limitations or biases, **pre-processing already applied by dataset authors**, and any unusual aspects observed during inspection. This summary seeds the Dataset Description section of `report.md`.

---

## 2. Step 2 — Meaningful Exploratory Data Analysis

EDA must not be superficial or performed for aesthetics. Every analysis must be motivated by a question that is directly relevant to the blood glucose forecasting problem. Every plot or table must have a written interpretation and a stated downstream implication.

### 2.1 Structural Analysis
- Data shapes, column names, data types, and memory usage.
- Identify the primary glucose column and verify its units.
- Map all feature columns into categories: temporal, physiological, contextual, static demographic.
- Verify time index monotonicity and regularity.
- Identify file-level inconsistencies across patients.

### 2.2 Data Quality Analysis
Questions to answer:
- How many missing values exist per column and per patient? Are gaps random or systematic? (→ informs imputation vs. segment exclusion decision)
- What is the distribution of gap lengths? (→ determines imputation threshold)
- Are there physiologically impossible glucose values (e.g., < 40 or > 400 mg/dL)? (→ informs outlier handling)
- Are there duplicate timestamps or out-of-order records? (→ informs deduplication and sorting)
- How complete are non-glucose modalities? Are insulin, meal, or activity records consistently timestamped? (→ informs modality inclusion/exclusion)

### 2.3 Glucose Signal Analysis
Questions to answer:
- What is the overall and per-patient distribution of glucose values? (→ informs normalization strategy and output range)
- What proportion of time is spent in hypoglycemia (< 70 mg/dL), euglycemia (70–180 mg/dL), and hyperglycemia (> 180 mg/dL)? Is there class imbalance? (→ informs loss function and evaluation focus)
- What are the ACF and PACF of the glucose signal? At which lags does autocorrelation drop below significance? (→ determines lookback window and lag feature selection)
- What is the glucose rate of change distribution? (→ determines whether velocity and acceleration features add value)
- Are there strong circadian (time-of-day) patterns? (→ determines whether time features are informative)
- Are there inter-subject differences in mean, variance, or pattern shape? (→ determines whether population model or patient-aware model is appropriate)

### 2.4 Multimodal Correlation Analysis (If Applicable)
Questions to answer:
- What is the lagged cross-correlation between insulin events and subsequent glucose? (→ informs insulin feature engineering and lag selection)
- What is the lagged cross-correlation between meal events and glucose? (→ informs carbohydrate feature engineering)
- Is physical activity associated with glucose changes at any lag? (→ informs activity feature inclusion)

### 2.5 Forecasting Feasibility Analysis
- Based on sampling frequency, which prediction horizons (e.g., 15, 30, 60, 90 minutes) are feasible?
- Is the dataset large enough to support deep learning, or should lightweight models be preferred?
- How many usable, gap-free sequences of sufficient length exist per patient?

---

## 3. Step 3 — Data Preprocessing

Apply only preprocessing steps that are warranted by findings from EDA. For each step applied, document: what it does, why it is required for this specific dataset, and what would happen if it were skipped. Never apply a step because it is conventional without verifying its necessity.

**HUPA-specific note (dataset selected for this thesis):** HUPA-UCM has **already been preprocessed** by the dataset authors (Hidalgo et al. 2024) using the glUCModel tool:
- Glucose: subsampled to 15-min then linearly interpolated to 5-min (1 hr filler max).
- Heart rate: rounded and linearly interpolated to 5-min.
- Insulin (basal): MDI long-acting injections divided by 288 to spread across the day.
- Calories: 1-min Fitbit records summed per 5-min bin.
- Steps: summed per 5-min bin; gaps zero-filled.
- Carbs: summed per 5-min bin, converted to servings (1 serving = 10 g); gaps zero-filled.

Therefore Section 3.2 (gap interpolation strategy) is **not a methodology contribution for this thesis**. The preprocessing module should focus on: (a) flagging censored values, (b) building modality availability masks, (c) handling the duration-imbalance among participants, (d) per-subject Z-score, (e) sequence construction. Document explicitly in the report that the imputation choices were inherited from the dataset authors.

### 3.1 Structural Cleaning
- Standardize column names to a consistent format.
- Parse and convert timestamps to a consistent datetime type.
- Sort records chronologically per patient.
- Remove exact duplicate records.
- For HUPA: join `patient_data_characteristic.xlsx` once to attach static clinical metadata.

### 3.2 Missing Value Handling
**Skip for HUPA** — gaps were already linearly interpolated by the dataset authors. Document this fact and proceed.

For datasets where this step is needed, choose strategy based on gap length distribution from EDA:
- **Short gaps (1–3 steps):** Forward-fill or linear interpolation.
- **Medium gaps (4–12 steps at 5-min frequency):** Cubic spline interpolation for physiologically smooth curves.
- **Long gaps (beyond threshold defined from EDA):** Exclude the segment entirely.
- Document chosen thresholds and justify them with data evidence.

### 3.3 Outlier and Censored Value Handling
- Distinguish sensor artifacts from true physiological extremes using domain knowledge thresholds.
- **Censored values are not outliers.** FreeStyle Libre 2 reports `LO` for glucose < 40 mg/dL and `HI` for > 400 mg/dL — these become exactly `40` and `>400` after preprocessing. Retain them but add binary flags `glucose_low_cap` and `glucose_high_extreme`. Optionally mask censored points in the loss function (Tobit-style) or report a sensitivity analysis with/without them.
- Use IQR or Z-score as secondary screening tools only after censoring is accounted for.

### 3.3.b Modality Availability (HUPA-specific)
For HUPA: 5 patients have ≥1 fully-missing modality (HUPA0011/0014/0015/0018/0020 = 5.89% of rows), and 4 patients have only partial basal coverage (HUPA0024/0026/0027/0028). Add the following features to every timestep:
- `basal_available` (binary): 1 if the patient has any basal record across their timeline, else 0.
- `bolus_available` (binary)
- `carb_available` (binary)
- `basal_coverage_24h` (continuous): fraction of past-24h bins with basal>0.

During training, apply modality dropout (randomly zero one branch with ~20% probability) so the model learns to handle missing modalities gracefully at inference. Never zero-fill silently without an availability flag — the model cannot distinguish "no event" from "no recording".

### 3.4 Resampling
- If CGM data is irregularly sampled, resample to a fixed interval (typically 5 minutes for modern CGM devices).
- Justify the chosen interval by device documentation or literature.

### 3.5 Multimodal Alignment
- Align all auxiliary modalities (insulin, meals, activity) to the CGM timestamp grid.
- Use forward-fill for event-based variables where physiologically appropriate (e.g., basal insulin).
- Aggregate activity or sleep data into window-level summaries if needed.
- Exclude modalities that are too sparse or misaligned; document exclusions as limitations.

### 3.6 Normalization and Scaling
- Choose between per-subject Z-score normalization, global min-max scaling, or robust scaling based on inter-subject variability findings from EDA.
- Fit all scalers on training data only. Apply fitted scalers to validation and test sets. Never refit on test data.

### 3.7 Sequence Construction
- Define the lookback window length based on ACF analysis and literature.
- Define forecasting horizons based on clinical relevance and dataset sampling frequency.
- Build sliding-window samples with a defined stride.
- Exclude windows that cross gap boundaries or contain imputed segments beyond a tolerable threshold.

### 3.8 Train/Validation/Test Split
- Split must be strictly chronological — never random for time series.
- If multi-patient: consider patient-aware splitting (some patients entirely in test) to evaluate generalization.
- Define and document split ratios explicitly (e.g., 70/15/15 or 60/20/20 chronological split).

---

## 4. Step 4 — Feature Engineering

All features must be motivated by either EDA findings or literature. After construction, apply feature selection to remove uninformative or redundant features. Validate usefulness before including in final training.

### 4.1 Glucose-Derived Features
- Past glucose values (lag features) at lags identified as significant by PACF.
- Rolling statistics: mean, standard deviation, min, max over windows of 15, 30, 60 minutes.
- Glucose rate of change (velocity): `ΔG/Δt`.
- Glucose acceleration: `Δ²G/Δt²`.
- Minimum and maximum glucose in the input window.
- Time since last valid glucose observation (for irregular data scenarios).
- Glycemic zone indicator at current timestep: binary flags for hypoglycemia risk (< 70) and hyperglycemia (> 180).

### 4.2 Time-Based Features
- Hour of day encoded as cyclical features: `sin(2π·h/24)`, `cos(2π·h/24)`.
- Day of week encoded as cyclical features: `sin(2π·d/7)`, `cos(2π·d/7)`.
- Weekend indicator (binary).
- Apply time features only if EDA showed significant time-of-day effects on glucose.

### 4.3 Multimodal Features (Apply Only If Modality Is Reliable)
- **Insulin:** Bolus insulin amount and timing; basal rate; estimated Insulin On Board (IOB) using pharmacokinetic decay model.
- **Meals:** Carbohydrate amount; estimated glucose absorption curve using meal absorption model (e.g., first-order or trapezoidal).
- **Activity:** Step count, heart rate, or MET values aggregated over recent windows.
- **Sleep:** Sleep duration or quality score if available.
- If a modality is too sparse, unreliable, or difficult to timestamp-align, exclude it and document the exclusion as a methodological limitation.

### 4.4 Patient-Level Static Features (If Multi-Subject)
- Age, sex, BMI, diabetes duration, HbA1c, insulin therapy type.
- Encode as a static context vector fed into a separate branch of the hybrid model.

### 4.5 Feature Selection
After constructing all candidate features:
- Compute Spearman correlation between each feature and future glucose target.
- Compute mutual information scores.
- Run permutation feature importance on a baseline tree model.
- Retain features with meaningful signal. Remove features that are redundant or show near-zero importance. Document removed features and reasons.

---

## 5. Step 5 — Modeling Strategy

The modeling strategy consists of three possible layers: a set of baseline models for comparison, the proposed hybrid deep learning model, and an optional stacked hybrid ensemble if base-model errors are complementary. Architecture choices must be justified by dataset characteristics, not by complexity alone.

### 5.0 Required Model-Choice Rationale
Before proposing or implementing a final model, write a short evidence-based rationale in this structure:

1. **Prior model landscape:** summarize the relevant model classes already used in the literature or in this project (e.g., persistence/statistical baselines, Ridge/ElasticNet, tree ensembles, LSTM/GRU, CNN, Transformer, symbolic models, multimodal fusion).
2. **Limitations or gaps:** identify what those models do not resolve for this thesis. Valid gaps include unclear validation methodology, risk of temporal leakage, weak per-patient generalization, inability to handle missing modalities, poor long-horizon performance, lack of clinical interpretability, excessive computational cost, or lack of deployment feasibility.
3. **Dataset evidence:** cite the EDA findings that motivate the selected architecture: ACF/PACF and lookback evidence, inter-patient heterogeneity, modality sparsity/coverage, hypo/hyper imbalance, sensor caps, long-patient dominance, or Colab/runtime constraints.
4. **Why this model:** state why the selected model is better aligned with the thesis goal than plausible alternatives. The explanation must be comparative, not just descriptive.
5. **Validation response:** state how the experiment design addresses prior limitations, especially chronological split, patient-level reporting, ablation studies, and held-out testing.

Use the following writing pattern in `report.md`: "Previous studies/models used A, B, and C. However, they are limited by X and Y for this dataset/task. This study therefore selects model Z because EDA shows P and Q, and because Z can address the identified gap through mechanism R. The claim will be tested by comparing against A/B/C under the same leakage-safe split."

### 5.1 Baseline Models
Implement all applicable baselines for a fair and credible comparison. Baselines must be properly tuned — not run with default parameters.

| Category | Models |
|---|---|
| Naive statistical | Persistence model (last observation), Exponential Smoothing |
| Classical ML | Linear Regression, Ridge Regression, SVR |
| Ensemble ML | Random Forest, XGBoost, LightGBM |
| Simple deep learning | MLP, LSTM, GRU, 1D-CNN |
| Advanced deep learning | Temporal Convolutional Network (TCN), Bidirectional GRU |
| Transformer-based | Informer or PatchTST (if dataset is large enough) |

For each baseline: tune the most impactful hyperparameters, report all evaluation metrics consistently, and include a brief description in the report.

### 5.2 Proposed Hybrid Model
The specific architecture must be finalized after EDA reveals available modalities and dataset scale. General design pattern:

- **Temporal encoder branch:** LSTM, GRU, or Transformer encoder applied to the CGM time series and aligned temporal modalities.
- **Auxiliary modality branches (if modalities exist):** Separate encoders for insulin history, meal history, or activity — e.g., 1D-CNN branches for event-based signals.
- **Static context branch (if multi-subject):** Dense embedding layers for patient-level static features.
- **Fusion layer:** Concatenation followed by cross-attention, gating, or a dense fusion block.
- **Output head:** Multi-step direct forecasting head (one output per horizon) or sequence-to-sequence decoder.

Candidate architectures (select based on data evidence):
- CNN-LSTM or CNN-GRU
- CNN-BiGRU
- CNN-GRU with self-attention
- CNN-GRU-Attention with static patient embedding
- Multimodal fusion model (if multiple reliable modalities available)

**Design principle:** Recommend the simplest architecture that works well first, then extend if justified. Complexity must earn its place.

### 5.2.b Stacked Hybrid Ensemble Modeling (Advanced Candidate)
Consider **Stacked Hybrid Ensemble Modeling** only after the baseline ladder and single hybrid model are working. This is an advanced thesis contribution, not the first model to implement.

Use this pattern when EDA and baseline results show that different model families make complementary errors across horizons, patients, or glycaemic zones:

1. **Level-0 base learners** trained under the same leakage-safe split:
   - Persistence / Ridge or ElasticNet on engineered lag features.
   - Tree ensemble such as Random Forest, XGBoost, or LightGBM if available.
   - Sequence model such as LSTM or GRU.
   - Proposed hybrid neural model such as CNN-GRU-Attention with static patient context.
2. **Out-of-fold prediction table**:
   - Generate base-model predictions only on validation folds not used to fit that base model.
   - Include horizon-specific predictions (`pred_30m`, `pred_60m`, `pred_90m`) and optional uncertainty summaries.
   - Never train the meta-learner on predictions from models evaluated on the same samples they were trained on.
3. **Level-1 meta-learner**:
   - Start with Ridge/ElasticNet or a shallow Gradient Boosting model.
   - Inputs may include base predictions, horizon ID, current glucose zone, modality-availability flags, and static patient features.
   - Keep the meta-learner simple enough to explain; do not use a large neural network as the stacker unless there is strong evidence.
4. **Gating / mixture-of-experts variant**:
   - If errors differ strongly by patient group, treatment type, missing-modality pattern, or glycaemic zone, evaluate a small gating network or rule-based model selector.
   - The gating model must be trained only on training/validation data and audited for leakage.
5. **Reporting requirement**:
   - Compare stacked ensemble vs. best single model per horizon and per glycaemic zone.
   - Report whether stacking improves hypoglycaemia and long-horizon performance, not only pooled RMSE.
   - Include computational cost and deployment complexity. If the stacked ensemble is too heavy for a mobile demo, use it as an offline benchmark and deploy a distilled/single model.

Do not use stacked ensembles to hide weak base models. If stacking improves only marginally and reduces interpretability or deployment feasibility, prefer the simpler validated model.

### 5.3 Training Protocol
- **Loss function:** MSE as primary. If hypoglycemia events are severely underrepresented, add an asymmetric penalty term that increases loss for errors in the hypoglycemic range.
- **Optimizer:** Adam with learning rate scheduling (ReduceLROnPlateau or cosine annealing with warm restarts).
- **Regularization:** Dropout (tuned), L2 weight decay.
- **Early stopping:** Monitor validation loss with patience of 10–20 epochs.
- **Reproducibility:** Set all random seeds — `numpy`, `torch`/`tensorflow`, `random`, `os.environ['PYTHONHASHSEED']`.
- **Checkpointing:** Save best model weights. Load from checkpoint for evaluation; never re-run training to evaluate.

### 5.4 Hyperparameter Tuning
Tune systematically, not randomly. Priority hyperparameters:
- Lookback window length
- Hidden layer size and number of layers
- Dropout rate
- Batch size and learning rate
- Forecasting horizon(s)

Tuning methods: manual grid search first for a small grid; optionally Optuna or Ray Tune for more thorough search if compute allows.

### 5.5 Evaluation Metrics
Report all metrics per model, per forecasting horizon, and per patient where feasible:

| Metric | Purpose |
|---|---|
| RMSE | Overall error magnitude |
| MAE | Robust average error |
| MAPE | Relative error (only if safe for glucose scale) |
| Clarke Error Grid Analysis (EGA) | Clinical safety classification — mandatory |
| Parkes (Consensus) Error Grid | Alternative clinical grid if applicable |
| Time-in-Range prediction accuracy | Classification into glycemic zones |
| Error by glycemic zone | Does model underperform in hypoglycemia range? |
| Error by patient | Is performance consistent across subjects? |
| Error by horizon | How does error grow with prediction distance? |

If the hybrid model does not outperform all baselines, analyze the reasons honestly and suggest concrete improvements. Do not manipulate results.

---

## 6. Step 6 — Training, Validation, and Fine-Tuning

### 6.1 Training Execution
- Run a small debug experiment first (tiny subset, 2–3 epochs) to verify the full pipeline runs without error on Colab.
- Run full training with checkpointing.
- Log training and validation loss per epoch to a CSV file.
- Plot learning curves before drawing any conclusions about model quality.

### 6.2 Fine-Tuning
- Analyze learning curves to diagnose overfitting or underfitting.
- If overfitting: increase dropout, add L2 regularization, reduce model size, or add data augmentation.
- If underfitting: increase model size, reduce regularization, or train longer.
- Perform patient-specific fine-tuning as an optional ablation if the dataset supports it.

### 6.3 Ablation Studies (Recommended)
Run controlled ablations to quantify the contribution of each component:
- Remove each modality branch individually.
- Remove the fusion mechanism (replace with simple concatenation).
- Remove time-based features.
- Compare single-horizon vs. multi-horizon training.

Ablation results provide strong scientific support for architectural choices and increase thesis credibility.

---

## 7. Step 7 — Explainable AI (XAI)

Implement at least two complementary XAI methods. XAI must be connected to the research question: which features and time steps drive glucose predictions, and does the model's behavior align with clinical knowledge?

### 7.1 SHAP (SHapley Additive Explanations)
- Use `shap.DeepExplainer` or `shap.GradientExplainer` for neural network components.
- Use `shap.TreeExplainer` for tree-based baselines.
- Generate: SHAP summary plot (global importance), SHAP force plots (individual predictions), temporal importance heatmaps.
- Focus case studies on hypoglycemia predictions — these are clinically most critical.

### 7.2 Attention Weight Visualization (If Attention Is in Architecture)
- Extract attention weights from the temporal encoder and visualize them over the input sequence.
- Verify that high-attention timesteps correspond to physiologically meaningful moments (e.g., post-meal glucose rise, insulin action onset).
- Acknowledge attention's limitations as an explanation mechanism.

### 7.3 Gradient-Based Methods
- Apply **Integrated Gradients** for neural network feature attribution.
- Apply **GradCAM** for 1D-CNN components if present.

### 7.4 Permutation Feature Importance
- Measure drop in validation RMSE when each feature group is randomly shuffled.
- Use this as a model-agnostic cross-check of SHAP results.

### 7.5 Modality Ablation (as XAI Evidence)
- Systematically remove each modality and retrain to quantify its contribution to prediction accuracy.
- This is both an engineering experiment and an explanation tool.

### 7.6 XAI Reporting Requirements
- Do not merely present plots. Write an analytical interpretation for each XAI result.
- Connect findings to clinical knowledge: does the model rely on recent glucose history (expected)? Does it respond to post-meal patterns (expected)?
- Critically assess failures: where does the model misattribute importance?
- Acknowledge XAI limitations: SHAP for sequential models has approximation error; attention does not always equal importance.

---

## 8. Step 8 — Extended Contributions

After the core modeling pipeline is validated, select and implement additional contributions to increase research impact. Evaluate feasibility before committing.

### Option A: LLM-Assisted Natural Language Explanation (Recommended)
Use an LLM (GPT-4o, Llama-3, or BioMedLM) as an explanation layer only — not as the prediction model.

Given: model's numerical forecast + SHAP-derived feature importance + current glycemic context, the LLM generates a natural language summary. Implementation requirements:
- Ground all LLM outputs strictly in model outputs and XAI results. Never allow hallucinated medical advice.
- Include a mandatory disclaimer in all LLM-generated outputs.
- Document prompt engineering choices.

### Option B: Uncertainty Quantification (Recommended)
Provide prediction intervals rather than point estimates.

Methods:
- **Monte Carlo Dropout:** Apply dropout at inference time across N forward passes; compute mean and standard deviation.
- **Conformal Prediction:** Distribution-free coverage guarantee without retraining.
- **Deep Ensemble:** Train 3–5 models with different seeds; use prediction spread as uncertainty.
- **Quantile Regression:** Train the model to predict lower/median/upper quantiles (e.g., 5th/50th/95th percentiles) using pinball loss.
- **Evidential Regression:** Predict distribution parameters directly; use only if the loss and calibration diagnostics are understood.
- **Stacked Ensemble Uncertainty:** Combine uncertainty from several base models; report both within-model uncertainty and between-model disagreement.

Required uncertainty outputs:
- Point forecast for 30, 60, and 90 minutes.
- Prediction interval for each horizon, preferably 80% and 90%.
- Calibration metrics: empirical coverage, interval width/sharpness, and coverage by glycaemic zone.
- Failure analysis: under-coverage in hypoglycaemia, hyperglycaemia, censored glucose windows, and missing-modality patients.

Implementation guidance:
- Start with MC Dropout because it is easiest to integrate with the neural model.
- Add conformal prediction as the most defensible interval wrapper if a chronological calibration split is available.
- Use Deep Ensembles or stacked-ensemble uncertainty only if compute allows.
- Do not report uncertainty as meaningful unless coverage is evaluated on a held-out test set.
- Avoid false reassurance: wide intervals, under-coverage, and poorly calibrated uncertainty must be reported honestly.

### Option C: Federated Learning Simulation
Simulate privacy-preserving training where each patient acts as a separate "device" using the **Flower (flwr)** framework.

Compare: federated model vs. centralized model vs. per-patient local model.

### Option D: Real-Time Application / Deployment (Recommended)
Build a demonstrable application using Streamlit or Gradio with:
- CGM data upload or manual entry
- Preprocessing pipeline in the background
- Glucose forecast visualization with uncertainty bands
- SHAP-based feature importance panel
- LLM-generated natural language interpretation (if Option A implemented)
- Hypoglycemia/hyperglycemia risk alert

Deployment targets: Hugging Face Spaces (free), Streamlit Cloud (free), or local Docker container.

### Option D2: Mobile / Wearable-Connected Research App (Aspirational)
If time allows after the validated model and Streamlit/Gradio demo, design a mobile-oriented prototype that connects glucose forecasts with wearable or biosensor streams. This is an extended contribution and must be clearly framed as **research decision-support only**, not a certified medical device.

Possible architecture:
- **Mobile frontend:** Flutter, React Native, or a lightweight progressive web app.
- **Sensor inputs:** CGM export/API where legally and technically available, smartwatch data (heart rate, steps, calories, sleep), manual carbohydrate/insulin entry, and optional phone notifications.
- **Backend:** FastAPI service with `/predict`, `/explain`, `/uncertainty`, `/health`, and `/model-info` endpoints.
- **On-device mode:** Optional distilled model or TensorFlow Lite / ONNX Runtime model for offline forecasting if privacy and latency are priorities.
- **Cloud mode:** Server-side inference for easier updates, heavier ensemble models, SHAP computation, and audit logging.

Expected app features:
- Real-time glucose trend view with 30/60/90-minute forecasts.
- Uncertainty bands and explicit "confidence low" states when intervals are wide or input modalities are missing.
- Alerts for predicted hypoglycaemia or hyperglycaemia risk, with configurable thresholds.
- Missing-sensor and stale-data warnings.
- Modality-status panel showing whether CGM, heart rate, steps, insulin, and carbohydrate streams are present.
- Explanation panel summarizing which recent signals most influenced the forecast.
- Research disclaimer shown consistently: forecasts are not medical advice and must not replace clinician guidance.

Mobile-app research questions:
- Does adding wearable streams reduce forecast error beyond CGM-only history?
- Does uncertainty reduce false confidence in alerts?
- How often do missing or delayed sensor streams make forecasts unreliable?
- Can a small distilled model preserve most of the ensemble/hybrid model's performance for mobile use?

Safety and ethics requirements:
- Never recommend insulin dosing, carbohydrate correction, or treatment action.
- Log alerts and prediction confidence for evaluation, but avoid storing personally identifiable data unless explicitly designed with consent and privacy controls.
- Treat data freshness as a first-class input; stale CGM/wearable data must block or downgrade alerts.
- Clearly separate retrospective demo mode from real-time mode.

### Option E: REST API Deployment
Wrap the trained model in a FastAPI service:
- `POST /predict` — accepts recent glucose sequence, returns forecasts
- `POST /explain` — returns SHAP values for a given input
- `POST /uncertainty` — returns prediction intervals, calibration metadata, and reliability flags
- `GET /health` — service health check
- `GET /model-info` — model metadata and metrics

**Recommendation:** Implement **Option B (Uncertainty) + Option D (App)** as the minimum extended contribution. Add **Stacked Hybrid Ensemble Modeling** if baseline errors are complementary and compute allows. Add **Option D2 (Mobile / Wearable-Connected Research App)** as a design/prototype contribution after the core Streamlit/Gradio app is stable. Add **Option A (LLM)** only if outputs can be grounded strictly in model predictions, uncertainty, and XAI evidence. This combination is practically demonstrable, scientifically defensible, and suitable for an undergraduate thesis defense.

---

## 9. Google Colab Compatibility

All code must run on both local machines and Google Colab without modification or errors.

### 9.1 Required Practices
- Begin every notebook with a Colab detection cell and conditional Google Drive mounting:
  ```python
  import os
  IN_COLAB = 'COLAB_GPU' in os.environ or 'google.colab' in str(get_ipython())
  if IN_COLAB:
      from google.colab import drive
      drive.mount('/content/drive')
      BASE_PATH = '/content/drive/MyDrive/glucose-thesis/'
  else:
      BASE_PATH = './'
  ```
- Use `BASE_PATH` as root for all file I/O. Never use absolute local paths.
- Include a `requirements.txt` and a `!pip install -r requirements.txt` cell at the top of each notebook.
- Check GPU availability and log it at the start of training notebooks.
- Provide a **debug mode** flag that uses a small data subset and 2–3 training epochs for quick sanity checks.
- Use `tqdm` for all loops of significant length.
- Log training metrics per epoch to a CSV file (not just stdout, which is lost on Colab session reset).
- Save all model weights and scalers to `outputs/models/` on Drive.
- Add clear error messages for missing files or incorrect paths.

### 9.2 Notebook Structure (Required Order per Notebook)
1. Install dependencies
2. Mount Drive (conditional on IN_COLAB)
3. Set seeds and load config
4. Load data from Drive
5. Execute the notebook's specific stage
6. Save all outputs to Drive
7. Print a completion summary

---

## 10. Code Organization

```
project-root/
│
├── data/
│   ├── README.md                   ← dataset documentation (read first)
│   ├── raw/                        ← original unmodified data files
│   ├── interim/                    ← partially processed data
│   └── processed/                  ← final cleaned and feature-engineered data
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   ├── 05_evaluation_xai.ipynb
│   └── 06_colab_demo.ipynb
│
├── src/
│   ├── config.py                   ← all hyperparameters and paths
│   ├── data_loading.py
│   ├── eda.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── dataset.py                  ← PyTorch/TF Dataset classes
│   ├── models.py                   ← all model architectures
│   ├── train.py
│   ├── evaluate.py
│   ├── explain.py                  ← XAI methods
│   └── utils.py
│
├── outputs/
│   ├── figures/
│   ├── tables/
│   ├── models/                     ← saved weights and scalers
│   └── logs/                       ← training CSVs
│
├── app/
│   ├── streamlit_app.py
│   └── api.py
│
├── reports/
│   └── report.md
│
├── requirements.txt
└── README.md
```

---

## 11. Report Format (`report.md`)

The `report.md` must be written in **formal academic English** (or formal academic Vietnamese if requested). It must read as a coherent research document, not a collection of bullet points. Each section must contain complete paragraphs that explain what was done, why it was done, how it supports the research goal, and what limitations apply.

### Required Sections

**Abstract** (150–200 words): Problem, dataset summary, methodology, key results, main conclusion.

**1. Introduction:** Clinical motivation for blood glucose forecasting in T1D. Limitations of existing approaches. Research gap. Thesis contributions and structure overview.

**2. Background and Related Work:** Key literature on CGM-based glucose forecasting, deep learning for time series, multimodal models in diabetes, XAI in healthcare, and clinical evaluation frameworks (Clarke EGA, Parkes EGA).

**3. Dataset Description:** Dataset name, source, collection protocol, CGM device, subjects, time span, sampling frequency, available modalities, and known limitations or biases.

**4. Exploratory Data Analysis:** Narrative presentation of EDA findings. For each analysis: question asked → finding → implication for pipeline design. Describe what was learned from each analysis and why it matters for the forecasting task.

**5. Data Preprocessing:** Each preprocessing step as a justified, reasoned choice. Cite EDA evidence or literature for each decision. State what was excluded and why.

**6. Feature Engineering:** Rationale for each feature group. Summary of feature selection process and outcome. Discuss multimodal feature decisions and exclusions.

**7. Modeling Strategy:** Architecture of all baselines, the proposed hybrid model, and any stacked hybrid ensemble if implemented. Training protocol, loss function rationale, uncertainty approach, hyperparameter tuning approach, and architectural diagrams if possible.

**8. Experimental Results and Evaluation:** Performance tables per model, per horizon, per metric. Comparative analysis. Clarke EGA discussion. Patient-level and horizon-level analysis. Honest assessment of where the hybrid model succeeds and where it does not.

**9. Explainability Analysis:** XAI method descriptions. Interpretation of SHAP, attention, and gradient findings. Connection to clinical domain knowledge. Critical assessment of explanation quality and limitations.

**10. Extended Contributions:** Description and critical evaluation of uncertainty quantification, stacked hybrid ensembling, LLM integration, web application, REST API, and/or mobile/wearable-connected research prototype — whichever was implemented.

**11. Discussion:** Synthesis of findings. What do the results imply for the research question? How do they compare to related work?

**12. Limitations:** Honest, specific discussion of data limitations, architectural assumptions, evaluation scope, and clinical applicability constraints.

**13. Future Work:** Concrete, actionable directions for extending this research.

**14. Conclusion:** Summary of contributions and key findings.

**15. References:** All cited papers, datasets, and tools in consistent format (APA or IEEE).

---

## 12. Mandatory Prohibited Actions

The assistant must never do any of the following:

- Assume dataset structure or content without inspecting the actual files.
- Invent dataset characteristics, patient numbers, or statistics.
- Invent experimental results or performance numbers.
- Invent or fabricate citations, paper titles, authors, or findings.
- Use future information in model inputs (look-ahead bias / data leakage).
- Perform random train/test splitting on time-series data without explicit documented justification.
- Fit scalers, normalizers, or encoders on validation or test data.
- Include all modalities without verifying their quality and completeness.
- Choose model complexity without justification from data or literature.
- Claim clinical usefulness without empirical evidence.
- Generate or imply medical advice in any output.
- Write `report.md` sections as disconnected bullet lists.
- Produce code that only works in one specific local environment.
- Ignore Google Colab compatibility requirements.
- Skip any step in the required workflow order.

---

## 13. Decision-Making and Reasoning Style

When the assistant must make a methodological decision, select the best option based on this priority order:

1. Dataset evidence (what the data actually shows)
2. Literature support (what recent papers recommend for similar datasets)
3. Clinical validity (does the choice make physiological sense?)
4. Model feasibility (can it train effectively given dataset scale and Colab constraints?)
5. Thesis contribution (does it add scientific value?)
6. Reproducibility (can others replicate it with the same seed?)
7. Implementation complexity (is it feasible within undergraduate thesis scope?)
8. Google Colab compatibility

When multiple options are viable, compare them explicitly and recommend the most suitable. If the right choice cannot be determined without inspecting data, state that and inspect first.

---

## 14. Required Workflow Order (Do Not Skip or Reorder)

```
Step 0  → Systematic literature review & research synthesis ⭐ LÀM ĐẦU TIÊN
         ├─ Search papers (CGM forecasting, DL time-series, XAI healthcare, uncertainty)
         ├─ For HUPA: include the 6 prior papers that used HUPA-UCM
         │   (Tena 2021, Alvarado 2023, Parra 2024, Ingelse 2023,
         │    Botella-Serrano 2023, Tena 2023 FPGA) — direct baselines
         ├─ Extract insights
         ├─ Create reports/literature_review.md
         └─ Document all methodology decisions with literature support
Step 1  → Read CLAUDE.md, skills/SKILL.md, and inspect data/data_hupa/ completely
         ├─ data/data_hupa/Article_data.md (source paper)
         ├─ Preprocessed/HUPA*.xlsx (per-patient files)
         ├─ patient_data_characteristic.xlsx (static metadata)
         └─ Note pre-processing already applied by dataset authors
Step 2  → Meaningful EDA (produce EDA section of report.md)
Step 3  → Preprocessing design justified by EDA findings
Step 4  → Feature engineering justified by EDA + literature
Step 5  → Baseline model implementation and evaluation
Step 6  → Hybrid model design, training, and evaluation
Step 7  → Comparative analysis: hybrid vs. all baselines
Step 8  → Fine-tuning and ablation studies
Step 9  → XAI integration and analysis
Step 10 → Extended contributions: at minimum Options B + D; consider stacked hybrid ensemble if base models are complementary
Step 11 → Final report.md consolidation (all sections complete)
Step 12 → Colab notebook cleanup and full end-to-end test on fresh runtime
Step 13 → Application deployment (if Option D/D2 chosen): web demo first, then optional mobile/wearable-connected research prototype
```

**Modeling must not begin before EDA is complete and preprocessing decisions are justified.
Report sections must not be written before the corresponding experiments are completed and results are verified.**
