# Literature Review and Methodological Rationale

This document records the evidence base for methodological choices in the HUPA-UCM thesis pipeline. It is written as a working research-control document: only claims supported by accessible sources are stated as facts; details not available from accessible abstracts/full text are marked as "not available from accessible text" rather than inferred.

## 1. Dataset Source and Local Evidence

HUPA-UCM is documented by Hidalgo et al. (2024) in *Data in Brief* and distributed through Mendeley Data with DOI `10.17632/3hbcscwz44.1`. The local source article in `data/data_hupa/Article_data.md` states that the dataset contains 25 adults with Type 1 Diabetes Mellitus, with FreeStyle Libre 2 CGM, insulin records, carbohydrate intake, Fitbit-derived calories, heart rate, steps, and sleep-related files. The modelling pipeline in this thesis uses the preprocessed 5-minute files under `data/data_hupa/Preprocessed/`.

Local inspection confirms 25 preprocessed Excel files with an identical eight-column schema: `time`, `glucose`, `calories`, `heart_rate`, `steps`, `basal_rate`, `bolus_volume_delivered`, and `carb_input`. Across all patients, the preprocessed files contain 309,392 rows on a 5-minute grid. The static metadata file `patient_data_characteristic.xlsx` contains all 25 participants and includes gender, HbA1c, age, diabetes duration, weight, height, and treatment.

Important implication: the thesis should not present interpolation and multimodal alignment as novel contributions, because HUPA-UCM was already resampled and curated by the dataset authors using the glUCModel tool. The thesis contribution should focus on leakage-safe feature engineering, patient-aware validation, multimodal ablation, uncertainty, and explainability.

## 2. Required HUPA-UCM and HUPA-Adjacent Prior Work

The HUPA-UCM source article identifies six prior studies as directly relevant to this dataset family. The table below separates papers that use HUPA-UCM from papers that are HUPA-adjacent methodological baselines.

| Paper | Dataset / cohort verified from accessible source | Task | Horizon | Main method | Validation details found | Thesis implication |
|---|---:|---|---|---|---|---|
| Tena et al. 2021 | OhioT1DM, not HUPA-UCM | Regression | 30, 60, 120 min | Ensemble of neural networks; comparison against 10 NN models | Uses common OhioT1DM preprocessing and metrics; details are not a HUPA validation protocol | Architecture benchmark only; not a direct HUPA baseline |
| Alvarado et al. 2023 | HUPA-style patient IDs; 20 real T1D patients | Hypoglycaemia classification | Following 24 h | Wavelet transform + CNN | 75% train, 15% validation, 10% test image split | Supports hypoglycaemia-specific evaluation; split is image-level and not the same as chronological forecasting |
| Parra et al. 2024 | HUPA-UCM cited by source article | Postprandial glucose prediction | Up to 2 h after meals, 15-min steps | Interpretable Sparse Identification by Grammatical Evolution | Four-hour segments; clustering on 2 h pre-meal glucose; Parkes Error Grid safety evaluation | Strong interpretability and clinical-safety baseline |
| Ingelse et al. 2025 journal / 2023 preprint | Real diabetes glucose prediction dataset from UCM/HUPA research line; 10 patients with Medtronic CGM | Regression | 120 min | Grammar-guided genetic programming | k-fold CV: k=10 if enough data, k=8 for selected patients | Shows symbolic regression competition; not the same 25-patient public HUPA release |
| Tena et al. 2023 / 2025 final record | Listed by HUPA source article as using this dataset for hardware LSTM testing | Regression | 30 min | LSTM optimized for wearable FPGA deployment | Full validation split not available from accessible text | Closest LSTM deployment baseline; must compare against plain LSTM/GRU |
| Botella-Serrano et al. 2023 | 25 adults, 14 days CGM + Fitbit; 3 patients discarded for sleep-data inconsistency | Sleep-glycaemic association | Same-day / following-day glycaemic metrics | Clustering and statistical/AI analysis of sleep structure vs glycaemic control | Observational analysis; not a forecasting split | Sleep is relevant but absent from current preprocessed timestep table |
| De La Cruz et al. 2024 | HUPA-style patient IDs and glUCModel context | Hypoglycaemia classification | 30, 60, 90, 120 min | Structured and Dynamic Structured Grammatical Evolution | Personalized, cluster, and general models | Sets a strong expectation for patient-level and interpretable evaluation |

### 2.1 Tena et al. 2021: Ensemble Deep Neural Networks

Tena et al. (2021) proposed two ensemble neural-network models for blood glucose prediction at 30, 60, and 120 minutes and compared them against ten neural-network architectures. The paper uses the OhioT1DM dataset, not the public HUPA-UCM release. Its value for this thesis is therefore methodological rather than dataset-direct: it establishes a neural architecture comparison ladder and demonstrates that model ranking should be performed under a shared preprocessing and metric framework.

Thesis response: do not call Tena et al. (2021) a direct HUPA baseline. Use it to justify comparing multiple baselines under the same leakage-safe HUPA split.

### 2.2 Alvarado et al. 2023: Wavelet Transform plus CNN for Hypoglycaemia

Alvarado et al. (2023) combined wavelet transforms and CNNs to predict hypoglycaemic events over a 24-hour horizon using glucose-only time-series images. The accessible full text reports 20 HUPA-style patients, 24-hour glucose windows, a default hypoglycaemia threshold of 70 mg/dL, rolling-window augmentation, and a 75/15/10 train/validation/test image split.

Thesis response: this is not a direct 30/60/90-minute regression baseline, but it is a direct reminder that hypoglycaemia-focused performance matters. The thesis must report zone-specific errors and not only RMSE/MAE averaged across all glucose values.

### 2.3 Parra et al. 2024: Structured Grammatical Evolution

Parra et al. (2024) proposed interpretable sparse identification by grammatical evolution for postprandial glycaemia prediction. Accessible metadata and abstract text confirm that the study divides data into four-hour segments, clusters based on the two-hour glucose window before meals, predicts two-hour post-meal trajectories in 15-minute steps, and evaluates prediction safety using Parkes Error Grid regions.

Thesis response: this paper is a direct interpretability and safety-evaluation comparator. A neural model must be paired with explanation and clinical-risk analysis, otherwise the thesis will underperform the interpretability standard already present in the HUPA research line.

### 2.4 Ingelse et al. 2025 / 2023: Grammar-Guided Genetic Programming

The HUPA source article cites the 2023 Research Square version. The work now has a peer-reviewed journal record: Ingelse et al. (2025), *Genetic Programming and Evolvable Machines*, DOI `10.1007/s10710-024-09502-5`. Accessible full text indicates a symbolic-regression task predicting glucose two hours ahead, using 10 patients with Medtronic CGM at five-minute intervals. It used four hours of past glucose and insulin/carbohydrate inputs from four hours before to two hours after the prediction time; it applied k-fold cross-validation, with k=10 for patients with enough data and k=8 for patients 1, 8, 9, and 10.

Thesis response: this is not the same public 25-patient HUPA-UCM Excel release, but it is a relevant UCM/HUPA-line symbolic-regression precedent. It also shows that using future insulin/carbohydrate assumptions can be valid in a "what-if" scenario, but the main thesis forecasting task must avoid future exogenous inputs unless explicitly framed as scenario simulation.

### 2.5 Tena et al. 2023 / 2025: LSTM Wearable System

The HUPA source article lists Tena et al.'s LSTM wearable system as a previous use of the dataset for a low-power LSTM medical device targeting 30-minute blood glucose prediction. Accessible sources confirm DOI `10.1109/JBHI.2023.3300511`, a 30-minute horizon, hardware implementation on a Xilinx Virtex-7 FPGA VC707 kit, and comparison with software smartphone implementations and other LSTM FPGA designs. Public index records indicate the final bibliographic record as IEEE JBHI 29(8), 5515-5526 in 2025, while the HUPA article cites the online 2023 DOI record.

Thesis response: this is the closest LSTM deployment baseline. Any proposed CNN-GRU/attention model must be compared against a plain LSTM/GRU and must discuss computational cost, not only accuracy.

### 2.6 Botella-Serrano et al. 2023: Sleep and Glycaemic Control

Botella-Serrano et al. (2023) studied 25 adults with T1DM over 14 days with simultaneous FreeStyle Libre CGM and Fitbit Ionic actigraphy. The paper analyzed 243 days/nights after discarding data from three patients due to sleep-data inconsistency. It reports that poor sleep quality was associated with lower time in range and greater glycaemic variability.

Thesis response: sleep should not disappear silently from the thesis. The current preprocessed Excel modelling table does not include sleep columns, so sleep will be excluded from the main model unless a separate raw-stream sleep feature pipeline is added. This exclusion should be reported as a limitation, not as evidence that sleep is unimportant.

### 2.7 De La Cruz et al. 2024: Explainable Hypoglycaemia Models

De La Cruz et al. (2024) used structured and dynamic structured grammatical evolution to predict hypoglycaemic events at 30, 60, 90, and 120 minutes. Accessible text confirms HUPA-style patient IDs and inputs using two-hour windows of glucose, heart rate, steps, and calories. The paper compares personalized, cluster, and general models and emphasizes if-then-else mathematical expressions as interpretable outputs.

Thesis response: patient-specific, cluster-level, and population-level behavior should be evaluated explicitly. A single aggregate score would be weaker than the evaluation style already used in the HUPA research line.

## 3. Broader Blood Glucose Forecasting Evidence

Nemat et al. (2024) performed a comparative analysis of classical time-series, traditional machine-learning, and deep neural-network approaches for Type 1 Diabetes glucose prediction using Ohio datasets. The accessible abstract states that traditional machine-learning models had the best prediction performance in that comparison and that simply adding extra variables did not necessarily improve prediction performance. This is important because HUPA has many modalities, but their inclusion must be earned empirically.

The literature consistently reports horizon-specific degradation. Shuvo and Islam (2023) report RMSE increasing from 16.06 mg/dL at 30 minutes to 30.89 at 60 minutes, 40.51 at 90 minutes, and 47.39 at 120 minutes on OhioT1DM. A multitask personalized glucose-prediction study reports RMSE increasing from 18.8 mg/dL at 30 minutes to 25.3 at 45 minutes, 31.8 at 60 minutes, 41.2 at 90 minutes, and 47.2 at 120 minutes. These numbers are not HUPA baselines, but they establish the expected direction and magnitude of degradation with longer horizons.

Thesis response: all results must be reported separately for 30, 60, and 90 minutes. Averaging horizons would hide the central difficulty of the task.

## 4. XAI, Multimodal Fusion, Static Context, and Uncertainty

### 4.1 Explainable AI for Healthcare Time Series

Di Martino and Delmastro (2023) reviewed XAI methods for clinical and remote-health applications involving tabular and time-series data. The review supports using different explanation families for different model types: attention can provide temporal/feature weighting for recurrent models, while gradient-based methods and Grad-CAM-style methods are commonly used for CNN components.

Sundararajan et al. (2017) introduced Integrated Gradients for attributing deep-network predictions to input features. Lundberg and Lee (2017) introduced SHAP as an additive feature-attribution framework. Selvaraju et al. (2017) introduced Grad-CAM for gradient-based localization in convolutional networks. For this thesis, the appropriate XAI strategy is not a single method but triangulation: attention weights, Integrated Gradients, SHAP or permutation importance, and modality ablation should be compared for consistency.

Time-series XAI requires caution. Survey literature on XAI for time series highlights temporal dependency and perturbation/masking issues: treating each timestep as an independent tabular feature can generate misleading explanations. Therefore, any SHAP-like analysis should aggregate by feature group and time window and be validated against ablation or permutation tests.

### 4.2 Multimodal Fusion and Patient Context

Nemat et al. (2024) shows that adding exogenous variables is not automatically useful, which supports a controlled modality ladder: glucose-only first, then time/static context, then insulin, carbohydrates, and activity/heart-rate features. Recent multimodal/personalized glucose forecasting work also supports combining CGM, insulin, carbohydrate, physical activity, and attention mechanisms, but those studies are not direct HUPA baselines.

Static and patient-specific modelling is justified by HUPA's large inter-subject variability and complete clinical metadata. Shuvo and Islam (2023) use multitask learning with shared LSTM layers and gender-specific/patient-specific dense layers for personalized glucose prediction. Meta-learning literature for personalized glucose prediction also treats each patient as a distinct task distribution. These works support a static patient branch or patient-aware adaptation, but HUPA-derived static statistics must be computed from training data only to avoid leakage.

For HUPA specifically, modality fusion must be missingness-aware. Local inspection shows fully missing basal, bolus, or carbohydrate signals for several patients and partial basal coverage for long-duration MDI patients. A model that silently treats missing modality as true zero risks learning false physiology.

### 4.3 Uncertainty Quantification

Gal and Ghahramani (2016) cast dropout as an approximate Bayesian method for representing model uncertainty, motivating Monte Carlo Dropout as a practical first uncertainty method. Lakshminarayanan et al. (2017) introduced deep ensembles as a strong uncertainty baseline, but they are more expensive because multiple models must be trained. Angelopoulos and Bates (2021) describe conformal prediction as distribution-free uncertainty quantification that can wrap a trained model, but chronological time-series data weakens naive exchangeability assumptions and requires careful calibration design.

Tan and McBeth (2026) is a verified arXiv preprint using HUPA-UCM for uncertainty-aware neural glucose prediction with LSTM, GRU, and Transformer families plus MC Dropout/evidential outputs. Because it is a preprint, it should be cited as emerging evidence only. It does not replace peer-reviewed UQ foundations.

Thesis response: implement MC Dropout as a feasible first UQ method; report interval coverage and sharpness if intervals are used. Add conformal prediction only with a clearly separated chronological calibration set.

## 5. Clinical Evaluation Literature

Battelino et al. (2019) provide international consensus targets for CGM interpretation, including time in range and time below range. For this thesis, glycaemic-zone analysis should use mg/dL thresholds consistent with the HUPA-UCM unit system: hypoglycaemia `<70`, target range `70-180`, and hyperglycaemia `>180`.

Clarke et al. (1987) introduced Error Grid Analysis for clinical accuracy of self-monitoring blood glucose systems. Parkes et al. (2000) introduced the consensus/Parkes error grid for type 1 and type 2 diabetes. Parra et al. (2024) used Parkes Error Grid safety regions for postprandial glucose prediction, making EGA directly relevant to this thesis.

Implementation decision: do not depend blindly on an unverified package. Use primary definitions for Clarke/Parkes zones where possible, or validate a Python implementation such as `kriventsov/Clarke-and-Parkes-Error-Grids` against known examples before reporting clinical grid percentages.

## 6. Identified Research Gaps

### Gap 1: Prior HUPA Work Does Not Fully Address Patient-Duration Imbalance

Local inspection shows that HUPA0027P alone contributes 53.43% of all preprocessed rows, while HUPA0026P and HUPA0028P bring the top-three share to 74.92%. Accessible prior work does not show a direct response to this exact imbalance in the public 25-patient preprocessed HUPA release.

Thesis response: use patient-level and per-patient reporting, avoid naive subject-mixed random splits, and consider truncation or sample weighting so long-duration patients do not dominate model selection.

### Gap 2: Prior Work Often Reports Aggregate or Task-Specific Performance

Alvarado reports event-classification metrics, Parra reports postprandial safety regions, and Tena's hardware LSTM emphasizes 30-minute deployment. These are valuable but do not provide one unified 30/60/90-minute multimodal regression evaluation over the public preprocessed HUPA table.

Thesis response: produce a unified regression benchmark with MAE/RMSE, zone-specific errors, patient-level errors, and clinical grid analysis for each horizon.

### Gap 3: Missing-Modality Handling Is Under-Specified

Local HUPA inspection shows fully missing basal, bolus, or carb columns for several patients, plus partial basal recording in long-duration MDI patients. Accessible prior papers do not clearly document a missing-modality mask strategy for the public HUPA preprocessed table.

Thesis response: add modality availability indicators, calculate basal coverage, and use modality ablation/dropout so the model does not conflate "not recorded" with "physiologically absent".

### Gap 4: Censored CGM Values Require Explicit Handling

HUPA contains glucose values at the FreeStyle Libre lower cap (`40 mg/dL`) and values above `400 mg/dL`. These are censored or extreme readings rather than ordinary continuous measurements.

Thesis response: retain these rows for clinical relevance, add `glucose_low_cap` and `glucose_high_extreme` indicators, and report sensitivity analysis with and without censored windows.

### Gap 5: Interpretability Is Either Built-In or Post-Hoc, Rarely Both

Symbolic approaches provide white-box rules; neural approaches often rely on post-hoc attribution. The thesis can contribute by combining a high-capacity multimodal sequence model with systematic explanation checks, instead of treating explainability as a final figure.

Thesis response: use attention visualization, Integrated Gradients, SHAP/permutation importance, and modality ablations. Explanations should be checked for consistency.

### Gap 6: Uncertainty Is Not Yet Standard in Peer-Reviewed HUPA Regression Pipelines

Short-term glucose forecasts are more useful when paired with calibrated uncertainty, especially near hypoglycaemic or hyperglycaemic thresholds. Peer-reviewed HUPA-line work emphasizes LSTM deployment, symbolic interpretability, event prediction, and sleep analysis more than calibrated interval forecasts.

Thesis response: implement MC Dropout as the minimum uncertainty layer and add conformal prediction if calibration design is defensible.

## 7. Validation Protocol Requirements for This Thesis

The prior-work review leads to the following non-negotiable validation protocol:

1. No random split of time-series windows for the main result.
2. Use chronological splits within participants or patient-level evaluation folds, and document the choice.
3. Fit scalers, encoders, and derived static features on training data only.
4. Report results per horizon: 30, 60, and 90 minutes.
5. Report per-patient metrics, not only pooled metrics.
6. Report zone-specific errors for `<70`, `70-180`, and `>180` mg/dL.
7. Include sensitivity analysis for censored windows if feasible.
8. Include modality ablation from glucose-only to full multimodal input.
9. Compare against persistence, Ridge/ElasticNet, tree model if available, LSTM, and GRU before claiming benefit from a hybrid model.
10. Include a clinical grid analysis after validating the implementation.

## 8. Model-Selection Rationale

Previous HUPA and HUPA-adjacent studies used ensemble neural networks, hardware LSTM, wavelet-CNN event prediction, grammatical-evolution symbolic models, and sleep-focused glycaemic analysis. These works establish that HUPA-family data are suitable for short-term prediction and event analysis, but they leave an opening for a unified, leakage-safe, public-HUPA, 30/60/90-minute multimodal regression benchmark that handles patient imbalance, missing modalities, censored values, and uncertainty.

This thesis should therefore start with conservative baselines: persistence, Ridge/ElasticNet, Random Forest or XGBoost if available, LSTM, and GRU. The proposed CNN-GRU/attention model with static patient context should only be finalized after EDA and ablation show that multimodal features add value beyond glucose history. Complexity must be justified by measurable gains under the same validation protocol.

## 9. Remaining Open Tasks

These are no longer literature-review blockers, but they remain implementation/reporting tasks:

- Extract exact split details from Tena et al. LSTM hardware full text if institutional access or full PDF becomes available.
- Decide whether sleep will be excluded or engineered from `Raw_Data/`; if excluded, document why.
- Implement and test Clarke/Parkes Error Grid zones.
- During EDA, quantify ACF/PACF, circadian signal, inter-subject variability, modality event density, and usable sequence counts.

## 10. References Checked

- Hidalgo, J. I., Alvarado, J., Botella, M., Aramendi, A., Velasco, J. M., & Garnica, O. (2024). HUPA-UCM diabetes dataset. *Data in Brief*, 55, 110559. https://doi.org/10.1016/j.dib.2024.110559
- Tena, F., Garnica, O., Lanchares, J., & Hidalgo, J. I. (2021). Ensemble models of cutting-edge deep neural networks for blood glucose prediction in patients with diabetes. *Sensors*, 21(21), 7090. https://doi.org/10.3390/s21217090
- Alvarado, J., Velasco, J. M., Chavez, F., Fernandez-de-Vega, F., & Hidalgo, J. I. (2023). Combining wavelet transform with convolutional neural networks for hypoglycemia events prediction from CGM data. *Chemometrics and Intelligent Laboratory Systems*, 243, 105017. https://doi.org/10.1016/j.chemolab.2023.105017
- Parra, D., Joedicke, D., Velasco, J. M., Kronberger, G., & Hidalgo, J. I. (2024). Learning difference equations with structured grammatical evolution for postprandial glycaemia prediction. *IEEE Journal of Biomedical and Health Informatics*, 28(5), 3067-3078. https://doi.org/10.1109/JBHI.2024.3371108
- Ingelse, L., Hidalgo, J. I., Colmenar, J. M., Lourenco, N., & Fonseca, A. (2025). A comparison of representations in grammar-guided genetic programming in the context of glucose prediction in people with diabetes. *Genetic Programming and Evolvable Machines*, 26, 5. https://doi.org/10.1007/s10710-024-09502-5
- Tena, F., Garnica, O., Davila, J. L., & Hidalgo, J. I. (2023/2025). An LSTM-based neural network wearable system for blood glucose prediction in people with diabetes. *IEEE Journal of Biomedical and Health Informatics*. https://doi.org/10.1109/JBHI.2023.3300511
- Botella-Serrano, M., Velasco, J. M., Sanchez-Sanchez, A., Garnica, O., & Hidalgo, J. I. (2023). Evaluating the influence of sleep quality and quantity on glycemic control in adults with type 1 diabetes. *Frontiers in Endocrinology*, 14, 998881. https://doi.org/10.3389/fendo.2023.998881
- De La Cruz, M., Garnica, O., Cervigon, C., Velasco, J. M., & Hidalgo, J. I. (2024). Explainable hypoglycemia prediction models through dynamic structured grammatical evolution. *Scientific Reports*, 14, 12591. https://doi.org/10.1038/s41598-024-63187-5
- Nemat, H., Khadem, H., Elliott, J., & Benaissa, M. (2024). Data-driven blood glucose level prediction in type 1 diabetes: a comprehensive comparative analysis. *Scientific Reports*, 14, 21863. https://doi.org/10.1038/s41598-024-70277-x
- Shuvo, M. M. H., & Islam, S. K. (2023). Deep multitask learning by stacked long short-term memory for predicting personalized blood glucose concentration. *IEEE Journal of Biomedical and Health Informatics*. https://doi.org/10.1109/JBHI.2022.3233486
- Battelino, T., Danne, T., Bergenstal, R. M., et al. (2019). Clinical targets for continuous glucose monitoring data interpretation: recommendations from the international consensus on time in range. *Diabetes Care*, 42(8), 1593-1603. https://doi.org/10.2337/dci19-0028
- Clarke, W. L., Cox, D., Gonder-Frederick, L. A., Carter, W., & Pohl, S. L. (1987). Evaluating clinical accuracy of systems for self-monitoring of blood glucose. *Diabetes Care*, 10(5), 622-628. https://doi.org/10.2337/diacare.10.5.622
- Parkes, J. L., Slatin, S. L., Pardo, S., & Ginsberg, B. H. (2000). A new consensus error grid to evaluate the clinical significance of inaccuracies in the measurement of blood glucose. *Diabetes Care*, 23(8), 1143-1148. https://doi.org/10.2337/diacare.23.8.1143
- Di Martino, F., & Delmastro, F. (2023). Explainable AI for clinical and remote health applications: a survey on tabular and time series data. *Artificial Intelligence Review*, 56, 5261-5315. https://doi.org/10.1007/s10462-022-10304-3
- Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic attribution for deep networks. *Proceedings of ICML 2017*, 3319-3328. https://proceedings.mlr.press/v70/sundararajan17a.html
- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30. https://papers.neurips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions
- Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: visual explanations from deep networks via gradient-based localization. *Proceedings of ICCV 2017*, 618-626. https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html
- Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation: representing model uncertainty in deep learning. *Proceedings of ICML 2016*, 1050-1059. https://proceedings.mlr.press/v48/gal16.html
- Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. *Advances in Neural Information Processing Systems*, 30. https://papers.neurips.cc/paper/7219-simple-and-scalable-predictive-uncertainty-estimation-using-deep-ensembles
- Angelopoulos, A. N., & Bates, S. (2021). A gentle introduction to conformal prediction and distribution-free uncertainty quantification. arXiv:2107.07511. https://arxiv.org/abs/2107.07511
- Tan, H. S., & McBeth, R. (2026). Uncertainty quantification in neural network-based glucose prediction for diabetes. arXiv:2603.04955. https://arxiv.org/abs/2603.04955
