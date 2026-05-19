# Memory Index

- [User Profile](user_profile.md) — Son, undergraduate student, thesis on blood glucose forecasting for T1D using deep learning
- [Project Context](project_context.md) — HUPA-UCM 25 patients; Steps 0-4 complete (2026-05-19), final 33-feature set (17 dyn + 16 static); next is Step 5 baseline ladder
- [Literature Review Gaps](literature_review_gaps.md) — Open todos in reports/literature_review.md: 3 missing HUPA papers, 4 suspicious citations to verify, missing topical coverage, no Gap Analysis section
- [Glucose Zone Weighting Convention](glucose_zone_weighting_convention.md) — Patient-averaged stats are primary (7.44/60.70/31.86); row-weighted (6.59/71.72/21.70) must always be labelled — differ by 11pt TIR due to HUPA0027/0028 dominance
- [Peri-event over Pearson](peri_event_over_pearson.md) — For sparse event streams (bolus/carb/steps), use peri-event Δglucose vs same-patient control, not raw Pearson r which is structurally near zero
- [Long-patient strategy](long_patient_strategy.md) — Sample-cap (N=5000) with adaptive stride on TRAIN only; never truncate first-N-days; val/test stride=1 + patient-averaged metric
- [Feature engineering writeup structure](feature_engineering_writeup_structure.md) — User-preferred logic: Why → Taxonomy → Per-group → Scaling/leakage → Selection → Implications. Decided 2026-05-19.
- [Deployment tier strategy](deployment_tier_strategy.md) — Step 6 modality-dropout training + Step 8 M0-M4 tier evaluation, planned but not implemented. Defends the "can carb be collected in deployment?" critique.
