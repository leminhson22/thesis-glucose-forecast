# Model-Choice Rationale for the Step 6 Hybrid Architecture

> Mandatory pre-Step-6 deliverable per `skills/SKILL.md` §5.0 (Rule 9). Five
> evidence-based sections justifying the proposed hybrid CNN-GRU encoder
> with cross-attention modality fusion and a static patient embedding
> branch. References Phase A/B/C.1 experimental results, the HUPA-UCM
> exploratory data analysis, and the prior literature in
> `reports/literature_review.md`. Will be merged into `reports/report.md`
> §7.5 once the Step 6 implementation is complete.

## 1. Prior model landscape

Two non-overlapping prior landscapes inform the choice of a Step 6 model.

**(a) The Phase A/B/C.1 ladder developed in this thesis.** Six baselines
have already been trained on the HUPA-UCM 25-patient cohort under a
strictly chronological per-patient 70/15/15 split with an 18-step boundary
buffer. All six consume the same 17 dynamic features over a 24-step (120
minute) lookback plus 16 static patient features, and all six are
evaluated on the same 45 395-sample test split. Their test pooled MAE in
mg/dL is summarised below.

| Model | 30 min | 60 min | 90 min |
|---|---:|---:|---:|
| Persistence (last-observation) | 13.52 | 22.92 | 29.85 |
| Ridge (α = 0.1) on flattened window | 14.25 | 21.59 | 27.85 |
| Random Forest (n = 300, depth ≤ 25) | 11.79 | 20.68 | 27.31 |
| HistGradientBoosting (best — 300 iters) | **10.40** | **19.77** | **26.27** |
| LSTM (Phase C.1, vanilla MSE) | 15.38 | 21.39 | 27.04 |
| GRU (Phase C.1, vanilla MSE) | 15.76 | 21.42 | 26.75 |

**(b) HUPA-UCM prior baselines from the literature.** Six previously
reported models trained on the same dataset family include
Tena&nbsp;et&nbsp;al. (2021, 2023 FPGA), Alvarado et al. (2023), Parra et
al. (2024), Ingelse et al. (2023), and Botella-Serrano et al. (2023). The
distribution of architectures is dominated by classical sequence models
(LSTM, GRU, 1D-CNN) and engineered-feature tree baselines, with one FPGA
deployment study. Multimodal fusion via cross-attention or modality-gating
is not present in the HUPA family at the time of writing
(`reports/literature_review.md`, §3.2). Outside the HUPA family the
broader CGM-forecasting literature includes Transformer/Informer variants,
TCN, and CNN-LSTM hybrids; these have not been evaluated on HUPA under a
shared chronological-split protocol and therefore offer methodological
inspiration rather than directly comparable numbers.

## 2. Limitations or gaps after Phase A/B/C.1

Step 5 has exposed three concrete gaps that no single baseline closes.

**Gap 1 — Pooled-MAE vs zone-MAE divergence (the TIR-efficiency gap).**
HistGradientBoosting wins pooled MAE at every horizon by roughly 12–25 %
over the recurrent encoders, but it does so by exploiting the
engineered-feature representation (rolling means, IOB/COB, lag features)
to compress predictions toward the time-in-range mean. On the test set
HistGB-300 reports 71.7 % of samples in the time-in-range zone — a
distribution that closely matches the row-weighted prior — and its tail
errors at 60 min hypo (26.75 mg/dL) and 90 min hypo (40.72 mg/dL) are
substantially worse than the recurrent encoders trained without any
asymmetric loss (LSTM 60 min hypo 20.84, 90 min hypo 28.88). This is the
pivotal Phase C.1 finding: pooled MAE alone is a misleading scorecard for
glucose forecasting because it is dominated by the TIR mass.

**Gap 2 — Hypoglycaemia under-detection at 30 min (the loss-bias gap).**
Persistence remains the best model in the hypo zone at every horizon
(test hypo MAE 9.06 / 18.34 / 27.26), and at 30 min no other Step 5
baseline reaches within 1.3 mg/dL of it. Recurrent encoders close the
60/90 min hypo gap with vanilla MSE but still lose 5.7 mg/dL to
Persistence at 30 min. This residual is not a capacity problem (RF and
HistGB have orders-of-magnitude more parameters than the GRU and still
trail Persistence in the hypo zone); it is a loss-function bias —
squared-error symmetric loss aggregates the rare hypo events into a
gradient signal that is dominated by the abundant TIR samples. Phase C.2
addresses this gap with zone-weighted and asymmetric variants on the
same GRU baseline, keeping the architecture unchanged, which keeps the
loss intervention inside Step 5.

**Gap 3 — Modality heterogeneity is not exploited (the fusion gap).**
HUPA-UCM contains four event-based modalities (basal, bolus, carbohydrate,
steps) whose presence varies substantially across patients. Five patients
have one or more fully missing modalities (HUPA0011/0014/0015/0018/0020
account for 5.89 % of rows); four long-duration MDI patients have only
40–66 % basal coverage; and the cross-patient standard deviation of the
peri-event Δglucose response is roughly twice the mean magnitude (EDA
§4.6.2). Flat-window models (Ridge/RF/HistGB) cannot route per-patient
emphasis between modalities, and a single recurrent encoder treating all
17 dynamic features as one homogeneous input cannot either. There is room
for a model that (a) carries a per-patient static embedding that
conditions the temporal encoder and (b) routes attention across modality
streams so that, e.g., the bolus-derived IOB feature is up-weighted for
patients with reliable insulin logging and de-weighted for patients
without.

These three gaps motivate Step 6 rather than further depth in Step 5.

## 3. Dataset evidence supporting a hybrid architecture

Five concrete EDA and Step 5 observations justify each architectural
component.

**Autocorrelation justifies a recurrent (not bag-of-lags) encoder.** The
glucose autocorrelation function decays smoothly across the first 18–24
lags and exhibits non-zero structure at horizons up to 36 lags. A
recurrent encoder can model this decay parametrically with O(F · H + H²)
weights per layer, whereas the flat-window baselines must allocate a
separate weight for every (lag × feature) pair (≈ 408 dynamic terms at T =
24). Phase C.1 demonstrates that the recurrent representation is
empirically tighter on tails: LSTM 90 min hypo MAE 28.88 vs HistGB 40.72
(both at 17 features and the same split). A purely convolutional encoder
would also fit, but a CNN front-end paired with a recurrent reader gives
the network short-range temporal pattern extractors (1D-CNN kernels
attuned to the post-meal rise pattern and the post-bolus drop pattern)
ahead of the long-range temporal integration.

**Per-patient response heterogeneity justifies a static embedding
branch.** Cross-patient SD of peri-event Δglucose response is roughly 2 ×
its mean magnitude (EDA §4.6.2). HbA1c ranges from 6.0 to 9.7 % across
the cohort, diabetes duration from 0.8 to 39.5 years, and treatment
type splits 14 CSII / 11 MDI (with HUPA0011P reclassified to MDI per the
data anomaly noted in CLAUDE.md). A static patient embedding fed into the
fusion layer lets the network condition its temporal-encoder output on
the patient's static profile rather than requiring the temporal encoder
to re-derive these constants from the lookback every sample. The Phase
A/B/C.1 baselines all consume the static vector by concatenation; the
Step 6 model promotes the static branch from a flat appendix to a
conditioning signal at the fusion layer.

**Modality availability heterogeneity justifies cross-attention.**
Patient-level analysis shows 13 patients with all four modalities
recorded continuously, 4 patients with intermittent basal, and 5 patients
with at least one fully missing modality. Concatenation-then-dense fusion
gives every patient the same modality weighting at training time, which
forces the network to either ignore a modality or pretend a missing
modality is "zero activity". Cross-attention across modality streams,
augmented by the per-patient static embedding as the query, lets the
network learn a patient- and sample-conditional modality weight. The same
mechanism plugs into the Step 6 modality-dropout training regime (30 %
random branch zero-out during training, per
[[deployment-tier-strategy]]), so that the M0–M4 deployment-tier
evaluation in Step 8 can mask modalities at inference time without
catastrophic accuracy collapse.

**Engineered-feature absorption justifies CNN front-end (not pure
Transformer).** HistGB-300 wins TIR because the engineered features
(rolling means at 30/60/120 min, IOB, COB, 150-min steps sum) act as
hand-crafted feature detectors that the gradient-boosted trees route
directly to leaves. A pure Transformer encoder over raw 5-minute samples
would need O(T²) attention to discover analogous detectors and a
substantially larger training set. A 1D-CNN front-end with kernel sizes
3, 5, and 7 5-minute steps (15, 25, 35 min receptive fields) directly
parameterises the same family of feature detectors with O(T · F · K)
weights, then hands the resulting time-step representations to the
recurrent reader. This is the "absorb tree-style features" mechanism
required to close the TIR-efficiency gap inside a neural model.

**Long-patient dominance justifies the patient-averaged early-stopping
target.** HUPA0027P, HUPA0026P, and HUPA0028P together account for
74.93 % of rows. Pooled validation MAE is therefore essentially the
weighted average of those three patients' performance. Phase C.1 already
adopted patient-averaged validation MAE as the early-stopping target;
Step 6 inherits this choice unchanged, since the cohort imbalance is a
property of the dataset and not the model.

## 4. Why CNN-GRU with cross-attention and static embedding (vs.
alternatives)

Four plausible alternatives were considered and rejected.

**LSTM with attention only (no CNN, no cross-modal fusion).** This is
the simplest extension of Phase C.1. It would close the TIR gap if the
gap were a temporal-coverage problem, but EDA evidence is that the gap is
a feature-detection problem (HistGB's wins come from engineered features,
not longer lookback). Without a CNN front-end, the LSTM is forced to
re-derive the rolling-mean and rate-of-change features the trees consume
ready-made.

**Transformer/Informer encoder over the same window.** Empirically
attractive but underpowered on this dataset scale: 68 395 training
sequences is at the low end of the Transformer regime, and attention's
O(T²) complexity over T = 24 yields only 576 attention scores per head —
the same parameter budget can buy a more sample-efficient CNN-GRU.
Additionally, no published HUPA baseline uses a Transformer encoder; the
direct comparison would be against the recurrent baselines we already
have.

**TCN (temporal convolutional network) replacing the recurrent reader.**
TCNs are competitive with LSTM/GRU for CGM forecasting at horizons up to
60 min in the broader literature, but they substitute one temporal
representation for another rather than adding the cross-modal routing the
fusion gap demands. A TCN reading the same 17-channel input is closer to
a "stronger Phase C.1" than to a Step 6 model.

**Stacked ensemble of HistGB plus the GRU.** This is a viable Step 10
follow-up (SKILL.md §5.2.b) and is preserved as an option after Step 6.
However, stacking adds deployment complexity and obscures the
mechanism-level claim of this thesis (that a single model can absorb
tree-style features and recurrent-style temporal integration). The Step 6
hybrid is the more falsifiable scientific claim.

The chosen architecture is therefore a **CNN-GRU with cross-attention
modality fusion and a static patient embedding**, with the following
sketch.

```
X_dyn (B, 24, 17)   ──► 1D-CNN front-end (kernels 3/5/7) ──► (B, 24, F')
                                                            │
                                                            ▼
                                            GRU reader (hidden 64, layers 2)
                                                            │
                       ┌────────────────────────────────────┤
X_dyn split by modality│                                    │
into (insulin, carbs,  │     cross-modal multi-head          │
activity, hr, glucose) │       attention (query = static)    │
                       └────────────────────────────────────┤
                                                            ▼
X_stat (B, 16)        ─► static MLP ─►   concat & fuse   ─► (B, 96)
                                                            │
                                                            ▼
                                              head ─► (B, 3) mg/dL
```

Training-time modality dropout zeroes one randomly chosen modality branch
with probability 0.3 (per `[[deployment-tier-strategy]]`), so that the
M0–M4 tier evaluation in Step 8 can mask modalities at inference time
without retraining.

## 5. Validation response — how the experiment design addresses prior
limitations

The proposed Step 6 experiment commits to four falsification mechanisms.

1. **Identical input contract.** The hybrid will consume the same 17
   dynamic features and 16 static features as Ridge / RF / HistGB / LSTM
   / GRU, evaluated on the same 45 395-sample test split. This removes
   "different features" as a confound.

2. **Identical loss-and-stop protocol as Phase C.2.** The hybrid will
   train with the Phase C.2 winning loss (selected once C.2 finishes) and
   the same `early_stopping_patience = 5` plus `dropout = 0.3` retunes
   from Phase C.1's overfitting analysis. This isolates the architectural
   change as the only variable when comparing Step 6 to Phase C.2.

3. **Per-zone reporting as a primary metric.** Step 7's master comparison
   table will report MAE by horizon × glycaemic zone × model alongside
   pooled MAE. A hybrid that improves pooled MAE while degrading hypo MAE
   would not be accepted; a hybrid that ties HistGB on TIR while beating
   it on hypo would be the desired outcome.

4. **Patient-aware reporting + M0–M4 modality ablation.** Step 7 will
   report patient-averaged MAE alongside pooled MAE (long-patient
   strategy: see `[[long-patient-strategy]]`). Step 8 will run the
   M0–M4 tier evaluation with inference-time modality masking, so the
   "what if a patient has no carb logging" critique is answered with
   numbers rather than a hand-wave.

The hypothesis the thesis will test is therefore: **the proposed CNN-GRU
hybrid with cross-attention modality fusion and a static patient
embedding, trained with the Phase C.2 winning loss and 30 % modality
dropout, outperforms HistGradientBoosting on patient-averaged 30-minute
hypoglycaemia MAE without degrading test-set pooled MAE relative to
HistGB-300, evaluated on the same chronological 70/15/15 HUPA-UCM split.**

If this hypothesis fails — e.g., the hybrid ties HistGB on hypo MAE but
loses pooled MAE — the thesis will report the failure honestly and
recommend a Step 10 stacked-ensemble fallback (HistGB + GRU + meta-Ridge)
as the most likely route to a clinically usable model.
