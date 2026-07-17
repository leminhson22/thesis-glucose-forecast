# GlucoSight — Application Plan

> Companion application cho thesis: *"A Multimodal Deep Learning Approach for Short-Term Blood Glucose Forecasting in Type 1 Diabetes"*.
>
> Tài liệu này KHÔNG phải là academic report. Nó là kế hoạch kỹ thuật để build `app/`. Báo cáo học thuật vẫn theo `reports/report.md` bằng tiếng Anh.

---

## 0. Working assumptions (cần Son confirm)

| # | Quyết định mặc định | Lý do | Cần đổi nếu... |
|---|---|---|---|
| A1 | Hỗ trợ cả Patient View và Clinician View (toggle) | Module 2 (Risk Radar) cho bệnh nhân, Module 4 (Pattern Reports) cho bác sĩ — đều đã có sẵn metric trong evaluation plan | Hội đồng yêu cầu chỉ 1 persona |
| A2 | Deploy lên Hugging Face Spaces (free tier, CPU) + có thể chạy local | HF Spaces có link public để hội đồng bấm vào xem; không phụ thuộc máy demo | Trường yêu cầu on-premise / có dữ liệu thật của bệnh nhân |
| A3 | Cho phép upload CSV theo schema HUPA, mặc định là playback HUPA | Upload tăng tính thực tiễn; playback đảm bảo demo luôn chạy | Hội đồng cấm upload (vấn đề dữ liệu nhạy cảm) |
| A4 | Streamlit (không phải React/FastAPI) | Scope undergrad, 1 người, 4-6 tuần; Streamlit đủ cho 4 module này | Cần real-time WebSocket hoặc multi-user state |
| A5 | Mọi text dùng tiếng Anh trong app | Đồng nhất với academic report; HUPA cohort gốc dùng tiếng Tây Ban Nha nhưng tài liệu thesis tiếng Anh | Hội đồng yêu cầu demo tiếng Việt |

---

## 1. File structure

```
app/
├── README.md                       # How to run locally + on HF Spaces
├── PLAN.md                         # File này
├── requirements.txt                # Streamlit + plotly + reportlab + torch (CPU)
├── app.py                          # Streamlit entry point + global config
├── config.py                       # Paths, thresholds, model version
│
├── inference/
│   ├── __init__.py
│   ├── model_loader.py             # Load .pt model + scaler.pkl từ outputs/models/
│   ├── predictor.py                # predict(window) -> {30m, 60m, 90m, CI}
│   ├── explainer.py                # SHAP/Integrated Gradients wrapper
│   └── risk_classifier.py          # forecast -> {SAFE, WATCH, ACT} + P(hypo)
│
├── data/
│   ├── __init__.py
│   ├── loader.py                   # Load HUPA Excel for playback
│   ├── uploader.py                 # Validate user-uploaded CSV
│   └── preprocessor.py             # ⚠️ Re-use src/preprocessing.py — KHÔNG viết lại
│
├── ui/
│   ├── __init__.py
│   ├── pages/
│   │   ├── 1_Forecast_Panel.py
│   │   ├── 2_Risk_Radar.py
│   │   ├── 3_Why_Explainer.py
│   │   └── 4_Pattern_Reports.py
│   ├── components/
│   │   ├── glucose_chart.py        # Plotly: history + forecast band
│   │   ├── risk_badge.py           # Traffic-light component
│   │   ├── shap_bars.py            # Horizontal SHAP bars
│   │   ├── disclaimer.py           # Mandatory medical disclaimer banner
│   │   └── sidebar.py              # Patient picker + playback controls
│   └── assets/
│       ├── style.css
│       └── disclaimer.md
│
├── reports/
│   ├── pdf_generator.py            # ReportLab-based PDF for Module 4
│   └── templates/
│       └── clinical_report.html    # Jinja2 template (nếu dùng WeasyPrint)
│
├── tests/
│   ├── test_predictor.py           # Smoke test: model loads, predicts 1 window
│   ├── test_preprocessor.py        # Đảm bảo app preprocessing == training
│   ├── test_risk_classifier.py     # Threshold edge cases
│   └── fixtures/
│       └── sample_window.npz
│
└── deploy/
    ├── Dockerfile                  # cho HF Spaces nếu cần
    └── space_config.yaml           # HF Spaces metadata (SDK: streamlit)
```

**Nguyên tắc tách biệt code**:

- `app/` KHÔNG được duplicate logic của `src/`. Mọi feature engineering, scaling, sequence building phải import lại từ `src/preprocessing.py`. Nếu phải copy thì đó là dấu hiệu cần refactor `src/`.
- Model artefact load qua **một version string duy nhất** trong `config.py` (vd. `MODEL_VERSION = "cnn_gru_v3_2026-05-20"`). Đổi model = đổi 1 dòng.
- Tất cả threshold (hypo = 70, hyper = 180, alert probability cutoff) tập trung trong `config.py`, không scatter trong UI.

---

## 2. UI mockups (wireframe text)

### 2.1 Global layout

```
┌───────────────────────────────────────────────────────────────────┐
│  🩸 GlucoSight                              [Patient ▼] [Clinician]│
├──────────────┬────────────────────────────────────────────────────┤
│              │                                                    │
│  SIDEBAR     │            MAIN PAGE (changes per tab)             │
│              │                                                    │
│  • Patient   │                                                    │
│    [HUPA0027▼│                                                    │
│  • or Upload │                                                    │
│    [📁 CSV]  │                                                    │
│              │                                                    │
│  Playback:   │                                                    │
│  ◀◀ ◀ ▶ ▶▶  │                                                    │
│  Speed 10×   │                                                    │
│              │                                                    │
│  ⏱ 2024-04   │                                                    │
│    -12 14:35 │                                                    │
│              │                                                    │
│  ─────────── │                                                    │
│  ⚠ Research  │                                                    │
│    only. Not │                                                    │
│    medical   │                                                    │
│    advice.   │                                                    │
└──────────────┴────────────────────────────────────────────────────┘
[Tabs: Forecast | Risk Radar | Why? | Pattern Reports]
```

### 2.2 Tab 1 — Forecast Panel

```
┌────────────────────────────────────────────────────────────────────┐
│  Current glucose: 142 mg/dL  ↗  (rising 1.2 mg/dL/min)             │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│    250 ┤                          ▓▓ forecast 95% CI ▓▓            │
│        │                       ╱─────╲                             │
│    180 ┤━━━━━━━━━━━━━━━━━━━━━━╱━━━━━━━╲━━━━━━━━━ hyper             │
│        │                ╱────╱          ╲                          │
│    142 ┤━━━━━━━━━━━━━━━●                                            │
│        │       ╱──────╱                                            │
│     70 ┤━━━━━━╱━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ hypo              │
│        │                                                           │
│     40 ┤                                                           │
│        └──┬───────┬───────┬───────┬───────┬───────┬───────┬────    │
│         -3h     -2h     -1h     now    +30m    +60m    +90m        │
│                                                                    │
│  Markers: ▼ bolus 4U @ 12:30   🍽 carb 45g @ 12:25   🏃 -2h         │
│                                                                    │
├────────────────────────────────────────────────────────────────────┤
│  Forecast:                                                         │
│    30 min:  167 ± 12 mg/dL  →                                      │
│    60 min:  185 ± 21 mg/dL  ↗                                      │
│    90 min:  178 ± 30 mg/dL  ↘                                      │
└────────────────────────────────────────────────────────────────────┘
```

### 2.3 Tab 2 — Risk Radar

```
┌────────────────────────────────────────────────────────────────────┐
│  RISK STATUS                                                       │
│                                                                    │
│   Next 30 min:   🟢 SAFE      P(hypo) 2%   P(hyper) 18%            │
│   Next 60 min:   🟡 WATCH     P(hypo) 8%   P(hyper) 42%            │
│   Next 90 min:   🟡 WATCH     P(hypo) 5%   P(hyper) 51%            │
│                                                                    │
│   Suggested re-check: 25 minutes                                   │
│                                                                    │
├────────────────────────────────────────────────────────────────────┤
│  Recent alerts (last 24h):                                         │
│  • 03:15  HYPO predicted 45 min ahead   actual: 62 mg/dL  ✓ HIT    │
│  • 09:40  HYPER predicted 60 min ahead  actual: 195 mg/dL ✓ HIT    │
│  • 13:20  HYPO predicted 30 min ahead   actual: 88 mg/dL  ✗ MISS   │
│                                                                    │
│  Lead-time avg: 47 min | Sensitivity: 0.82 | FAR/day: 0.6          │
└────────────────────────────────────────────────────────────────────┘
```

**Threshold logic** (in `risk_classifier.py`):
- ACT  → P(hypo) ≥ 0.6 OR P(hyper>250) ≥ 0.6
- WATCH → 0.3 ≤ P(hypo) < 0.6 OR 0.3 ≤ P(hyper) < 0.6
- SAFE → còn lại

Threshold sẽ calibrate trên validation set, không hardcode arbitrary.

### 2.4 Tab 3 — Why Explainer

```
┌────────────────────────────────────────────────────────────────────┐
│  Why is glucose predicted to RISE in the next 60 minutes?          │
│                                                                    │
│   Top contributors (SHAP):                                         │
│                                                                    │
│   carb_60m_sum (45g)            ████████████████  +23 mg/dL  ↑     │
│   bolus_remaining_iob (1.8U)    ██████             -14 mg/dL  ↓    │
│   glucose_velocity (+1.2)       ██████████         +18 mg/dL  ↑    │
│   steps_30m_sum (0)             ███                 +6 mg/dL  ↑    │
│   hour_sin (post-lunch)         ██                  +4 mg/dL  ↑    │
│                                                                    │
│  Plain-English summary:                                            │
│    Glucose is rising because of the 45g carb intake at 12:25       │
│    which has not been fully covered by the 4U bolus at 12:30.      │
│    Low activity in the last 30 minutes amplifies the rise.         │
│                                                                    │
│  [Hover over any past event in the timeline to see its impact]     │
└────────────────────────────────────────────────────────────────────┘
```

### 2.5 Tab 4 — Pattern Reports

```
┌────────────────────────────────────────────────────────────────────┐
│  Date range: [2024-04-01] → [2024-04-14]   [📄 Generate PDF]       │
├────────────────────────────────────────────────────────────────────┤
│  Time in Range by hour-of-day (heatmap):                           │
│                                                                    │
│    Mon ▓░░▓▓▒▒░░░░▓▓▒▒░░░░▓▓▒▒░░  TIR: 68%                         │
│    Tue ▓▓▒░▓▓▒▒░░░░▓▓▒▒░░░░▓▓▒▒░  TIR: 72%                         │
│    Wed ▓▓▒░▓▓▒▒░░░░▓▓▒▒░░░░▓▓▒▒░  TIR: 65%                         │
│    ...                                                             │
│                                                                    │
│  Recurring scenarios detected:                                     │
│    1. Nocturnal hypo @ 03:00-04:00 on days after evening exercise  │
│       (3 occurrences in selected range)                            │
│    2. Post-breakfast spike >220 mg/dL                              │
│       (5 occurrences, mean peak +110 mg/dL)                        │
│    3. Pre-lunch hypo if last bolus >5h ago                         │
│       (2 occurrences)                                              │
│                                                                    │
│  [PDF preview]                                                     │
└────────────────────────────────────────────────────────────────────┘
```

PDF nội dung: tóm tắt TIR/hypo/hyper, heatmap hour×day, recurring scenarios + ví dụ minh hoạ, disclaimer "research artefact". Mục đích: bệnh nhân in ra mang đi gặp bác sĩ.

---

## 3. Inference contract

`predictor.predict(window: np.ndarray, static: np.ndarray) -> dict`:

```python
{
    "horizons": {
        30: {"mean": 167.0, "lower_95": 143.0, "upper_95": 191.0},
        60: {"mean": 185.0, "lower_95": 144.0, "upper_95": 226.0},
        90: {"mean": 178.0, "lower_95": 118.0, "upper_95": 238.0},
    },
    "probabilities": {
        30: {"hypo": 0.02, "in_range": 0.80, "hyper": 0.18},
        60: {"hypo": 0.08, "in_range": 0.50, "hyper": 0.42},
        90: {"hypo": 0.05, "in_range": 0.44, "hyper": 0.51},
    },
    "model_version": "cnn_gru_v3_2026-05-20",
    "timestamp": "2024-04-12T14:35:00",
}
```

CI tính bằng **MAE percentile từ validation set per horizon**, không phải bootstrapping toàn bộ mô hình mỗi lần inference (quá chậm). Probability tính bằng mô hình phụ classifier head HOẶC bằng cách giả định Gaussian quanh mean với σ = RMSE_val.

---

## 4. Implementation roadmap (6 tuần)

Assumption: Son làm part-time song song với việc training/evaluation model chính. Mỗi tuần ~10-15h trên `app/`.

### Tuần 1 — Foundation & skeleton

**Deliverable**: `streamlit run app/app.py` mở ra một trang trống có sidebar, disclaimer, và playback control giả (chưa gọi model).

- [ ] Tạo `app/` structure đúng như mục 1
- [ ] `requirements.txt`: streamlit, plotly, pandas, numpy, torch (cpu), reportlab, jinja2, openpyxl
- [ ] `app.py`: setup `st.set_page_config`, sidebar, 4 tab placeholder
- [ ] `ui/components/disclaimer.py`: banner đỏ ở header + footer trên mọi page
- [ ] `data/loader.py`: load 1 file HUPA, expose `get_window(patient_id, t)` → trả về 24 timestep
- [ ] Smoke test: chọn HUPA0027, slider thời gian di chuyển, console print window shape

**Acceptance**: app chạy local không crash, sidebar đổi patient thay đổi current time.

### Tuần 2 — Module 1 (Forecast Panel)

**Deliverable**: Tab 1 hiển thị history + forecast cho mô hình baseline (Persistence hoặc LSTM).

- [ ] `inference/model_loader.py`: load `.pt` từ `outputs/models/` + scaler
- [ ] `inference/predictor.py`: implement `predict()` contract như mục 3 cho baseline
- [ ] `ui/components/glucose_chart.py`: Plotly chart với history line, forecast line, CI band, reference zones
- [ ] Annotations: bolus ▼, carb 🍽, exercise markers từ window data
- [ ] Number cards: 30/60/90 min predicted ± MAE

**Acceptance**: với HUPA0027 ở thời điểm bất kỳ, chart hiển thị đúng, forecast extend ra phải, CI band visible.

⚠️ **Phụ thuộc**: Cần ít nhất 1 model đã train xong (có thể là baseline). Nếu model chính chưa xong, dùng Persistence baseline để app vẫn chạy được.

### Tuần 3 — Module 2 (Risk Radar)

**Deliverable**: Tab 2 với traffic-light, probability bars, alert history.

- [ ] `inference/risk_classifier.py`: forecast + CI → P(hypo/in-range/hyper)
- [ ] Calibrate threshold trên validation set, lưu vào `config.py`
- [ ] `ui/components/risk_badge.py`: traffic-light visual (HTML/CSS)
- [ ] Alert history trong `st.session_state`, hiển thị bảng
- [ ] Live metric: lead-time avg, sensitivity, FAR/day (tính trên playback đã chạy)

**Acceptance**: Khi playback qua một episode hypo thật trong HUPA, app phát alert trước ≥ 30 min ở ≥ 70% trường hợp.

### Tuần 4 — Module 3 (Why Explainer)

**Deliverable**: Tab 3 với SHAP bars + plain-English summary.

- [ ] `inference/explainer.py`: SHAP DeepExplainer hoặc Integrated Gradients (gradient-based nhanh hơn cho mô hình deep)
- [ ] Cache SHAP values cho mỗi window (key = patient_id + timestamp)
- [ ] `ui/components/shap_bars.py`: horizontal bar chart top 5 features
- [ ] Template-based summary generator: rule-based, không dùng LLM
  - if `carb_60m_sum` top positive → "carb intake of Xg at HH:MM"
  - if `bolus_iob` top negative → "active insulin from bolus at HH:MM"
  - v.v.
- [ ] Interactive: `st.plotly_chart` event để hover xem contribution

**Acceptance**: với 5 case demo có sẵn (sau ăn, sau insulin, sau tập, đêm, sáng), summary đọc tự nhiên và đúng causal direction.

### Tuần 5 — Module 4 (Pattern Reports) + Clinician View

**Deliverable**: Tab 4 với heatmap + recurring scenarios + PDF download.

- [ ] Heatmap TIR theo hour × day-of-week (Plotly)
- [ ] Recurring scenario detector: cluster các excursion bằng rule-based (`scipy.signal.find_peaks` + grouping theo điều kiện) — KHÔNG cần unsupervised ML
- [ ] `reports/pdf_generator.py`: ReportLab generate PDF từ template
- [ ] PDF nội dung: thông tin bệnh nhân anonymous, TIR summary, heatmap, scenarios, disclaimer
- [ ] Clinician View toggle: ẩn Risk Radar, hiện thêm cohort-level metrics nếu có nhiều patients

**Acceptance**: PDF render đúng, mở được trên Adobe Reader, không vỡ Unicode.

### Tuần 6 — Upload, deploy, polish

**Deliverable**: HF Spaces public link + README hoàn chỉnh + demo script.

- [ ] `data/uploader.py`: validate CSV upload đúng schema HUPA (8 cột bắt buộc, 5-min grid)
- [ ] Error messages thân thiện khi schema sai
- [ ] `deploy/Dockerfile` + `space_config.yaml`
- [ ] Push lên HF Spaces, test public link
- [ ] `README.md`: hướng dẫn local + cloud + format CSV input
- [ ] Demo script: 5-phút workflow cho buổi bảo vệ (chọn HUPA0027 → đến đúng moment có hypo → show Risk Radar alert → switch sang Why → show carb cause → export PDF)
- [ ] Test trên cả 25 HUPA patient để chắc không có patient gây crash (đặc biệt 4 patient missing modality)

**Acceptance**: hội đồng bấm vào HF link, app load < 30s, demo flow chạy không crash.

---

## 5. Risk register

| Rủi ro | Likelihood | Impact | Mitigation |
|---|---|---|---|
| SHAP quá chậm trên timeseries dài | High | Medium | Dùng Integrated Gradients (gradient-based, ~10× nhanh) hoặc kernel SHAP với sample nhỏ. Cache mọi kết quả. |
| HF Spaces free tier OOM khi load PyTorch model | Medium | High | Quantize model về int8 hoặc serve baseline (LSTM nhỏ) làm default; full model cho local |
| Upload CSV với schema lạ làm crash | High | Medium | Validation chặt, list rõ 8 cột + dtype + sampling rate. Reject sớm với message rõ ràng. |
| PDF generation lỗi font Unicode tiếng Việt | Low | Low | App dùng tiếng Anh (A5), không cần Unicode đặc biệt |
| Patient có missing modality (HUPA0011 etc.) làm sai forecast | Medium | High | Đảm bảo preprocessor giống training (modality flag), test riêng 4 patient này ở tuần 6 |
| Bệnh nhân hiểu nhầm app là medical advice | High | **Very High** | Disclaimer banner cứng trên mọi page, mọi PDF; wording theo `skills/SKILL.md` rule 5 |

---

## 6. Out of scope (cố tình không làm)

Để tránh scope creep — undergrad thesis có thời hạn cứng.

- ❌ Mobile native app (iOS/Android) — Streamlit responsive là đủ
- ❌ Real-time CGM stream từ Libre/Dexcom API — chỉ playback + upload
- ❌ Multi-user authentication / cloud DB — session-state là đủ
- ❌ "What-if" simulator — rủi ro medical advice (đã loại ở vòng ý tưởng)
- ❌ Tự động đề xuất liều insulin — tuyệt đối không
- ❌ Đa ngôn ngữ — chỉ tiếng Anh
- ❌ Train model trong app — model train offline, app chỉ inference

---

## 7. Mapping app modules ↔ thesis contribution

Khi bảo vệ, mỗi module phải link rõ tới một mục trong báo cáo:

| Module | Trả lời câu hỏi của hội đồng | Mục trong thesis |
|---|---|---|
| Forecast Panel | "Mô hình có chạy được không?" | Chương 4 — Model architecture & training |
| Risk Radar | "Mô hình giúp gì trong thực tế?" | Chương 5 — Zone-specific & Parkes Error Grid |
| Why Explainer | "Mô hình giải thích được không, hay là black box?" | Chương 5 — XAI section |
| Pattern Reports | "Mô hình có dùng được bởi bác sĩ không?" | Chương 6 — Subgroup analysis, per-participant performance |
| Upload + HF deploy | "Có ai khác kiểm chứng được không?" | Reproducibility statement |

---

## 8. Câu hỏi mở cần Son confirm trước Tuần 1

1. **Model artefact format**: Son sẽ save `.pt` (PyTorch state_dict) hay `.onnx`? App load thế nào tùy lựa chọn này.
2. **Scaler**: sklearn `StandardScaler` pickle có ổn không, hay Son muốn custom min-max trong PyTorch? Cần đồng nhất với training pipeline.
3. **Probability head**: mô hình có một classification head riêng cho 3 zone, hay app phải tự derive probability từ regression output + RMSE? Cái thứ nhất gọn hơn, cái thứ hai linh hoạt hơn.
4. **HF Spaces hardware**: bắt đầu với CPU free tier, hay Son có budget cho T4 GPU? Ảnh hưởng tới việc có quantize model hay không.

Trả lời 4 câu này xong là có thể bắt đầu Tuần 1.
