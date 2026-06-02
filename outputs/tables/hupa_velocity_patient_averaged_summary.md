**Table 3.X. Absolute 5-minute glucose velocity quantiles under pooled and patient-level aggregation**

| Aggregation | P50 | P90 | P95 | P99 | Interpretation |
|---|---:|---:|---:|---:|---|
| Pooled intervals (row-weighted) | 1.67 | 6.67 | 9.00 | 17.00 | All 5-minute intervals are pooled; long-duration patients have more weight. |
| Mean of patient-level quantiles | 2.42 | 8.87 | 11.77 | 19.82 | Each patient contributes one quantile value; mean summarises the 25 patients. |
| Median of patient-level quantiles | 2.33 | 7.67 | 10.00 | 16.00 | Each patient contributes one quantile value; median is robust to outlier patients. |

*Note.* Values are in mg/dL per 5 minutes and are computed from `|glucose(t) - glucose(t-1)|` after sorting each patient's records by time.
