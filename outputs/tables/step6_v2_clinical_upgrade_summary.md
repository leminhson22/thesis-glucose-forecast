# Step 6 v2 Clinical CG-EGA Upgrade Summary

## Six-model CG-EGA comparison with upgraded proposed model
| model                        |   horizon_min |   AP_pct |   BE_pct |   EP_pct |   n_total |
|:-----------------------------|--------------:|---------:|---------:|---------:|----------:|
| persistence                  |            30 |    91.77 |     5.73 |     2.50 |     45320 |
| ridge_a0.1                   |            30 |    87.11 |     9.90 |     2.99 |     45320 |
| rf_n300                      |            30 |    87.26 |     8.10 |     4.64 |     45320 |
| gbm_n300                     |            30 |    89.06 |     7.75 |     3.20 |     45320 |
| gru_c2_zwh30a                |            30 |    83.87 |    13.06 |     3.06 |     45320 |
| step6_v2_pers_resid_clinical |            30 |    88.50 |     9.85 |     1.65 |     45320 |
| persistence                  |            60 |    85.40 |     9.45 |     5.15 |     45320 |
| ridge_a0.1                   |            60 |    79.28 |    14.56 |     6.16 |     45320 |
| rf_n300                      |            60 |    78.14 |    13.22 |     8.64 |     45320 |
| gbm_n300                     |            60 |    79.17 |    12.60 |     8.23 |     45320 |
| gru_c2_zwh30a                |            60 |    78.03 |    16.43 |     5.54 |     45320 |
| step6_v2_pers_resid_clinical |            60 |    79.46 |    16.68 |     3.86 |     45320 |
| persistence                  |            90 |    81.89 |    11.65 |     6.46 |     45320 |
| ridge_a0.1                   |            90 |    76.19 |    15.39 |     8.42 |     45320 |
| rf_n300                      |            90 |    75.60 |    14.41 |     9.99 |     45320 |
| gbm_n300                     |            90 |    76.95 |    13.38 |     9.67 |     45320 |
| gru_c2_zwh30a                |            90 |    76.01 |    16.80 |     7.18 |     45320 |
| step6_v2_pers_resid_clinical |            90 |    76.86 |    17.71 |     5.44 |     45320 |

## Upgrade ablation: old proposed vs CG-EGA checkpoint vs clinical
| model                          |   horizon_min |   AP_pct |   BE_pct |   EP_pct |   n_total |
|:-------------------------------|--------------:|---------:|---------:|---------:|----------:|
| step6_v2_pers_resid            |            30 |    90.13 |     7.37 |     2.50 |     45320 |
| step6_v2_pers_resid            |            60 |    80.46 |    13.63 |     5.91 |     45320 |
| step6_v2_pers_resid            |            90 |    75.92 |    15.93 |     8.15 |     45320 |
| step6_v2_pers_resid_cgega_ckpt |            30 |    89.54 |     8.10 |     2.36 |     45320 |
| step6_v2_pers_resid_cgega_ckpt |            60 |    79.64 |    14.96 |     5.39 |     45320 |
| step6_v2_pers_resid_cgega_ckpt |            90 |    75.87 |    16.52 |     7.61 |     45320 |
| step6_v2_pers_resid_clinical   |            30 |    88.50 |     9.85 |     1.65 |     45320 |
| step6_v2_pers_resid_clinical   |            60 |    79.46 |    16.68 |     3.86 |     45320 |
| step6_v2_pers_resid_clinical   |            90 |    76.86 |    17.71 |     5.44 |     45320 |

## Clinical variant by glycaemic zone
|   horizon_min | glycaemic_zone   |   AP_pct |   BE_pct |   EP_pct |   n_total |
|--------------:|:-----------------|---------:|---------:|---------:|----------:|
|            30 | hyper            |    90.91 |     7.47 |     1.62 |      9275 |
|            30 | hypo             |    94.40 |     0.14 |     5.46 |      3625 |
|            30 | tir              |    87.15 |    11.62 |     1.24 |     32420 |
|            60 | hyper            |    83.44 |    11.61 |     4.96 |      9262 |
|            60 | hypo             |    84.09 |     0.14 |    15.77 |      3620 |
|            60 | tir              |    77.81 |    19.98 |     2.22 |     32438 |
|            90 | hyper            |    78.81 |    13.58 |     7.61 |      9251 |
|            90 | hypo             |    72.95 |     0.00 |    27.05 |      3619 |
|            90 | tir              |    76.73 |    20.86 |     2.41 |     32450 |

## MAE/RMSE trade-off across proposed variants
| variant_label   | model                          |   horizon_min |   mae |   rmse |   mae_pat_avg |   clarke_pct_A |   clarke_pct_D |
|:----------------|:-------------------------------|--------------:|------:|-------:|--------------:|---------------:|---------------:|
| old_pers_resid  | step6_hybrid_v2_pers_resid     |            30 | 10.56 |  15.43 |         12.01 |          91.37 |           1.16 |
| old_pers_resid  | step6_hybrid_v2_pers_resid     |            60 | 19.52 |  27.60 |         23.01 |          74.65 |           3.62 |
| old_pers_resid  | step6_hybrid_v2_pers_resid     |            90 | 26.10 |  36.18 |         31.62 |          63.50 |           5.80 |
| cgega_ckpt      | step6_v2_pers_resid_cgega_ckpt |            30 | 10.96 |  15.65 |         12.26 |          91.39 |           0.96 |
| cgega_ckpt      | step6_v2_pers_resid_cgega_ckpt |            60 | 19.60 |  27.84 |         22.85 |          75.37 |           2.95 |
| cgega_ckpt      | step6_v2_pers_resid_cgega_ckpt |            90 | 25.91 |  36.29 |         31.12 |          64.23 |           5.17 |
| clinical        | step6_v2_pers_resid_clinical   |            30 | 14.00 |  19.43 |         15.04 |          85.05 |           0.38 |
| clinical        | step6_v2_pers_resid_clinical   |            60 | 25.98 |  35.40 |         28.26 |          61.96 |           1.59 |
| clinical        | step6_v2_pers_resid_clinical   |            90 | 33.81 |  45.61 |         37.68 |          50.04 |           3.28 |