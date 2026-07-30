# Card visual features EDA

Labeled crops: 162 informative ("matched") vs 161 confound ("no_relevant_text"), from data/processed/card_crops/card_text_matches.json ("skipped" entries excluded -- bad detections, not a relevance judgment).

## Cross-validated performance (5-fold, same labeled subset for all rows)

| classifier | accuracy | precision | recall |
|---|---|---|---|
| keyword_baseline | 0.505 | 0.505 | 0.642 |
| visual_logistic_regression | 0.833 | 0.818 | 0.858 |
| visual_random_forest | 0.870 | 0.853 | 0.895 |
| combined_random_forest | 0.848 | 0.834 | 0.870 |

`keyword_baseline` is the existing `classify_card_relevance.is_relevant_card` (no cross-validation needed, it has no fitted parameters). The rest are 5-fold cross-validated on `data/processed/card_crops/visual_features.csv`.

## Feature importance (random forest, fit on all labeled crops)

- mean_saturation: 0.166
- tint_fraction: 0.113
- border_contrast: 0.096
- mean_hue: 0.089
- std_saturation: 0.085
- border_ink_fraction: 0.070
- fft_highfreq_ratio: 0.065
- mean_value: 0.061
- std_value: 0.061
- edge_density: 0.061
- aspect_ratio: 0.052
- interior_ink_fraction: 0.048
- laplacian_var: 0.034

Distributions for the top features: `feature_distributions.png`
