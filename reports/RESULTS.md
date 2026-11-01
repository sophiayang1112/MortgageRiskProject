# Measured results on synthetic data

Generated from 4,000 fictional applications with seed 42. No real borrower data or historical project results are represented.

Split: 2400 training / 800 validation / 800 holdout. Overall simulated default rate: 11.0%.

| Model | Validation ROC-AUC | Holdout ROC-AUC | Holdout PR-AUC | Holdout Brier |
| :--- | ---: | ---: | ---: | ---: |
| baseline | 0.500 | 0.500 | 0.110 | 0.098 |
| logistic | 0.749 | 0.794 | 0.309 | 0.087 |
| cart | 0.645 | 0.683 | 0.226 | 0.095 |
| random_forest | 0.712 | 0.778 | 0.297 | 0.089 |
| gradient_boosting | 0.718 | 0.765 | 0.291 | 0.090 |
| pca_logistic | 0.747 | 0.786 | 0.312 | 0.087 |

Selected using validation ROC-AUC: **logistic**. Threshold: **0.15**, chosen using validation data and an illustrative 5:1 false-negative/false-positive cost.

At that threshold, holdout precision = 0.277, recall = 0.557, and F1 = 0.370.

These are modest, simulation-specific results. They do not establish performance, calibration, fairness, or financial value on real lending data.

## Verification

- Six automated tests passed: generation reproducibility, schema validation, all candidate models with missing/unknown values, train-only imputation, threshold cost policy, and serialization consistency.
- Editable package installation and the installed prediction command succeeded.
- Both fictional sample applications produced finite probabilities in [0, 1].
- Exploratory analysis, model comparison, feature importance, and segmentation charts were generated from the executed pipeline.
