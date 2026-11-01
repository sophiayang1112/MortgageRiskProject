# Methodology and model boundaries

## Data and outcome

All records are generated locally from a fixed random seed. There are no names, account numbers, employers, locations, or protected demographic attributes. A binary label is sampled from a nonlinear probability function involving credit score, affordability, leverage, delinquencies, interest rate, employment and purpose. Missingness is injected afterward. This is a simulation without an empirically defined real-world default horizon.

| Feature | Interpretation |
| :--- | :--- |
| annual_income | Fictional yearly income in USD |
| loan_amount | Fictional principal in USD |
| credit_score | Simulated score, 400-850 |
| debt_to_income | Ratio, expressed as a fraction |
| loan_to_value | Ratio, expressed as a fraction |
| employment_years | Simulated employment tenure |
| interest_rate | Annual rate as a fraction |
| prior_delinquencies | Simulated count |
| loan_purpose | purchase, refinance, cash_out |
| property_type | house, condo, townhouse |
| default | Synthetic binary target; excluded from predictors |

## Leakage prevention

A seeded, stratified split reserves 20% for test, 20% for validation and 60% for training. Imputers, scalers, encoders, PCA, and estimators are fit only on training data. The model is selected by validation ROC-AUC; a 201-point threshold grid minimizes a declared validation cost. Selection is frozen before test scoring, and no refit is performed. All candidate holdout scores are reported for transparency, not for selecting the winning model.

The false-negative cost of 5 and false-positive cost of 1 are illustrative, not estimated economic losses. A borrower-risk decision requires exposure, recovery, timing, operational capacity, and policy constraints that are absent here. The chosen threshold can be unstable on small validation samples.

ROC-AUC summarizes ranking, average precision summarizes precision/recall performance relative to class prevalence, and Brier score measures probability error. Reported probability estimates are not separately calibrated on an independent calibration sample. Confusion matrices use rows [actual 0, actual 1] and columns [predicted 0, predicted 1].

## Segmentation and interpretation

K-means uses transformed training features and four fixed clusters. Default labels are only summarized after segmentation, never passed to clustering. Cluster IDs are arbitrary and are not credit-risk tiers. t-SNE displays up to 800 training records; it is not an out-of-sample transformation or proof of cluster quality. Permutation importance is measured on validation data, may dilute importance among correlated features, and does not establish causal effects.

## Reproducibility and limitations

The seed and dependency versions are saved in metrics.json. A tested environment is recorded in requirements-lock.txt. Timing and exact floating-point results can vary by platform. A random synthetic holdout is not an out-of-time evaluation; there is no temporal leakage claim for real lending data. The current project does not estimate fairness, drift, confidence intervals, policy impact, or regulatory suitability.
