import numpy as np
import pandas as pd

NUMERIC = ["annual_income", "loan_amount", "credit_score", "debt_to_income",
           "loan_to_value", "employment_years", "interest_rate", "prior_delinquencies"]
CATEGORICAL = ["loan_purpose", "property_type"]
FEATURES = NUMERIC + CATEGORICAL


def validate_features(frame):
    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise ValueError("Missing features: " + ", ".join(sorted(missing)))
    if frame.empty:
        raise ValueError("Applications cannot be empty")
    data = frame[FEATURES].copy()
    for column in NUMERIC:
        data[column] = pd.to_numeric(data[column], errors="raise")
        if np.isinf(data[column].to_numpy(dtype=float)).any():
            raise ValueError("Infinite numeric values are not supported")
    for column in CATEGORICAL:
        data[column] = data[column].astype(object).where(data[column].notna(), np.nan)
        if not data[column].dropna().map(lambda value: isinstance(value, str)).all():
            raise ValueError("Categorical values must be strings or missing")
    return data


def synthetic_applications(rows=4000, seed=42):
    """Generate fictional applicants, labels, and missing values; no real people."""
    if rows < 100:
        raise ValueError("At least 100 rows are required")
    rng = np.random.default_rng(seed)
    income = np.clip(rng.lognormal(np.log(85000), .5, rows), 22000, 500000)
    score = np.clip(rng.normal(690, 65, rows), 400, 850)
    dti = np.clip(rng.beta(3, 6, rows) * .9, .02, .85)
    ltv = rng.uniform(.35, 1.08, rows)
    employment = np.clip(rng.exponential(6, rows), 0, 35)
    rate = rng.uniform(.025, .105, rows)
    delinquency = rng.poisson(.35, rows)
    amount = income * rng.uniform(1.2, 5.8, rows)
    purpose = rng.choice(["purchase", "refinance", "cash_out"], rows, p=[.6, .25, .15])
    property_type = rng.choice(["house", "condo", "townhouse"], rows, p=[.65, .2, .15])
    logit = (-2.7 + (660 - score) / 85 + 4 * (dti - .3)
             + 3 * np.maximum(ltv - .75, 0) + .7 * delinquency
             + .15 * (amount / income - 3) + 6 * (rate - .055)
             - .025 * employment + .35 * (purpose == "cash_out"))
    probability = 1 / (1 + np.exp(-logit))
    target = rng.binomial(1, probability)
    frame = pd.DataFrame(dict(annual_income=income, loan_amount=amount,
        credit_score=score, debt_to_income=dti, loan_to_value=ltv,
        employment_years=employment, interest_rate=rate,
        prior_delinquencies=delinquency, loan_purpose=purpose,
        property_type=property_type))
    # Labels depend on latent complete values; missingness simulates collection gaps.
    for column in NUMERIC:
        frame.loc[rng.random(rows) < .025, column] = np.nan
    for column in CATEGORICAL:
        frame.loc[rng.random(rows) < .015, column] = np.nan
    frame["default"] = target
    return frame
