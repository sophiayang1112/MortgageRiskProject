import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
    precision_score, recall_score, f1_score, confusion_matrix)
from .data import NUMERIC, CATEGORICAL


def preprocessing():
    numeric = Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("numeric", numeric, NUMERIC), ("categorical", categorical, CATEGORICAL)])


def candidate_models(seed=42):
    estimators = {
        "baseline": DummyClassifier(strategy="prior"),
        "logistic": LogisticRegression(max_iter=1500, random_state=seed),
        "cart": DecisionTreeClassifier(max_depth=5, min_samples_leaf=30, random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=160, max_depth=8,
            min_samples_leaf=12, random_state=seed, n_jobs=2),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=100,
            max_depth=2, learning_rate=.06, random_state=seed),
    }
    models = {name: Pipeline([("preprocess", preprocessing()), ("model", estimator)])
              for name, estimator in estimators.items()}
    models["pca_logistic"] = Pipeline([("preprocess", preprocessing()),
        ("pca", PCA(n_components=.95, svd_solver="full")),
        ("model", LogisticRegression(max_iter=1500, random_state=seed))])
    return models


def choose_threshold(y, probabilities, false_negative_cost=5, false_positive_cost=1):
    """Minimize a declared illustrative cost using validation labels only."""
    thresholds = np.linspace(0, 1, 201)
    costs = []
    for threshold in thresholds:
        tn, fp, fn, tp = confusion_matrix(y, probabilities >= threshold, labels=[0, 1]).ravel()
        costs.append((false_negative_cost * fn + false_positive_cost * fp) / len(y))
    selected = int(np.argmin(costs))
    return float(thresholds[selected]), float(costs[selected])


def evaluate(y, probabilities, threshold=.5):
    predicted = probabilities >= threshold
    return {"roc_auc": float(roc_auc_score(y, probabilities)),
        "pr_auc_average_precision": float(average_precision_score(y, probabilities)),
        "brier_score": float(brier_score_loss(y, probabilities)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist()}
