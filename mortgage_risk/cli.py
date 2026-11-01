import argparse
import json
from pathlib import Path
import platform
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
from sklearn.inspection import permutation_importance
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .data import synthetic_applications, validate_features, NUMERIC
from .modeling import candidate_models, preprocessing, evaluate, choose_threshold


def train(output, rows=4000, seed=42):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame = synthetic_applications(rows, seed)
    x, y = validate_features(frame), frame["default"]
    x_dev, x_test, y_dev, y_test = train_test_split(x, y, test_size=.2, stratify=y, random_state=seed)
    x_train, x_valid, y_train, y_valid = train_test_split(x_dev, y_dev, test_size=.25,
        stratify=y_dev, random_state=seed)
    eda = {"scope": "training partition only", "rows": len(x_train),
           "default_rate": float(y_train.mean()),
           "missing_fraction": {key: float(value) for key, value in x_train.isna().mean().items()},
           "numeric_summary": json.loads(x_train[NUMERIC].describe().to_json())}
    (output / "eda.json").write_text(json.dumps(eda, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, feature in zip(axes, ["credit_score", "debt_to_income", "loan_to_value"]):
        for label, color in [(0, "#327d8e"), (1, "#df8a45")]:
            ax.hist(x_train.loc[y_train == label, feature].dropna(), bins=22,
                    density=True, alpha=.55, label=f"default={label}", color=color)
        ax.set_title(feature); ax.set_ylabel("Density"); ax.legend(fontsize=8)
    fig.suptitle("Training-only exploratory analysis | synthetic applicants")
    fig.tight_layout(); fig.savefig(output / "eda.png", dpi=150); plt.close(fig)
    # Every transformer and estimator is fit exclusively on the training partition.
    models = candidate_models(seed)
    validation = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        validation[name] = evaluate(y_valid, model.predict_proba(x_valid)[:, 1])
    eligible = [name for name in models if name != "baseline"]
    chosen = max(eligible, key=lambda name: validation[name]["roc_auc"])
    selected = models[chosen]
    threshold, cost = choose_threshold(y_valid, selected.predict_proba(x_valid)[:, 1])
    # Selection is frozen before any holdout metrics are calculated. No test-based refit.
    test = {name: evaluate(y_test, model.predict_proba(x_test)[:, 1]) for name, model in models.items()}
    probabilities = selected.predict_proba(x_test)[:, 1]
    selected_test = evaluate(y_test, probabilities, threshold)
    report = {"data_source": "synthetic; no real borrower records", "seed": seed,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "rows": rows, "default_rate": float(y.mean()),
        "split_sizes": {"train": len(y_train), "validation": len(y_valid), "test": len(y_test)},
        "selection_rule": "highest validation ROC-AUC excluding baseline",
        "selected_model": chosen, "threshold": threshold,
        "cost_assumption": {"false_negative": 5, "false_positive": 1,
                            "validation_cost_per_application": cost},
        "validation_at_0_5": validation, "test_at_0_5": test,
        "selected_test_at_validation_threshold": selected_test}
    joblib.dump({"pipeline": selected, "threshold": threshold, "data_source": "synthetic"}, output / "model.joblib")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for name, model in models.items():
        p = model.predict_proba(x_test)[:, 1]
        RocCurveDisplay.from_predictions(y_test, p, name=name, ax=axes[0])
        PrecisionRecallDisplay.from_predictions(y_test, p, name=name, ax=axes[1])
    fig.suptitle("Synthetic holdout | selected using validation data only")
    fig.tight_layout(); fig.savefig(output / "evaluation.png", dpi=150); plt.close(fig)
    importance = permutation_importance(selected, x_valid, y_valid, scoring="roc_auc",
        n_repeats=5, random_state=seed, n_jobs=1)
    ranking = sorted(zip(x.columns, importance.importances_mean, importance.importances_std),
                     key=lambda row: row[1], reverse=True)
    report["validation_permutation_importance"] = [
        {"feature": name, "mean_auc_decrease": float(mean), "std": float(std)} for name, mean, std in ranking]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh([row[0] for row in ranking][::-1], [row[1] for row in ranking][::-1],
            xerr=[row[2] for row in ranking][::-1], color="#287d8e")
    ax.set_xlabel("Validation ROC-AUC decrease after permutation")
    ax.set_title("Predictive association, not causal explanation")
    fig.tight_layout(); fig.savefig(output / "feature_importance.png", dpi=150); plt.close(fig)
    # Segmentation is fit on training features without target labels.
    transform = preprocessing()
    z = transform.fit_transform(x_train)
    clusterer = KMeans(n_clusters=4, n_init=10, random_state=seed)
    clusters = clusterer.fit_predict(z)
    summaries = []
    for label in range(4):
        subset = x_train.iloc[np.flatnonzero(clusters == label)]
        summary = {"cluster": label, "count": len(subset),
                   "observed_synthetic_default_rate": float(y_train.iloc[np.flatnonzero(clusters == label)].mean())}
        summary.update({column: float(subset[column].median()) for column in NUMERIC})
        summaries.append(summary)
    report["training_segments"] = summaries
    # t-SNE is visualization only: distances/clusters are not evaluation metrics.
    sample = np.random.default_rng(seed).choice(len(z), min(800, len(z)), replace=False)
    embedding = TSNE(n_components=2, perplexity=min(30, (len(sample)-1)/3),
        init="pca", learning_rate="auto", random_state=seed, max_iter=750).fit_transform(z[sample])
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=clusters[sample], cmap="viridis", s=13, alpha=.7)
    ax.set_title("Training applicants | illustrative t-SNE view of K-means segments")
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    fig.colorbar(scatter, ax=ax, label="Cluster ID (not a risk score)")
    fig.tight_layout(); fig.savefig(output / "segments.png", dpi=150); plt.close(fig)
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"selected_model": chosen, "threshold": threshold,
                      "holdout": selected_test, "output": str(output)}, indent=2))
    return report


def predict(model_path, input_path, output_path):
    # Only load artifacts you created or trust; joblib can execute code.
    bundle = joblib.load(model_path)
    frame = pd.read_csv(input_path)
    probabilities = bundle["pipeline"].predict_proba(validate_features(frame))[:, 1]
    result = pd.DataFrame({"synthetic_default_probability": probabilities,
        "review_flag": probabilities >= bundle["threshold"]})
    result.to_csv(output_path, index=False)
    print(f"Saved {len(result)} predictions to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Synthetic mortgage default modeling demo")
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("train")
    fit.add_argument("--output", default="reports")
    fit.add_argument("--rows", type=int, default=4000)
    fit.add_argument("--seed", type=int, default=42)
    score = sub.add_parser("predict")
    score.add_argument("--model", default="reports/model.joblib")
    score.add_argument("--input", required=True)
    score.add_argument("--output", default="predictions.csv")
    args = parser.parse_args()
    if args.command == "train":
        if args.rows < 1000: parser.error("Use at least 1000 rows for stable stratified partitions")
        train(args.output, args.rows, args.seed)
    else:
        predict(args.model, args.input, args.output)


if __name__ == "__main__":
    main()
