"""Reproducible RF, GA-RF and XGBoost classification benchmark.

Uses the repository crop classification data only. It does not claim yield
prediction or external ICAR/Eurostat dataset integration.
"""
import itertools
import json
import random
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "crop_data.csv"
OUT = ROOT / "models"
SEED = 42


def genetic_search(X, y):
    """Small seeded GA over RF hyperparameters, scored by 3-fold CV accuracy."""
    rng = random.Random(SEED)
    grid = list(itertools.product((100, 160), (None, 18), ("sqrt", 0.8), (1, 2)))
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    cache = {}

    def score(individual):
        if individual not in cache:
            trees, depth, features, leaf = individual
            estimator = RandomForestClassifier(
                n_estimators=trees, max_depth=depth, max_features=features,
                min_samples_leaf=leaf, random_state=SEED, n_jobs=-1,
            )
            cache[individual] = float(cross_val_score(
                estimator, X, y, cv=cv, scoring="accuracy", n_jobs=1
            ).mean())
        return cache[individual]

    population = rng.sample(grid, 6)
    for _ in range(3):
        ranked = sorted(population, key=score, reverse=True)
        next_population = ranked[:2]
        while len(next_population) < 6:
            p1, p2 = rng.sample(ranked[:4], 2)
            child = tuple(rng.choice(pair) for pair in zip(p1, p2))
            if rng.random() < 0.25:
                child = rng.choice(grid)
            next_population.append(child)
        population = list(dict.fromkeys(next_population))
        while len(population) < 6:
            population.append(rng.choice(grid))
    best = max(cache, key=cache.get)
    return best, cache[best], len(cache)


def measure(name, estimator, X_train, X_test, y_train, y_test, labels):
    started = time.perf_counter()
    estimator.fit(X_train, y_train)
    train_seconds = time.perf_counter() - started
    started = time.perf_counter()
    pred = estimator.predict(X_test)
    test_seconds = time.perf_counter() - started
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, pred, average="weighted", zero_division=0
    )
    row = {
        "model": name,
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "train_seconds": round(train_seconds, 3),
        "test_seconds": round(test_seconds, 4),
        "test_samples": int(len(y_test)),
        "split": "stratified 80/20, random_state=42",
    }
    return row, estimator, pred


def main():
    frame = pd.read_csv(DATA)
    target = "label" if "label" in frame.columns else frame.columns[-1]
    X = frame.drop(columns=[target])
    encoder = LabelEncoder()
    y = encoder.fit_transform(frame[target])
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    # Both compared estimators are tree models and do not require feature scaling.
    X_train = X_train_raw.to_numpy()
    X_test = X_test_raw.to_numpy()

    best, cv_score, candidates = genetic_search(X_train, y_train)
    trees, depth, features, leaf = best
    estimators = [
        ("Random Forest", RandomForestClassifier(
            n_estimators=100, random_state=SEED, n_jobs=-1
        )),
        ("GA-Random Forest", RandomForestClassifier(
            n_estimators=trees, max_depth=depth, max_features=features,
            min_samples_leaf=leaf, random_state=SEED, n_jobs=-1
        )),
        ("XGBoost", XGBClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9, objective="multi:softprob",
            num_class=len(encoder.classes_), eval_metric="mlogloss",
            random_state=SEED, n_jobs=2, tree_method="hist",
        )),
    ]

    metrics, fitted, predictions = [], {}, {}
    for name, estimator in estimators:
        result, estimator, pred = measure(
            name, estimator, X_train, X_test, y_train, y_test, encoder.classes_
        )
        if name == "GA-Random Forest":
            result["ga_3fold_cv_accuracy"] = cv_score
            result["ga_candidates_evaluated"] = candidates
            result["ga_parameters"] = {
                "n_estimators": trees, "max_depth": depth,
                "max_features": features, "min_samples_leaf": leaf,
            }
        metrics.append(result)
        fitted[name] = estimator
        predictions[name] = pred
        print(f"{name}: accuracy={result['accuracy']:.4f}, weighted F1={result['f1_weighted']:.4f}")

    OUT.mkdir(exist_ok=True)
    try:
        import shap
        sample_count = min(150, len(X_test))
        explainer = shap.TreeExplainer(fitted["GA-Random Forest"])
        shap_output = explainer.shap_values(X_test[:sample_count])
        if isinstance(shap_output, list):
            importance = np.mean(np.abs(np.stack(shap_output, axis=-1)), axis=(0, 2))
        else:
            shap_array = np.asarray(shap_output)
            importance = (
                np.mean(np.abs(shap_array), axis=(0, 2))
                if shap_array.ndim == 3 else np.mean(np.abs(shap_array), axis=0)
            )
        pd.DataFrame({"feature": list(X.columns), "mean_abs_shap": importance}) \
            .sort_values("mean_abs_shap", ascending=False) \
            .to_csv(OUT / "ga_rf_shap_global.csv", index=False)
        print(f"Saved global SHAP summary using {sample_count} held-out samples.")
    except Exception as exc:
        print(f"Global SHAP summary was skipped: {exc}")

    (OUT / "benchmark_results.json").write_text(json.dumps({
        "dataset": str(DATA.relative_to(ROOT)),
        "rows": int(len(frame)), "features": list(X.columns),
        "classes": encoder.classes_.tolist(), "seed": SEED,
        "metrics": metrics,
    }, indent=2))
    joblib.dump({"models": fitted, "label_encoder": encoder,
                 "feature_names": list(X.columns)}, OUT / "benchmark_models.joblib")
    for name, pred in predictions.items():
        slug = name.lower().replace("-", "_").replace(" ", "_")
        cm = confusion_matrix(y_test, pred)
        pd.DataFrame(cm, index=encoder.classes_, columns=encoder.classes_).to_csv(
            OUT / f"{slug}_confusion_matrix.csv", index_label="actual_crop"
        )
        pd.DataFrame(classification_report(
            y_test, pred, target_names=encoder.classes_,
            output_dict=True, zero_division=0,
        )).T.to_csv(OUT / f"{slug}_classification_report.csv")
    print(f"Saved benchmark artefacts in {OUT}")


if __name__ == "__main__":
    main()
