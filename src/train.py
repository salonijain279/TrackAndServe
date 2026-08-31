"""Train, evaluate, track, and package a reproducible classifier."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import joblib
from sklearn.datasets import load_wine
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split


def train_model(random_state: int = 42) -> dict:
    dataset = load_wine()
    x_train, x_test, y_train, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=0.25,
        random_state=random_state,
        stratify=dataset.target,
    )
    model = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=2,
        random_state=random_state,
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_roc_auc": float(roc_auc_score(y_test, probabilities, multi_class="ovr", average="macro")),
    }
    return {
        "model": model,
        "feature_names": list(dataset.feature_names),
        "target_names": list(dataset.target_names),
        "metrics": metrics,
        "random_state": random_state,
    }


def save_bundle(bundle: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/model.joblib"))
    parser.add_argument("--disable-mlflow", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mlflow = None
    if not args.disable_mlflow:
        try:
            import mlflow as mlflow_module

            mlflow = mlflow_module
        except ImportError:
            print("MLflow is unavailable; training will continue without tracking.")

    run_context = mlflow.start_run(run_name="wine_gradient_boosting") if mlflow else nullcontext()
    with run_context:
        bundle = train_model()
        save_bundle(bundle, args.output)
        if mlflow:
            model = bundle["model"]
            mlflow.log_params(
                {
                    "n_estimators": model.n_estimators,
                    "learning_rate": model.learning_rate,
                    "max_depth": model.max_depth,
                    "random_state": bundle["random_state"],
                }
            )
            mlflow.log_metrics(bundle["metrics"])
            mlflow.log_artifact(str(args.output))
        print(bundle["metrics"])


if __name__ == "__main__":
    main()
