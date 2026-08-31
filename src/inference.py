"""Model-bundle loading and validated batch inference."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np


def load_bundle(path: str | Path) -> dict:
    return joblib.load(path)


def predict_records(bundle: dict, records: list[list[float]]) -> dict:
    matrix = np.asarray(records, dtype=float)
    expected = len(bundle["feature_names"])
    if matrix.ndim != 2 or matrix.shape[1] != expected:
        raise ValueError(f"Expected a 2D matrix with {expected} features per record")
    model = bundle["model"]
    class_ids = model.predict(matrix)
    probabilities = model.predict_proba(matrix)
    labels = [bundle["target_names"][int(class_id)] for class_id in class_ids]
    return {"predictions": labels, "probabilities": probabilities.tolist()}
