import pytest

from src.inference import predict_records
from src.train import train_model


@pytest.fixture(scope="module")
def bundle() -> dict:
    return train_model()


def test_training_reaches_reasonable_accuracy(bundle: dict) -> None:
    assert bundle["metrics"]["accuracy"] >= 0.85
    assert bundle["metrics"]["macro_roc_auc"] >= 0.90


def test_batch_inference_returns_probabilities(bundle: dict) -> None:
    sample = [[13.2, 2.7, 2.5, 18.5, 99, 2.6, 2.3, 0.3, 1.8, 5.2, 1.0, 3.0, 1050]]
    result = predict_records(bundle, sample)
    assert len(result["predictions"]) == 1
    assert sum(result["probabilities"][0]) == pytest.approx(1.0)


def test_invalid_feature_count_is_rejected(bundle: dict) -> None:
    with pytest.raises(ValueError):
        predict_records(bundle, [[1.0, 2.0]])

