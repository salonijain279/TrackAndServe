"""FastAPI prediction service."""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference import load_bundle, predict_records


MODEL_PATH = Path("artifacts/model.joblib")
app = FastAPI(title="Tracked Wine Classifier", version="1.0.0")
bundle = load_bundle(MODEL_PATH)


class PredictionRequest(BaseModel):
    records: list[list[float]] = Field(min_length=1)


class PredictionResponse(BaseModel):
    predictions: list[str]
    probabilities: list[list[float]]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> dict:
    try:
        return predict_records(bundle, request.records)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

