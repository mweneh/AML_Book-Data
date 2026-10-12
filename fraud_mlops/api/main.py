"""Fraud-detection API. Serves the production model exported by train.py.

Run locally:  uvicorn api.main:app --reload     ->  http://127.0.0.1:8000/docs
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_DIR = Path(__file__).parent.parent / "model"

model    = joblib.load(MODEL_DIR / "model.joblib")
metadata = json.loads((MODEL_DIR / "metadata.json").read_text())

app = FastAPI(title="Fraud Detection API", version="1.0")


class Transaction(BaseModel):
    """One mobile-money transaction. The example below is a real row from the dataset."""
    abs_amount:    float = Field(examples=[1329.0])
    log_amount:    float = Field(examples=[7.1929])
    is_debit:      int   = Field(examples=[1])
    balance_after: float = Field(examples=[8123.0])
    log_balance:   float = Field(examples=[9.0026])
    risk_score:    float = Field(examples=[0.26])
    hour:          int   = Field(examples=[14])
    day_of_week:   int   = Field(examples=[1])
    day_of_month:  int   = Field(examples=[30])
    month:         int   = Field(examples=[6])
    is_weekend:    int   = Field(examples=[0])
    is_night:      int   = Field(examples=[0])
    has_gps:       int   = Field(examples=[0])
    gps_lat:       float = Field(examples=[0.0])
    gps_lon:       float = Field(examples=[0.0])
    txn_type_enc:  int   = Field(examples=[6])
    region_enc:    int   = Field(examples=[4])
    segment_enc:   int   = Field(examples=[2])


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    """Which model is deployed and how it scored in MLflow."""
    return metadata


@app.post("/predict")
def predict(txn: Transaction):
    X = pd.DataFrame([txn.model_dump()])[metadata["features"]]
    probability = float(model.predict_proba(X)[0, 1])
    return {
        "is_fraud":          int(probability >= 0.5),
        "fraud_probability": round(probability, 4),
        "model_version":     metadata["version"],
    }
