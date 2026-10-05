from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Load the model once, when the server starts
MODEL_PATH = Path(__file__).parent.parent / "model" / "fraud_model.joblib"
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
FEATURES = bundle["features"]

app = FastAPI(title="Fraud Detection API")


# The shape of one transaction sent to the API (FastAPI validates it for us)
class Transaction(BaseModel):
    abs_amount: float
    log_amount: float
    is_debit: int
    balance_after: float
    log_balance: float
    risk_score: float
    hour: int
    day_of_week: int
    day_of_month: int
    month: int
    is_weekend: int
    is_night: int
    has_gps: int
    gps_lat: float
    gps_lon: float
    txn_type_enc: int
    region_enc: int
    segment_enc: int

    # Example shown in the /docs page
    model_config = {
        "json_schema_extra": {
            "example": {
                "abs_amount": 1571.0, "log_amount": 7.36, "is_debit": 0,
                "balance_after": 15237.0, "log_balance": 9.63, "risk_score": 1.0,
                "hour": 0, "day_of_week": 1, "day_of_month": 16, "month": 9,
                "is_weekend": 0, "is_night": 1, "has_gps": 1,
                "gps_lat": -3.74336, "gps_lon": 33.67018,
                "txn_type_enc": 5, "region_enc": 0, "segment_enc": 2,
            }
        }
    }


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(txn: Transaction):
    # Turn the request into a 1-row DataFrame with columns in training order
    row = pd.DataFrame([txn.model_dump()])[FEATURES]
    prob = float(model.predict_proba(row)[0, 1])
    return {"is_fraud": int(prob >= 0.5), "fraud_probability": round(prob, 4)}
