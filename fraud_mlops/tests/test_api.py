from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

TRANSACTION = {
    "abs_amount": 1329.0, "log_amount": 7.1929, "is_debit": 1, "balance_after": 8123.0,
    "log_balance": 9.0026, "risk_score": 0.26, "hour": 14, "day_of_week": 1, "day_of_month": 30,
    "month": 6, "is_weekend": 0, "is_night": 0, "has_gps": 0, "gps_lat": 0.0, "gps_lon": 0.0,
    "txn_type_enc": 6, "region_enc": 4, "segment_enc": 2,
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_info():
    info = client.get("/model-info").json()
    assert info["model_name"] == "fraud-detector"
    assert info["metrics"]["f1_score"] >= 0.50


def test_predict_returns_valid_answer():
    result = client.post("/predict", json=TRANSACTION).json()
    assert result["is_fraud"] in (0, 1)
    assert 0.0 <= result["fraud_probability"] <= 1.0


def test_predict_rejects_missing_fields():
    response = client.post("/predict", json={"abs_amount": 100})
    assert response.status_code == 422
