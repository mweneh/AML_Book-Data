# Fraud Detection: End-to-End MLOps

A complete machine-learning project, from data to a running service:

| Stage | Tool | File |
|---|---|---|
| 1. Solve the ML problem | scikit-learn | [`train.py`](train.py) |
| 2. Track & monitor experiments | **MLflow** | [`train.py`](train.py) |
| 3. Deploy as a web service | **FastAPI** | [`api/main.py`](api/main.py) |
| 4. Package it | **Docker** | [`Dockerfile`](Dockerfile) |
| 5. Automate everything | **GitHub Actions (CI/CD)** | [`../.github/workflows/fraud-mlops.yml`](../.github/workflows/fraud-mlops.yml) |

---

## The problem

**Predict whether a mobile-money transaction is fraud.**

- Data: [`Data/CleanedFeaturesFromLoan.csv`](../Data/CleanedFeaturesFromLoan.csv), with 183,600 transactions and 18 features (amount, balance, hour, region, GPS, risk score, …).
- Target: `is_fraud` (1 = fraud, 0 = genuine). Only **~8%** of transactions are fraud, so the classes are imbalanced.
- Because of that, we judge models by **F1-score** (a balance of catching fraud vs. false alarms), not accuracy. A model that always says "not fraud" would score 92% accuracy and be useless.

---

## How it fits together

```
                ┌──────────────────────────── CI (every push / PR) ───────────────────────────┐
 git push ───►  │ train.py: train 4 models ─► MLflow logs params + metrics + models            │
                │           best F1 ≥ 0.50 ? ── no ──► ❌ stop                                  │
                │           register "production" in MLflow ─► export model/model.joblib       │
                │ pytest: test the FastAPI endpoints                                           │
                │ docker build ─► run container ─► curl /health   (does the image really work?) │
                └──────────────────────────────────────┬──────────────────────────────────────┘
                                                       │ only if CI passed AND branch = master
                ┌─────────────────────────────── CD ───▼──────────────────────────────────────┐
                │ docker build ─► docker push ghcr.io/nuelbundi/fraud-api:latest               │
                └──────────────────────────────────────┬──────────────────────────────────────┘
                                                       ▼
                         anyone can run:  docker run -p 8000:8000 ghcr.io/nuelbundi/fraud-api
```

---

## 1. Training + MLflow ([`train.py`](train.py))

| Model | Parameters |
|---|---|
| Logistic Regression (with scaling) | `C = 1.0` |
| Decision Tree | `max_depth = 8` |
| Random Forest | 100 trees, depth 10 |
| Random Forest | 200 trees, depth 15 |

All use `class_weight="balanced"` to handle the rare fraud class.

**What MLflow records for every model:** parameters, ROC-AUC, precision, recall, F1-score, and the saved model.

Then the script:
1. Picks the model with the best F1-score.
2. Applies the **quality gate**: if F1 < 0.50, it exits with an error and nothing is deployed.
3. Registers the model in the **MLflow Model Registry** as `fraud-detector` with the alias `production`.
4. Exports it to `model/model.joblib`, plus `model/metadata.json` (version, metrics, feature list) for the API.

**Monitor the experiments:**
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db     # http://127.0.0.1:5000
```
Open **fraud-detection**, select the runs and click **Compare** to see the models side by side. Open **Models → fraud-detector** to see each version and which one is `production`.

---

## 2. The API ([`api/main.py`](api/main.py))

| Endpoint | Method | What it does |
|---|---|---|
| `/health` | GET | Returns `{"status": "ok"}`. Used to check the service is alive |
| `/model-info` | GET | Which model version is deployed and its MLflow metrics |
| `/predict` | POST | Send one transaction, get back `is_fraud` and `fraud_probability` |
| `/docs` | GET | Interactive page to try the API in the browser |

Example request:
```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "abs_amount": 1329.0, "log_amount": 7.1929, "is_debit": 1, "balance_after": 8123.0,
  "log_balance": 9.0026, "risk_score": 0.26, "hour": 14, "day_of_week": 1, "day_of_month": 30,
  "month": 6, "is_weekend": 0, "is_night": 0, "has_gps": 0, "gps_lat": 0.0, "gps_lon": 0.0,
  "txn_type_enc": 6, "region_enc": 4, "segment_enc": 2 }'
```
Example response:
```json
{"is_fraud": 0, "fraud_probability": 0.1245, "model_version": 1}
```

The API is tested in [`tests/test_api.py`](tests/test_api.py): health check, model info, a valid prediction, and rejecting a request with missing fields (HTTP 422).

---

## 3. Docker ([`Dockerfile`](Dockerfile))

The image contains **only what is needed to serve predictions**: Python, the packages in [`requirements-api.txt`](requirements-api.txt), the `api/` code and the `model/` folder. MLflow, the training data and the tests are left out, which keeps the image small.

```bash
python train.py                          # creates model/ first
docker build -t fraud-api .
docker run -p 8000:8000 fraud-api        # http://127.0.0.1:8000/docs  (port busy? use -p 8001:8000)
```

---

## 4. CI/CD ([`fraud-mlops.yml`](../.github/workflows/fraud-mlops.yml))

**CI runs on every push and pull request** that changes this folder or the dataset:

| Step | Fails if… |
|---|---|
| `python train.py` | the best model's F1 < 0.50 |
| `pytest` | any API test fails |
| `docker build` | the image cannot be built |
| run the container + `curl /health` | the container starts but the API doesn't answer |

**CD runs only on `master`, and only if CI passed.** It builds the Docker image with the model CI trained and pushes it to the **GitHub Container Registry** (`ghcr.io/nuelbundi/fraud-api`). No passwords are needed: GitHub's built-in `GITHUB_TOKEN` is used. After the first push, the image appears under **Packages** on your GitHub profile.

---

## Run everything on your own computer

```bash
cd fraud_mlops
pip install -r requirements.txt

python train.py                     # 1. train + MLflow + gate  (~3 minutes)
pytest -q                           # 2. test the API
uvicorn api.main:app --reload       # 3. run the API  -> http://127.0.0.1:8000/docs

docker build -t fraud-api .         # 4. Docker
docker run -p 8000:8000 fraud-api
```

---

## Exercises

1. **Break the gate.** Set `MIN_F1 = 0.90` in `train.py` and push. CI fails, and the Docker image is never published.
2. **Add a model.** Add `GradientBoostingClassifier` to the `experiments` list. Does it beat the random forest in MLflow?
3. **Change the threshold.** In `api/main.py` a transaction is fraud if probability ≥ 0.5. What happens to precision and recall at 0.3?
4. **Break the API.** Remove a field from the `Transaction` class. Which test fails?

> **Note:** `mlflow.db`, `mlruns/` and `model/` are created by `train.py` and are **not committed**. CI builds fresh ones on every run.
