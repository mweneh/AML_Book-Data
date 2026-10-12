"""Train fraud-detection models, track them in MLflow, and export the best one for the API.

Run from this folder:  python train.py
"""
import json
import sys
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

HERE      = Path(__file__).parent
DATA_PATH = HERE.parent / "Data" / "CleanedFeaturesFromLoan.csv"
MODEL_DIR = HERE / "model"

MODEL_NAME = "fraud-detector"
MIN_F1     = 0.50   # quality gate: the best model must reach this F1-score to be deployed

# 1. Load data
df = pd.read_csv(DATA_PATH)
TARGET   = "is_fraud"
FEATURES = [c for c in df.columns if c not in ("txn_dt", TARGET)]

X_train, X_test, y_train, y_test = train_test_split(
    df[FEATURES], df[TARGET], test_size=0.2, random_state=42, stratify=df[TARGET]
)

# 2. Models to compare (class_weight="balanced" because fraud is rare: ~8% of transactions)
experiments = [
    ("logistic_regression", make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)),
     {"C": 1.0}),
    ("decision_tree", DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=42),
     {"max_depth": 8}),
    ("random_forest", RandomForestClassifier(n_estimators=100, max_depth=10, class_weight="balanced", n_jobs=-1, random_state=42),
     {"n_estimators": 100, "max_depth": 10}),
    ("random_forest", RandomForestClassifier(n_estimators=200, max_depth=15, class_weight="balanced", n_jobs=-1, random_state=42),
     {"n_estimators": 200, "max_depth": 15}),
]

# 3. Train and track every model in MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("fraud-detection")

results = []   # (metrics, name, run_id, model, model_uri) for every run in this session
for name, model, params in experiments:
    with mlflow.start_run(run_name=name) as run:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        metrics = {
            "roc_auc":   roc_auc_score(y_test, probs),
            "precision": precision_score(y_test, preds),
            "recall":    recall_score(y_test, preds),
            "f1_score":  f1_score(y_test, preds),
        }
        mlflow.log_param("model", name)
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        logged = mlflow.sklearn.log_model(model, name="model", skops_trusted_types=["sklearn.tree._tree.Tree"])

        results.append((metrics, name, run.info.run_id, model, logged.model_uri))
        print(f"{name:20} {params}  f1={metrics['f1_score']:.3f}  recall={metrics['recall']:.3f}  auc={metrics['roc_auc']:.3f}")

# 4. Pick the best run by F1-score (balances catching fraud vs. false alarms)
best_metrics, best_name, best_run_id, best_model, best_uri = max(results, key=lambda r: r[0]["f1_score"])
print(f"\nBest model: {best_name}  f1={best_metrics['f1_score']:.3f}")

# 5. Quality gate: stop the pipeline if the best model is not good enough
if best_metrics["f1_score"] < MIN_F1:
    print(f"FAILED: F1 is below {MIN_F1}. Not deploying.")
    sys.exit(1)

# 6. Register in MLflow and mark as "production"
version = mlflow.register_model(best_uri, MODEL_NAME)
mlflow.MlflowClient().set_registered_model_alias(MODEL_NAME, "production", version.version)

# 7. Export the production model for the API / Docker image
MODEL_DIR.mkdir(exist_ok=True)
joblib.dump(best_model, MODEL_DIR / "model.joblib")
(MODEL_DIR / "metadata.json").write_text(json.dumps({
    "model_name": MODEL_NAME,
    "version":    version.version,
    "algorithm":  best_name,
    "run_id":     best_run_id,
    "features":   FEATURES,
    "metrics":    {k: round(v, 4) for k, v in best_metrics.items()},
}, indent=2))
print(f"PASSED: {MODEL_NAME} version {version.version} is now 'production' -> exported to model/")
