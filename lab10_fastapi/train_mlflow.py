
import argparse
import os
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, PrecisionRecallDisplay, average_precision_score,
    f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

HERE        = Path(__file__).parent
DATA_PATH   = HERE.parent / "Data" / "CleanedFeaturesFromLoan.csv"
MODEL_PATH  = HERE / "model" / "fraud_model.joblib"

# Set MLFLOW_TRACKING_URI to log to an online server; otherwise a local SQLite file is used
TRACKING    = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{(HERE / 'mlflow.db').as_posix()}")
EXPERIMENT  = "lab10_fraud"
REG_NAME    = "lab10_fraud_model"
SEED        = 42

parser = argparse.ArgumentParser()
parser.add_argument("--trials", type=int, default=15, help="Optuna trials per model")
args = parser.parse_args()

optuna.logging.set_verbosity(optuna.logging.WARNING)
mlflow.set_tracking_uri(TRACKING)
mlflow.set_experiment(EXPERIMENT)


# 1. Data: train (60%) / validation (20%, used for tuning) / test (20%, final check only)
df        = pd.read_csv(DATA_PATH)
TARGET   = "is_fraud"
FEATURES = [c for c in df.columns if c not in ("txn_dt", TARGET)]
X, y = df[FEATURES], df[TARGET]

X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=SEED)
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()   # for XGBoost's imbalance handling


# 2. Search spaces: each function turns an Optuna trial into a model
def logreg(t):
    return make_pipeline(StandardScaler(), LogisticRegression(
        C=t.suggest_float("C", 1e-3, 10, log=True),
        class_weight="balanced", max_iter=2000,
    ))

def random_forest(t):
    return RandomForestClassifier(
        n_estimators=t.suggest_int("n_estimators", 100, 400, step=50),
        max_depth=t.suggest_int("max_depth", 4, 20),
        min_samples_leaf=t.suggest_int("min_samples_leaf", 1, 20),
        max_features=t.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
        class_weight="balanced", n_jobs=-1, random_state=SEED,
    )

def xgboost(t):
    return XGBClassifier(
        n_estimators=t.suggest_int("n_estimators", 100, 600, step=50),
        max_depth=t.suggest_int("max_depth", 3, 10),
        learning_rate=t.suggest_float("learning_rate", 0.01, 0.3, log=True),
        subsample=t.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=t.suggest_float("colsample_bytree", 0.6, 1.0),
        min_child_weight=t.suggest_int("min_child_weight", 1, 10),
        scale_pos_weight=pos_weight, eval_metric="aucpr",
        n_jobs=-1, random_state=SEED,
    )

def lightgbm(t):
    return LGBMClassifier(
        n_estimators=t.suggest_int("n_estimators", 100, 600, step=50),
        num_leaves=t.suggest_int("num_leaves", 15, 255),
        learning_rate=t.suggest_float("learning_rate", 0.01, 0.3, log=True),
        subsample=t.suggest_float("subsample", 0.6, 1.0), subsample_freq=1,
        colsample_bytree=t.suggest_float("colsample_bytree", 0.6, 1.0),
        min_child_samples=t.suggest_int("min_child_samples", 5, 100),
        class_weight="balanced", n_jobs=-1, random_state=SEED, verbose=-1,
    )

MODELS = {"logistic_regression": logreg, "random_forest": random_forest,
          "xgboost": xgboost, "lightgbm": lightgbm}


# 3. Helpers
def scores(y_true, proba, threshold=0.5):
    pred = (proba >= threshold).astype(int)
    return {
        "pr_auc":    average_precision_score(y_true, proba),   # main metric (fraud is rare)
        "roc_auc":   roc_auc_score(y_true, proba),
        "f1":        f1_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall":    recall_score(y_true, pred),
    }

def best_threshold(y_true, proba):
    """Threshold that maximises F1 on the validation set."""
    p, r, t = precision_recall_curve(y_true, proba)
    f1 = 2 * p * r / (p + r + 1e-12)
    return float(t[np.argmax(f1[:-1])])

def log_plots(model, y_true, proba, threshold, prefix):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ConfusionMatrixDisplay.from_predictions(y_true, (proba >= threshold).astype(int), ax=ax[0], colorbar=False)
    ax[0].set_title(f"Confusion matrix (threshold={threshold:.2f})")
    PrecisionRecallDisplay.from_predictions(y_true, proba, ax=ax[1])
    ax[1].set_title("Precision-Recall curve")
    fig.tight_layout()
    mlflow.log_figure(fig, f"{prefix}_evaluation.png")
    plt.close(fig)

    est = model[-1] if hasattr(model, "steps") else model
    imp = getattr(est, "feature_importances_", None)
    if imp is None and hasattr(est, "coef_"):
        imp = np.abs(est.coef_[0])
    if imp is not None:
        s = pd.Series(imp, index=FEATURES).sort_values()
        fig, ax = plt.subplots(figsize=(7, 6))
        s.plot.barh(ax=ax)
        ax.set_title("Feature importance")
        fig.tight_layout()
        mlflow.log_figure(fig, f"{prefix}_feature_importance.png")
        plt.close(fig)
        mlflow.log_dict(s.sort_values(ascending=False).to_dict(), f"{prefix}_feature_importance.json")


# 4. Tune + train every model. One parent run per model, one nested run per Optuna trial.
dataset = mlflow.data.from_pandas(df, source=str(DATA_PATH), targets=TARGET, name="CleanedFeaturesFromLoan")
results = []

for name, build in MODELS.items():
    print(f"\n=== {name}: {args.trials} trials ===")
    with mlflow.start_run(run_name=name) as parent:
        mlflow.set_tags({"model_type": name, "stage": "tuning", "tuner": "optuna-TPE"})
        mlflow.log_input(dataset, context="training")
        mlflow.log_params({"n_trials": args.trials, "optimize_metric": "val_pr_auc",
                           "n_features": len(FEATURES), "n_train": len(X_train),
                           "n_val": len(X_val), "n_test": len(X_test), "seed": SEED})

        def objective(trial):
            with mlflow.start_run(run_name=f"{name}_trial_{trial.number}", nested=True):
                model = build(trial)
                model.fit(X_train, y_train)
                m = scores(y_val, model.predict_proba(X_val)[:, 1])
                mlflow.log_params(trial.params)
                mlflow.log_metrics({f"val_{k}": v for k, v in m.items()})
                return m["pr_auc"]

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(objective, n_trials=args.trials)

        # Refit the best configuration, pick a decision threshold on validation, report on test
        model = build(optuna.trial.FixedTrial(study.best_params))
        model.fit(X_train, y_train)
        val_proba  = model.predict_proba(X_val)[:, 1]
        test_proba = model.predict_proba(X_test)[:, 1]
        thr = best_threshold(y_val, val_proba)

        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_param("decision_threshold", round(thr, 4))
        mlflow.log_metrics({f"val_{k}":  v for k, v in scores(y_val,  val_proba,  thr).items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in scores(y_test, test_proba, thr).items()})
        log_plots(model, y_test, test_proba, thr, prefix="test")
        mlflow.log_dict({"features": FEATURES, "target": TARGET}, "features.json")
        mlflow.log_table(study.trials_dataframe(), "optuna_trials.json")

        mlflow.sklearn.log_model(
            model, name="model",
            signature=infer_signature(X_val, val_proba),
            input_example=X_val.head(3),
            serialization_format="cloudpickle",   # skops (the default) rejects tree models
        )

        test = scores(y_test, test_proba, thr)
        print(f"  best val PR-AUC={study.best_value:.4f} | test PR-AUC={test['pr_auc']:.4f} "
              f"F1={test['f1']:.4f} (thr={thr:.2f})")
        results.append({"name": name, "run_id": parent.info.run_id, "model": model,
                        "threshold": thr, "val_pr_auc": study.best_value, **{f"test_{k}": v for k, v in test.items()}})


# 5. Pick the winner on VALIDATION PR-AUC (test is only for reporting), register and export it
board = pd.DataFrame(results).drop(columns="model").sort_values("val_pr_auc", ascending=False)
print("\n=== Leaderboard ===")
print(board.to_string(index=False, float_format="%.4f"))

best = max(results, key=lambda r: r["val_pr_auc"])
with mlflow.start_run(run_id=best["run_id"]):
    mlflow.set_tag("best_model", "true")
mv = mlflow.register_model(f"runs:/{best['run_id']}/model", REG_NAME)
client = mlflow.MlflowClient()
client.set_registered_model_alias(REG_NAME, "champion", mv.version)

with mlflow.start_run(run_name="leaderboard"):
    mlflow.set_tag("stage", "summary")
    mlflow.log_table(board, "leaderboard.json")
    mlflow.log_params({"winner": best["name"], "winner_run_id": best["run_id"],
                       "registered_version": mv.version})

# Same bundle format the FastAPI app already loads
MODEL_PATH.parent.mkdir(exist_ok=True)
joblib.dump({"model": best["model"], "features": FEATURES, "threshold": best["threshold"],
             "model_name": best["name"], "mlflow_run_id": best["run_id"]}, MODEL_PATH)

print(f"\nWinner: {best['name']} -> registered as {REG_NAME} v{mv.version} (@champion)")
print(f"Saved to {MODEL_PATH}")
print("View: mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000")
