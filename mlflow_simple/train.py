import sys

import mlflow
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

MODEL_NAME = "simple-classifier"
MIN_ACCURACY = 0.90   # quality gate: the best model must beat this to be deployed

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Each entry: a name, a model, and the parameters to try
experiments = [
    ("logistic_regression", LogisticRegression, {"C": 1.0, "max_iter": 5000}),
    ("logistic_regression", LogisticRegression, {"C": 0.1, "max_iter": 5000}),
    ("decision_tree", DecisionTreeClassifier, {"max_depth": 3}),
    ("decision_tree", DecisionTreeClassifier, {"max_depth": 10}),
    ("random_forest", RandomForestClassifier, {"n_estimators": 50, "max_depth": 5}),
    ("random_forest", RandomForestClassifier, {"n_estimators": 200, "max_depth": 10}),
    ("gradient_boosting", GradientBoostingClassifier, {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3}),
    ("knn", KNeighborsClassifier, {"n_neighbors": 5}),
    ("knn", KNeighborsClassifier, {"n_neighbors": 15}),
]

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("simple-model-comparison")

for name, Model, params in experiments:
    with mlflow.start_run(run_name=name):
        model = Model(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mlflow.log_param("model", name)
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy_score(y_test, preds))
        mlflow.log_metric("precision", precision_score(y_test, preds))
        mlflow.log_metric("recall", recall_score(y_test, preds))
        mlflow.log_metric("f1_score", f1_score(y_test, preds))
        mlflow.sklearn.log_model(model, name="model", skops_trusted_types=["sklearn.tree._tree.Tree"])

        print(f"{name:20} {params}  accuracy={accuracy_score(y_test, preds):.3f}")

# Pick the best run
best = mlflow.search_runs(order_by=["metrics.accuracy DESC"]).iloc[0]
print("\nBest model:", best["params.model"], " accuracy:", round(best["metrics.accuracy"], 3))

# Quality gate: stop the pipeline if the best model is not good enough
if best["metrics.accuracy"] < MIN_ACCURACY:
    print(f"FAILED: accuracy is below {MIN_ACCURACY}. Not deploying.")
    sys.exit(1)

# Register the best model and mark it as "production"
version = mlflow.register_model(f"runs:/{best['run_id']}/model", MODEL_NAME)
mlflow.MlflowClient().set_registered_model_alias(MODEL_NAME, "production", version.version)
print(f"PASSED: {MODEL_NAME} version {version.version} is now 'production'.")
