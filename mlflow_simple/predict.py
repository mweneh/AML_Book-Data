import mlflow
from sklearn.datasets import load_breast_cancer

mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Load whichever model is currently marked "production" in the MLflow registry
model = mlflow.sklearn.load_model("models:/simple-classifier@production")

X, y = load_breast_cancer(return_X_y=True)
print("Predictions:", model.predict(X[:5]))
print("Actual:     ", y[:5])
