import mlflow
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score




mlflow.set_tracking_uri("sqlite:///mlflow.db")


def load_production_model():
    return mlflow.sklearn.load_model("models:/simple-classifier@production")


def test_model_loads():
    assert load_production_model() is not None


def test_predictions_are_0_or_1():
    X, _  = load_breast_cancer(return_X_y=True)
    preds = load_production_model().predict(X[:10])
    assert set(preds) <= {0, 1}


def test_accuracy_is_good_enough():
    X, y = load_breast_cancer(return_X_y=True)
    preds = load_production_model().predict(X)
    assert accuracy_score(y, preds) >= 0.90
