# MLflow Simple: Experiment Tracking + CI/CD

This folder trains several machine-learning models, tracks them with **MLflow**, and uses **GitHub Actions** to test and deploy the best one automatically.

- **CI (Continuous Integration)** decides whether the new model is good enough.
- **CD (Continuous Deployment)** ships the model, but only if CI says yes.

---

## Files

| File | Purpose |
|---|---|
| [`train.py`](train.py) | Trains 9 models, logs them to MLflow, applies the quality gate, and registers the best model as `production` |
| [`test_model.py`](test_model.py) | Automated tests run on the `production` model |
| [`predict.py`](predict.py) | The "deployment": loads the `production` model and makes predictions |
| [`requirements.txt`](requirements.txt) | Python packages needed |
| [`../.github/workflows/mlflow-simple.yml`](../.github/workflows/mlflow-simple.yml) | The CI/CD pipeline |

---

## The whole flow

```
 git push
    │
    ▼
 ┌─────────── CI ───────────┐
 │ train 9 models → MLflow   │
 │ best accuracy ≥ 0.90 ?    │── no ──► ❌ stop (nothing deployed)
 │        │ yes              │
 │ register as "production"  │
 │ pytest passes ?           │── no ──► ❌ stop
 └────────┬─────────────────┘
          │ yes, and branch = master
          ▼
 ┌─────────── CD ───────────┐
 │ load "production" model   │
 │ predict  ✅ deployed      │
 └──────────────────────────┘
```

---

## Step 1: You push code

```bash
git push
```

GitHub reads `.github/workflows/mlflow-simple.yml`. The workflow only starts if something inside `mlflow_simple/` (or the workflow file itself) changed. Pushes that touch other labs don't trigger it.

---

## Step 2: CI job ("Is the new model good?")

GitHub starts a fresh, empty Linux computer and runs these steps in order. **If any step fails, everything stops.**

| Step | Command | What happens |
|---|---|---|
| Install | `pip install -r requirements.txt` | Installs mlflow, scikit-learn and pytest |
| Train | `python train.py` | Trains 9 models and logs each one to MLflow (`mlflow.db`) |
| Quality gate | end of `train.py` | Finds the run with the best accuracy. **Below 0.90 → `sys.exit(1)` → pipeline fails ❌** |
| Register | end of `train.py` | Saves the best model in the MLflow **Model Registry** as `simple-classifier` and labels it `production` |
| Test | `pytest test_model.py` | Checks the `production` model (see below) |
| Save | upload-artifact | Saves `mlflow.db` and `mlruns/` so the CD job (and you) can use them |

### Models trained

| Model | Parameters tried |
|---|---|
| Logistic Regression | `C = 1.0`, `C = 0.1` |
| Decision Tree | `max_depth = 3`, `max_depth = 10` |
| Random Forest | 50 trees at depth 5, 200 trees at depth 10 |
| Gradient Boosting | 100 trees, learning rate 0.1, depth 3 |
| K-Nearest Neighbours | 5 neighbours, 15 neighbours |

For every run, MLflow logs the model name, its parameters, **accuracy, precision, recall and F1-score**, and the saved model itself.

### What `test_model.py` checks

Each test loads `models:/simple-classifier@production` from MLflow:

1. `test_model_loads`: the model exists and can be loaded.
2. `test_predictions_are_0_or_1`: it returns valid classes (0 = malignant, 1 = benign).
3. `test_accuracy_is_good_enough`: its accuracy is still at least 0.90.

If any test fails, pytest exits with an error and the pipeline stops.

---

## Step 3: CD job ("Ship it")

The CD job only runs if **both** of these are true:

- the CI job passed (`needs: ci`), **and**
- the push was to `master` (a pull request runs CI only, never CD).

It starts another fresh computer, downloads the MLflow results saved by CI, and runs:

```bash
python predict.py
```

That script loads the `production` model and makes predictions. In a real project, this step would be "copy the model to the server" or "restart the API".

---

## Where MLflow fits

- **Tracking:** every model, its parameters and its metrics are recorded, so you can compare them.
- **Model Registry:** the winner gets the `production` label. `test_model.py` and `predict.py` never hard-code which model won; they just ask for *whatever is `production`*.

So if a better model wins next time, the label moves to it, and the tests and predict script use it automatically, with no code changes.

---

## Run it on your own computer

```bash
cd mlflow_simple
pip install -r requirements.txt
python train.py          # train, gate, register
pytest -q test_model.py  # test the production model
python predict.py        # "deploy"
```

View the results in the MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open <http://127.0.0.1:5000>, go to **simple-model-comparison**, select the runs and click **Compare**.

---

## Try it: make the pipeline fail

In `train.py`, change:

```python
MIN_ACCURACY = 0.90
```

to:

```python
MIN_ACCURACY = 0.99
```

Push the change. CI stops with a red ❌ and **the CD job never runs**. This is how CI/CD protects production from a bad model.

---

> **Note:** Do not commit `mlflow.db`. It is created fresh on every run, and a copy from your computer contains file paths that will not work on GitHub.
