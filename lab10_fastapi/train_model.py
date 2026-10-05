
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

HERE = Path(__file__).parent
DATA_PATH = HERE.parent / "Data" / "CleanedFeaturesFromLoan.csv"
MODEL_PATH = HERE / "model" / "fraud_model.joblib"

# 1. Load data
df = pd.read_csv(DATA_PATH)

# 2. Features (X) and target (y). txn_dt is dropped: its parts (hour, month, ...) are already features.
TARGET = "is_fraud"
FEATURES = [c for c in df.columns if c not in ("txn_dt", TARGET)]
X = df[FEATURES]
y = df[TARGET]

# 3. Train / test split (stratify keeps the fraud ratio the same in both sets)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Train a simple model. class_weight="balanced" helps because fraud is rare.
model = RandomForestClassifier(
    n_estimators=100, max_depth=10, class_weight="balanced", n_jobs=-1, random_state=42
)
model.fit(X_train, y_train)

# 5. Evaluate
print(classification_report(y_test, model.predict(X_test), digits=3))

# 6. Export the model together with the feature order it expects
MODEL_PATH.parent.mkdir(exist_ok=True)
joblib.dump({"model": model, "features": FEATURES}, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
