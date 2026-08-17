import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GroupKFold
import joblib

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to parse strings, extract datetimes, generate cyclical features,
    and compute customer-level RFM features. This ensures all feature engineering is
    computed purely on the training folds during CV to prevent data leakage.
    """
    def __init__(self, scoring_ts_str="2026-08-01 00:00:00"):
        self.scoring_ts_str = scoring_ts_str
        self.scoring_ts = pd.Timestamp(scoring_ts_str)
        self.rfm_agg_ = None
        
    def fit(self, X, y=None):
        X_df = X.copy()
        
        # 1. Row-Level Parsing
        X_df['amount_signed'] = X_df['amount'].apply(self._parse_amount)
        X_df['amount_abs'] = X_df['amount_signed'].abs()
        X_df['is_debit'] = X_df['amount_signed'] < 0
        X_df['ts'] = pd.to_datetime(X_df['txn_time'], errors='coerce')
        X_df['customer_id'] = X_df['reg_id'].astype(str).str.strip().str.upper()
        
        # 2. Historical RFM Computation (Strictly before scoring_ts)
        # We only aggregate over transactions seen in the training fold.
        hist = X_df[X_df['ts'] < self.scoring_ts]
        by = hist.groupby('customer_id')
        
        agg = pd.DataFrame()
        if not hist.empty:
            agg['recency_days'] = (self.scoring_ts - by['ts'].max()).dt.total_seconds() / 86400
            agg['frequency'] = by.size()
            
            # Using temporary columns 'o' and 'i' to sum out/in amounts
            agg['monetary_out'] = hist.assign(o=np.where(hist['is_debit'], hist['amount_abs'], 0)).groupby('customer_id')['o'].sum()
            agg['monetary_in'] = hist.assign(i=np.where(~hist['is_debit'], hist['amount_abs'], 0)).groupby('customer_id')['i'].sum()
            
            agg['n_counterparty'] = by['counterparty'].nunique()
            agg['cashout_ratio'] = hist.assign(c=(hist['txn_type'] == 'cashout').astype(int)).groupby('customer_id')['c'].mean()
            agg['out_in_ratio'] = agg['monetary_out'] / (agg['monetary_in'] + 1.0)
        
        # Store for the transform step
        self.rfm_agg_ = agg
        return self
        
    def transform(self, X):
        X_df = X.copy()
        
        # 1. Re-apply Row-Level Parsing
        X_df['amount_signed'] = X_df['amount'].apply(self._parse_amount)
        X_df['amount_abs'] = X_df['amount_signed'].abs()
        X_df['log_amt'] = np.log1p(X_df['amount_abs'])
        
        X_df['ts'] = pd.to_datetime(X_df['txn_time'], errors='coerce')
        X_df['customer_id'] = X_df['reg_id'].astype(str).str.strip().str.upper()
        
        # 2. Extract Cyclical / Temporal / Missingness Features
        X_df['hr_sin'] = np.sin(2 * np.pi * X_df['ts'].dt.hour / 24)
        X_df['hr_cos'] = np.cos(2 * np.pi * X_df['ts'].dt.hour / 24)
        X_df['dow_sin'] = np.sin(2 * np.pi * X_df['ts'].dt.dayofweek / 7)
        X_df['dow_cos'] = np.cos(2 * np.pi * X_df['ts'].dt.dayofweek / 7)
        
        day = X_df['ts'].dt.day
        X_df['payday_dist'] = np.minimum((day - 1).abs(), (day - 28).abs())
        X_df['gps_missing'] = X_df['gps_lat'].isna().astype(int)
        
        # 3. Merge Customer-Level RFM features learned during fit()
        if self.rfm_agg_ is not None and not self.rfm_agg_.empty:
            X_df = X_df.merge(self.rfm_agg_, on='customer_id', how='left')
        else:
            # Fallback if no history was present during training
            for col in ['recency_days', 'frequency', 'monetary_out', 'monetary_in', 'n_counterparty', 'cashout_ratio', 'out_in_ratio']:
                X_df[col] = np.nan
                
        # 4. Handle "Cold Start" Customers (present in transform but not seen in fit)
        X_df['frequency'] = X_df['frequency'].fillna(0)
        X_df['monetary_out'] = X_df['monetary_out'].fillna(0)
        X_df['monetary_in'] = X_df['monetary_in'].fillna(0)
        X_df['n_counterparty'] = X_df['n_counterparty'].fillna(0)
        X_df['cashout_ratio'] = X_df['cashout_ratio'].fillna(0)
        X_df['out_in_ratio'] = X_df['out_in_ratio'].fillna(0)
        
        # Missing recency means we've never seen them - impute with an arbitrarily large penalty
        X_df['recency_days'] = X_df['recency_days'].fillna(365) 
        
        return X_df
        
    def _parse_amount(self, s):
        """Helper to convert string amount to float."""
        s = str(s).strip()
        neg = s.startswith("(") or s.startswith("-") or "Dr" in s
        digits = re.sub(r"[^0-9]", "", s)
        if digits == "":
            return np.nan
        v = float(digits)
        return -v if neg else v


def build_and_evaluate_pipeline(data_path):
    print("Loading raw statements data...")
    raw = pd.read_csv(data_path)
    
    # EXACT DEDUPLICATION & GROUP ASSIGNMENT
    # This must happen OUTSIDE the pipeline to prevent train/validation overlap.
    df = raw.drop_duplicates().reset_index(drop=True)
    y = df["is_fraud"].values
    
    # We must construct canonical group boundaries prior to GroupKFold to enforce Leakage Cause 5 rules.
    groups = df["reg_id"].astype(str).str.strip().str.upper().values

    # FEATURE COLUMNS
    NUMERICAL_COLS = [
        "amount_abs", "log_amt", "hr_sin", "hr_cos", "dow_sin", "dow_cos",
        "payday_dist", "gps_missing", "recency_days", "frequency",
        "monetary_out", "monetary_in", "n_counterparty", "cashout_ratio", "out_in_ratio"
    ]
    CATEGORY_COLS = ["txn_type", "region", "segment"]
    
    # Leaky columns `manual_review_score` and `settlement_status` are naturally dropped
    # by the `ColumnTransformer` (remainder="drop" default).

    print("Assembling Scikit-Learn Pipeline...")
    
    # Scalers & Imputers
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", RobustScaler())
    ])

    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # Join Processing Streams
    prep = ColumnTransformer(
        transformers=[
            ("num", num_pipe, NUMERICAL_COLS),
            ("cat", cat_pipe, CATEGORY_COLS)
        ],
        remainder="drop" 
    )

    # Master End-to-End Pipeline
    full_pipeline = Pipeline([
        ("feature_engineer", FeatureEngineer(scoring_ts_str="2026-08-01 00:00:00")),
        ("preprocess", prep),
        ("clf", LogisticRegression(max_iter=2000))
    ])

    print("\nExecuting Group-Aware Cross-Validation (5-Folds)...")
    cv = GroupKFold(n_splits=5)
    
    auc_scores = cross_val_score(full_pipeline, df, y, cv=cv, groups=groups, scoring="roc_auc", n_jobs=1)
    
    print("-" * 50)
    print("FINAL PIPELINE RESULTS (LEAKAGE SAFE)")
    print(f"ROC-AUC: {auc_scores.mean():.4f}  (+/- {auc_scores.std():.4f})")
    print("-" * 50)

    # Save Pipeline to Disk
    joblib.dump(full_pipeline, 'pipeline.joblib')
    print("Success: Pipeline exported to 'pipeline.joblib'")

if __name__ == "__main__":
    import os
    
    # Fallback to local path if not provided
    LOCAL_PATH = "/home/master/Downloads/MODULE-4/AML_2026_Class/AML_Book-Data/Data/mobile_money_statements.csv"
    
    # We will just write a try-except to handle local testing vs submission.
    try:
        build_and_evaluate_pipeline(LOCAL_PATH)
    except FileNotFoundError:
        print(f"Warning: Could not find {LOCAL_PATH}. Please provide a valid file path to the statements CSV.")
