# Credit Risk Model Evaluation: Data Quality and Leakage Assessment

**To:** Head of Credit Risk \
**From:** David Mauti \
**Date:** August 17, 2026 \
**Subject:** Mobile Money Fraud Model: Data Audit, Feature Dictionary, and Leakage Analysis

---

## 1. Data Quality Audit

We ran a full diagnostic pass over the raw mobile money transaction data before any modelling work began. The audit covered missingness rates, duplicate detection, data type validation, and statistical screening for suspicious correlations. Ten columns showed the most significant problems, summarised in the table below.

| Column | Issue | Evidence | Missing-Data Class | Remediation and Why |
| :--- | :--- | :--- | :--- | :--- |
| `txn_id` | Exact duplicate rows. The same transaction record appears more than once, most likely from retry events or log merge failures. | `df.duplicated()` counted 3,600 exact duplicate rows out of 183,600 total (2.0%). The duplicated `txn_id` count matched exactly. | N/A | Drop duplicates before any train-test split. If a duplicate row lands in both train and validation, the model sees the same observation twice and performance estimates become optimistic. Dropping first resolves this with no information loss. |
| `amount` | Stored as a free-text string rather than a number. The same monetary value can appear in three different encodings. | `dtype` is `object`. Sampled values: `"1,329/- Dr"`, `"(1,307/-)"`, `"-2,136/-"`. A direct cast to `float` produces 100% NaN. | N/A | Apply a regex parser that strips commas, the `/-` suffix, and brackets, then infers sign from context (`Dr`, leading `-`, or parentheses). Zero parse failures were observed after applying this rule. |
| `txn_time` | Three incompatible datetime formats coexist in the same column. | Sampled values: ISO format `"2026-06-30 14:24:19"`, UK-style `"07/04/2026 11:21"`, and long-form `"Jan 29, 2026 01:17 PM"`. A single-format parse call fails on roughly two thirds of rows. | N/A | Use `pd.to_datetime` with `errors="coerce"` across three sequential format attempts. This recovers all valid timestamps with zero residual NaT values. |
| `reg_id` | The same customer appears under multiple spelling variants due to case and whitespace differences in the KYC field. | Raw `reg_id` has 11,722 distinct values. After `.strip().upper()`, this collapses to 3,991 unique customers, a 66% reduction. Using the raw value would incorrectly fragment one person's history across multiple pseudo-identities. | N/A | Canonicalise to uppercase with whitespace stripped before any grouping or cross-validation. This is the minimum normalisation needed to ensure one customer maps to exactly one group. |
| `agent_id` | Missingness is concentrated in specific regions, not spread randomly. | `agent_id` is missing 13.69% overall. Broken down by region: Arusha 45.2%, Jinja 45.1%, Mwanza 44.8%, all other regions 0.0%. The fraud rate is equal between missing and present rows (8.39% vs 8.41%), ruling out outcome-driven dropout. | MAR | Because missingness is fully explained by the observed `region` column (a legacy logger issue in three cities), multiple imputation or a dedicated "Unknown" category is appropriate. The pattern is predictable from data already in the model. |
| `device_id` | Missingness is uniform across all strata, with no detectable driver. | Missing rate is 5.91% and is flat across every region (range: 5.6% to 6.1%). The fraud rate among rows with a missing `device_id` is 8.38%, versus 8.40% where it is present. No observed column predicts the dropout. | MCAR | Treating missing as a separate "Unknown" category is sufficient. Because there is no systematic pattern, more complex imputation adds no value and risks introducing artificial structure. |
| `gps_lat` | Missingness is high and cannot be explained by any observed column, suggesting the probability of a missing coordinate depends on the coordinate itself. | Missing rate is 36.21% (66,464 rows). The rate is nearly uniform across all ten regions (range: 35.4% to 36.7%), so `region` does not explain it. Remote or indoor transactions are inherently more likely to fail GPS capture, making the missing rate dependent on the unrecorded location. | MNAR | Create a binary `gps_missing` flag rather than attempting imputation. The absence of a GPS ping may itself carry predictive signal (e.g., transactions initiated from inside buildings or in areas with no tower coverage). Imputing coordinates would fabricate geographic information. |
| `gps_lon` | Identical dropout pattern to `gps_lat`. | Missing fraction: 36.21%, perfectly correlated with `gps_lat` missingness (both are NaN on exactly the same rows). | MNAR | Handled by the same `gps_missing` flag as `gps_lat`. No separate action is needed. |
| `manual_review_score` | A continuous score that encodes post-event human judgment about fraud. Including it in training constitutes target leakage. | Pearson correlation with `is_fraud`: ` | r | = 0.981`. This value is physically impossible for a pre-event predictor on a real 8% fraud rate. The score is written by an analyst only after a transaction has already been flagged. |
| `settlement_status` | A categorical outcome written after the transaction lifecycle completes. Fraud cases are almost always reversed, so this field proxies the label. | Fraud rate by category: `reversed` = 100.0%, `held` = 65.1%, `pending` = 0.0%, `settled` = 0.4%. The `reversed` category alone gives near-perfect class separation. Settlement can take hours to days and is unknown at prediction time. | N/A | Drop the column entirely. No settlement outcome is observable at the moment a transaction is submitted for a fraud decision. |

---

## 2. Feature Dictionary

We engineered 15 features across five families. All customer-level aggregates are calculated strictly over transactions timestamped before `SCORING_TS = 2026-08-01`, ensuring no future information bleeds into the feature values.

| Feature | Definition | Window | Upstream Fields | Write Time of Upstream Fields | Fairness Note |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `amount_abs` | Absolute value of the transaction amount in local currency units. | Single transaction | `amount` | Written to the transaction ledger at the moment the customer initiates the payment. Available immediately. | N/A |
| `log_amt` | Natural log of `amount_abs + 1`. Compresses the long right tail caused by occasional high-value transfers. | Single transaction | `amount` | Same as `amount_abs`. | N/A |
| `hr_sin` | Sine encoding of the transaction hour (0-23), preserving the cyclic relationship between midnight and 23:00. | Single transaction | `txn_time` | Written to the ledger when the transaction timestamp is recorded at initiation. | N/A |
| `hr_cos` | Cosine encoding of the transaction hour, paired with `hr_sin` for a complete cyclic representation. | Single transaction | `txn_time` | Same as `hr_sin`. | N/A |
| `dow_sin` | Sine encoding of the day of the week (0 = Monday, 6 = Sunday). | Single transaction | `txn_time` | Same as `hr_sin`. | N/A |
| `dow_cos` | Cosine encoding of the day of the week, paired with `dow_sin`. | Single transaction | `txn_time` | Same as `hr_sin`. | N/A |
| `payday_dist` | Days to the nearest typical payday, defined as the 1st or 28th of the month. Captures salary-cycle spending patterns. | Single transaction | `txn_time` | Same as `hr_sin`. | N/A |
| `gps_missing` | Binary flag: 1 if `gps_lat`/`gps_lon` is null, 0 otherwise. Converts MNAR missingness into an explicit signal. | Single transaction | `gps_lat`, `gps_lon` | GPS coordinates are captured by the mobile app at transaction time. If the fix fails, the flag is set immediately. | Rural customers may have systematically higher rates of missing GPS due to infrastructure gaps, not behaviour. This feature should be monitored for disparate impact across geographic segments. |
| `recency_days` | Days since the customer's most recent prior transaction before `SCORING_TS`. A longer gap may indicate an account falling dormant or being reactivated for fraud. | All history before `SCORING_TS` | `txn_time`, `customer_id` | `txn_time` is available at transaction initiation; `customer_id` is derived from the KYC-verified `reg_id` at account opening. | N/A |
| `frequency` | Total count of the customer's transactions before `SCORING_TS`. Represents overall account activity level. | All history before `SCORING_TS` | `txn_time`, `customer_id` | Same as `recency_days`. | N/A |
| `monetary_out` | Sum of all outgoing (debit) transaction values for the customer before `SCORING_TS`. | All history before `SCORING_TS` | `amount`, `customer_id` | `amount` is written to the ledger at transaction initiation. | N/A |
| `monetary_in` | Sum of all incoming (credit) transaction values for the customer before `SCORING_TS`. | All history before `SCORING_TS` | `amount`, `customer_id` | Same as `monetary_out`. | N/A |
| `n_counterparty` | Count of distinct counterparties the customer has transacted with before `SCORING_TS`. A high number of unique recipients may indicate smurfing or money mule behaviour. | All history before `SCORING_TS` | `counterparty`, `customer_id` | `counterparty` is recorded at transaction initiation and is available immediately. | N/A |
| `cashout_ratio` | Proportion of the customer's prior transactions that were cash withdrawals. | All history before `SCORING_TS` | `txn_type`, `customer_id` | `txn_type` is set at the point of transaction creation. | N/A |
| `out_in_ratio` | Ratio of total money sent out to total money received, computed as `monetary_out / (monetary_in + 1)`. Values well above 1.0 may indicate funds flowing out faster than they arrive. | All history before `SCORING_TS` | `amount`, `customer_id` | Same as `monetary_out`. | N/A |

---

## 3. Leakage Assessment Memo

Two columns in the raw dataset were found to carry information that would not be available at the time a real-time fraud decision must be made. These are not modelling choices we can adjust around: they represent a fundamental mismatch between when the data is written and when the model needs to score. Left in the training set, they produce offline metrics that dramatically overstate how well the model will actually perform.

**Background: the six causes of data leakage**

The textbook identifies six mechanisms by which future or outcome-related information contaminates model training: (1) including the target variable directly as a feature, (2) fitting preprocessing steps on the full dataset before splitting, (3) including post-event fields that are only written after the outcome is known, (4) allowing duplicate records to straddle the train and validation split, (5) failing to group related records (e.g., same customer, same household) on the same side of the split, and (6) using proxy variables that encode the label without directly naming it. Both leaks identified below fall under cause 3 and cause 6.

**Leak 1: `manual_review_score`**

Where it occurs: this numeric column is stored directly in the transaction ledger alongside raw event data, making it easy to accidentally include in a feature matrix.

How the leakage happens: the score is generated by a fraud analyst who reviews a transaction only after it has already been flagged as suspicious. It therefore encodes the analyst's conclusion about whether fraud occurred (cause 3, post-event field) and functions as a near-perfect proxy for the target label (cause 6, proxy variable). Both causes apply simultaneously.

Impact: including the column produces a cross-validated ROC-AUC of 1.0000. Removing it brings the honest score down to 0.6252, a reduction of 0.3748 AUC points. Any model champion selected on the inflated metric would have been selected entirely on the basis of this one column.

Governance recommendation: the fraud review score should be stored in a separate operational table with a strict write timestamp, never joined to the raw transaction ledger used for model development. Any feature store query should enforce a point-in-time cutoff and explicitly exclude tables written after the scoring timestamp.

**Leak 2: `settlement_status`**

This categorical field sits in the same transaction table as the raw input features and describes the final resolution of the payment.

How the leakage happens: settlement is a downstream business process that resolves hours or days after the transaction is submitted. The value "reversed" is only written once a fraud dispute has been concluded (cause 3). Because reversed transactions correspond to confirmed fraud cases at a rate of 100%, the field is also a near-perfect class separator (cause 6). At real-time scoring, the status field would show only "pending" for every new transaction, collapsing to a constant and providing no information.

On its own, this column was the secondary driver of the inflated AUC. Once `manual_review_score` is removed but `settlement_status` is retained, the "reversed" category still creates a near-perfect categorical split (100% fraud rate), keeping the AUC close to 1.0. Only when both columns are removed simultaneously does the ROC-AUC settle at the honest 0.6252. The combined inflation attributable to both leaks together is 0.3748 AUC points.

Governance recommendation: settlement status should be treated as an outcome field, not an input field. It should live in a separate post-settlement table that is excluded from all feature pipelines by default. Access to outcome tables should require a documented exception reviewed by the model risk team.

---

## 4. Reproducible Machine Learning Pipeline

The pipeline is implemented as a single scikit-learn `Pipeline` object. All feature engineering, imputation, scaling, and encoding steps run inside the pipeline, which means they are re-fitted from scratch on each training fold during cross-validation. This eliminates preprocessing leakage by construction.

**Architecture:** A custom `FeatureEngineer` transformer handles amount parsing, datetime extraction, cyclical encoding, missingness flagging, and customer-level RFM aggregation. It is followed by a `ColumnTransformer` that applies `SimpleImputer` and `RobustScaler` to numeric features and `SimpleImputer` plus `OneHotEncoder` to categorical features. A logistic regression classifier sits at the end of the chain.

**Cross-validation:** `GroupKFold` with 5 splits groups records by the canonicalised `customer_id`. This ensures no customer's transactions appear on both sides of a fold boundary (preventing leakage cause 5).

**Result:** The honest, leakage-free cross-validated ROC-AUC is **0.6037 (+/- 0.0073)**. For reference, the contaminated model (both leaky columns included) scored 1.0000, confirming the 0.3748 inflation documented in the leakage memo above.

**Serialisation:** The trained pipeline is saved to `pipeline.joblib` using `joblib.dump`. It can be reloaded with `joblib.load("pipeline.joblib")` and applied to a new raw transaction table without any preprocessing steps outside the object.
