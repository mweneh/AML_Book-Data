# Lab 02 — Messy Data and Leakage-Safe Features

Lab 02 works on a synthetic mobile-money transaction dump of roughly 183,000 rows and 18
columns, stored in `Data/mobile_money_statements.csv`. The target column is `is_fraud`,
a binary flag marking transactions confirmed as fraudulent.

## Task 1 — Audit

Students compute missingness per column, count exact duplicate rows, check validity
violations, and classify each incomplete column as MCAR, MAR or MNAR. The expected answers
are: `agent_id` is MAR, `device_id` is MCAR, and `gps_lat` and `gps_lon` are MNAR. The GPS
columns are MNAR because location is missing precisely when the customer has disabled
location sharing, which is itself correlated with fraud risk.

## Task 2 — Clean and aggregate

The `amount` column arrives as a raw string in three awkward forms: `"1,500/-"` for a plain
credit, `"(227/-)"` using accounting parentheses for a negative, and `"4,742/- Dr"` using a
debit suffix. The `txn_time` column arrives in three different date formats. Entities are
resolved to a canonical `customer_id` through the `reg_id` KYC national-registration key,
because one customer may hold several `msisdn` mobile numbers.

Customer-level features are then built as of a fixed `SCORING_TS` timestamp: recency,
frequency and monetary (RFM) features, ratio features, and cyclical encodings of hour and
day of week using sine and cosine transforms.

## Task 3 — Hunt the leak

Two columns are post-outcome fields and must be dropped: `manual_review_score` and
`settlement_status`. Both are only written *after* a transaction has already been scored
and investigated, so they cannot be available at prediction time.

## Task 4 — Pipeline

A single `ColumnTransformer` inside a `Pipeline` is evaluated with `GroupKFold` grouped by
`customer_id`, so that no customer appears in both a training and a validation fold.

**Headline result:** ROC-AUC is approximately 0.63 on the honest feature set and
approximately 1.00 once the two leaky columns are added back — about +0.37 of pure
illusion. This is the single most important lesson of the lab: a suspiciously perfect score
is evidence of leakage, not of skill.
