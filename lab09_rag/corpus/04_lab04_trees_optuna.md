# Lab 04 — Trees, Ensembles and Hyperparameter Search with Optuna

Lab 04 builds from a single decision tree up to gradient-boosted ensembles, then tunes them
with Optuna.

## Models compared

1. **Decision tree** — interpretable, but high variance. An unpruned tree memorises the
   training set and generalises poorly.
2. **Random forest** — many deep trees trained on bootstrap samples with a random subset of
   features at each split. Variance is reduced by averaging. Bagging.
3. **Gradient boosting** — trees added sequentially, each one fitted to the residual errors
   of the ensemble so far. Boosting.
4. **XGBoost / LightGBM** — optimised boosting implementations with regularisation.

## Bagging versus boosting

Bagging trains its base learners **in parallel on different bootstrap samples** and reduces
variance. Boosting trains its base learners **sequentially, each correcting the previous
one**, and reduces bias. Boosting usually wins on tabular data but is far more sensitive to
its hyperparameters and will overfit if the learning rate is too high or there are too many
estimators.

## Optuna

Optuna performs hyperparameter search using a Tree-structured Parzen Estimator (TPE)
sampler, which models the distribution of good and bad configurations and samples where
improvement is likely. This is more sample-efficient than grid search, which wastes trials
on unpromising regions, and than random search, which does not learn from past trials.

The lab runs 50 trials with a `MedianPruner` that stops unpromising trials early. The
objective function returns the mean cross-validated PR-AUC, and the study direction is
`maximize`.

Key searched hyperparameters: `n_estimators`, `max_depth`, `learning_rate`,
`min_child_weight`, `subsample`, `colsample_bytree`, and the L1/L2 regularisation terms
`reg_alpha` and `reg_lambda`.

## Feature importance

The lab contrasts three views: built-in Gini importance (biased toward high-cardinality
features), permutation importance (measured on held-out data, so more trustworthy), and
SHAP values (per-prediction attributions that also show the direction of an effect).
