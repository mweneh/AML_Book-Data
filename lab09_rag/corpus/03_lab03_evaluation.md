# Lab 03 — Model Evaluation and Imbalanced Classes

Lab 03 uses the cleaned loan feature table in `Data/CleanedFeaturesFromLoan.csv`. The lab
is about choosing the right metric rather than about squeezing out a better model.

## Why accuracy fails

On the loan default data roughly 6% of borrowers default. A model that predicts "no
default" for every single applicant therefore scores 94% accuracy while being completely
useless. Accuracy is a bad metric whenever the classes are imbalanced, because the majority
class dominates the score.

## Metrics covered

- **Precision** — of the applicants the model flagged, what fraction actually defaulted.
- **Recall** — of the applicants who actually defaulted, what fraction the model caught.
- **F1** — the harmonic mean of precision and recall, useful when you need one number.
- **ROC-AUC** — the probability that a randomly chosen positive is ranked above a randomly
  chosen negative. Insensitive to the decision threshold.
- **PR-AUC** — area under the precision-recall curve. Preferred over ROC-AUC when positives
  are rare, because ROC-AUC can look flattering when true negatives are abundant.

## Threshold selection

The default threshold of 0.5 is an arbitrary convention, not a law. The lab has students
sweep the threshold and pick the operating point using an explicit cost matrix: a missed
default costs the lender roughly ten times what a wrongly rejected good applicant costs.
Under that cost matrix the optimal threshold lands near 0.23, not 0.5.

## Calibration

A `CalibratedClassifierCV` wrapper with isotonic regression is used to check whether the
predicted probabilities mean what they claim. A reliability diagram plots predicted
probability against observed frequency; a well-calibrated model lies on the diagonal.

## Cross-validation

`StratifiedKFold` with five folds is used throughout so that each fold preserves the 6%
positive rate. Results are reported as mean plus or minus one standard deviation across
folds, never as a single number from a single split.
