# Model Card: Churn Prediction

This file is updated automatically by `.github/workflows/retrain.yml` whenever a retrained candidate model beats the current production model on the fixed holdout set. It is not meant to be edited by hand.

## Current production metrics

| metric | value |
|---|---|
| roc_auc | 0.7595 |
| f1 | 0.5263 |
| precision | 0.4386 |
| recall | 0.6579 |
| accuracy | 0.7300 |
| n_holdout | 1500 |

## Promotion history

- **2026-10-28 21:13 UTC** — promoted. no production model exists yet
- **2026-10-28 22:02 UTC** — promoted. roc_auc improved 0.7556 -> 0.7595
