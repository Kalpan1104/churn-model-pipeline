import argparse
import json

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.features import build_features


def evaluate_model(model_bundle_path, holdout_csv_path):
    bundle = joblib.load(model_bundle_path)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    df = pd.read_csv(holdout_csv_path)
    feat_df, _ = build_features(df)

    X = feat_df.reindex(columns=feature_cols, fill_value=0.0)
    y = feat_df["churned"]

    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= 0.5).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y, proba),
        "f1": f1_score(y, preds, zero_division=0),
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "accuracy": accuracy_score(y, preds),
        "n_holdout": len(y),
    }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--holdout", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    metrics = evaluate_model(args.model, args.holdout)
    with open(args.out, "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
