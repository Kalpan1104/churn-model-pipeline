import argparse

import joblib
import pandas as pd

from src.features import build_features


def score_customers(model_bundle_path, customers_csv_path):
    bundle = joblib.load(model_bundle_path)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    df = pd.read_csv(customers_csv_path)
    feat_df, _ = build_features(df)
    X = feat_df.reindex(columns=feature_cols, fill_value=0.0)

    proba = model.predict_proba(X)[:, 1]
    result = df[["customer_id"]].copy()
    result["churn_risk"] = proba
    return result.sort_values("churn_risk", ascending=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/production_model.joblib")
    parser.add_argument("--customers", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    result = score_customers(args.model, args.customers)
    result.to_csv(args.out, index=False)
    print(f"scored {len(result)} customers -> {args.out}")
    print(result.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
