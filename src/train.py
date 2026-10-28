import argparse

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.features import build_features


def train_model(train_csv_path, seed=42):
    df = pd.read_csv(train_csv_path)
    feat_df, feature_cols = build_features(df)

    X = feat_df[feature_cols]
    y = feat_df["churned"]

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model, feature_cols


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out-model", required=True)
    args = parser.parse_args()

    model, feature_cols = train_model(args.data)
    joblib.dump({"model": model, "feature_cols": feature_cols}, args.out_model)
    print(f"trained model on {args.data} -> {args.out_model}")


if __name__ == "__main__":
    main()
