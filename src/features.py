import pandas as pd

CONTRACT_TYPES = ["month-to-month", "one-year", "two-year"]
SIGNUP_CHANNELS = ["organic", "paid-ads", "referral", "partner"]

NUMERIC_COLS = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "support_tickets_90d",
    "payment_delay_days",
    "usage_30d",
    "charges_per_tenure",
    "high_support_load",
]


def build_features(df):
    df = df.copy()
    df["tenure_months"] = df["tenure_months"].astype(float)
    df["monthly_charges"] = df["monthly_charges"].astype(float)
    df["total_charges"] = df["total_charges"].astype(float)
    df["support_tickets_90d"] = df["support_tickets_90d"].astype(float)
    df["payment_delay_days"] = df["payment_delay_days"].astype(float)
    df["usage_30d"] = df["usage_30d"].astype(float)

    df["charges_per_tenure"] = df["total_charges"] / df["tenure_months"].clip(lower=1)
    df["high_support_load"] = (df["support_tickets_90d"] >= 3).astype(float)

    contract_dummies = pd.get_dummies(
        pd.Categorical(df["contract_type"], categories=CONTRACT_TYPES),
        prefix="contract",
    ).astype(float)
    channel_dummies = pd.get_dummies(
        pd.Categorical(df["signup_channel"], categories=SIGNUP_CHANNELS),
        prefix="channel",
    ).astype(float)

    feature_df = pd.concat([df[NUMERIC_COLS], contract_dummies, channel_dummies], axis=1)
    feature_cols = list(feature_df.columns)

    result = pd.concat([df[["customer_id"]].reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)
    if "churned" in df.columns:
        result["churned"] = df["churned"].astype(int).reset_index(drop=True)

    return result, feature_cols
