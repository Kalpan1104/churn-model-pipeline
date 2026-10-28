import unittest

import pandas as pd

from src.features import build_features


class TestBuildFeatures(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame([
            {
                "customer_id": "cust-000001",
                "tenure_months": 10,
                "monthly_charges": 50.0,
                "total_charges": 500.0,
                "contract_type": "month-to-month",
                "support_tickets_90d": 1,
                "payment_delay_days": 0,
                "usage_30d": 40.0,
                "signup_channel": "organic",
                "churned": 0,
            },
            {
                "customer_id": "cust-000002",
                "tenure_months": 60,
                "monthly_charges": 80.0,
                "total_charges": 4800.0,
                "contract_type": "two-year",
                "support_tickets_90d": 0,
                "payment_delay_days": 0,
                "usage_30d": 90.0,
                "signup_channel": "referral",
                "churned": 0,
            },
        ])

    def test_fixed_category_columns_present_even_when_absent_from_data(self):
        _, feature_cols = build_features(self.df)
        self.assertIn("contract_one-year", feature_cols)
        self.assertIn("channel_paid-ads", feature_cols)
        self.assertIn("channel_partner", feature_cols)

    def test_no_nulls_in_output(self):
        feat_df, feature_cols = build_features(self.df)
        self.assertEqual(feat_df[feature_cols].isnull().sum().sum(), 0)

    def test_charges_per_tenure_computed_correctly(self):
        feat_df, _ = build_features(self.df)
        self.assertAlmostEqual(feat_df.loc[0, "charges_per_tenure"], 50.0)

    def test_high_support_load_flag(self):
        df = self.df.copy()
        df.loc[0, "support_tickets_90d"] = 4
        feat_df, _ = build_features(df)
        self.assertEqual(feat_df.loc[0, "high_support_load"], 1.0)
        self.assertEqual(feat_df.loc[1, "high_support_load"], 0.0)

    def test_churned_label_passed_through(self):
        feat_df, _ = build_features(self.df)
        self.assertIn("churned", feat_df.columns)


if __name__ == "__main__":
    unittest.main()
