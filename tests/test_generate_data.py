import unittest

from src.generate_data import generate_customers


class TestGenerateCustomers(unittest.TestCase):
    def test_row_count_and_schema(self):
        rows = generate_customers(200, seed=1)
        self.assertEqual(len(rows), 200)
        expected_cols = {
            "customer_id", "tenure_months", "monthly_charges", "total_charges",
            "contract_type", "support_tickets_90d", "payment_delay_days",
            "usage_30d", "signup_channel", "churned",
        }
        self.assertEqual(set(rows[0].keys()), expected_cols)

    def test_deterministic_given_same_seed(self):
        rows_a = generate_customers(100, seed=42)
        rows_b = generate_customers(100, seed=42)
        self.assertEqual(rows_a, rows_b)

    def test_different_seeds_differ(self):
        rows_a = generate_customers(100, seed=1)
        rows_b = generate_customers(100, seed=2)
        self.assertNotEqual(rows_a, rows_b)

    def test_churn_rate_in_plausible_range(self):
        rows = generate_customers(3000, seed=7)
        churn_rate = sum(r["churned"] for r in rows) / len(rows)
        self.assertTrue(0.05 < churn_rate < 0.45)

    def test_month_to_month_churns_more_than_two_year(self):
        rows = generate_customers(4000, seed=11)
        mtm = [r for r in rows if r["contract_type"] == "month-to-month"]
        two_yr = [r for r in rows if r["contract_type"] == "two-year"]
        mtm_rate = sum(r["churned"] for r in mtm) / len(mtm)
        two_yr_rate = sum(r["churned"] for r in two_yr) / len(two_yr)
        self.assertGreater(mtm_rate, two_yr_rate)


if __name__ == "__main__":
    unittest.main()
