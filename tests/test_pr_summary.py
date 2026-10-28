import unittest

from src.pr_summary import render_summary


class TestRenderSummary(unittest.TestCase):
    def setUp(self):
        self.candidate = {
            "roc_auc": 0.80, "f1": 0.55, "precision": 0.5,
            "recall": 0.6, "accuracy": 0.75,
        }

    def test_handles_no_production_model_yet(self):
        summary = render_summary(self.candidate, None)
        self.assertIn("n/a", summary)
        self.assertIn("would promote", summary)

    def test_better_candidate_says_would_promote(self):
        production = {**self.candidate, "roc_auc": 0.70}
        summary = render_summary(self.candidate, production)
        self.assertIn("would promote", summary)
        self.assertNotIn("would NOT promote", summary)

    def test_worse_candidate_says_would_not_promote(self):
        production = {**self.candidate, "roc_auc": 0.95}
        summary = render_summary(self.candidate, production)
        self.assertIn("would NOT promote", summary)

    def test_table_includes_all_metrics(self):
        summary = render_summary(self.candidate, self.candidate)
        for key in ("roc_auc", "f1", "precision", "recall", "accuracy"):
            self.assertIn(key, summary)


if __name__ == "__main__":
    unittest.main()
