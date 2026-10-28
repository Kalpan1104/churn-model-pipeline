import json
import tempfile
import unittest
from pathlib import Path

from src.promote import promote, should_promote


class TestShouldPromote(unittest.TestCase):
    def test_promotes_when_no_production_exists(self):
        promoted, reason = should_promote({"roc_auc": 0.7}, None)
        self.assertTrue(promoted)
        self.assertIn("no production model", reason)

    def test_promotes_on_meaningful_improvement(self):
        promoted, _ = should_promote({"roc_auc": 0.80}, {"roc_auc": 0.75})
        self.assertTrue(promoted)

    def test_rejects_on_worse_score(self):
        promoted, _ = should_promote({"roc_auc": 0.70}, {"roc_auc": 0.75})
        self.assertFalse(promoted)

    def test_rejects_on_negligible_improvement_within_noise_margin(self):
        promoted, _ = should_promote({"roc_auc": 0.7505}, {"roc_auc": 0.75})
        self.assertFalse(promoted)


class TestPromoteIntegration(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)

        self.candidate_model = self.tmp_path / "candidate_model.joblib"
        self.candidate_metrics = self.tmp_path / "candidate_metrics.json"
        self.production_model = self.tmp_path / "production_model.joblib"
        self.production_metrics = self.tmp_path / "production_metrics.json"
        self.model_card = self.tmp_path / "MODEL_CARD.md"

        self.candidate_model.write_text("fake model bytes v1")
        self.candidate_metrics.write_text(json.dumps({
            "roc_auc": 0.80, "f1": 0.5, "precision": 0.5, "recall": 0.5,
            "accuracy": 0.7, "n_holdout": 100,
        }))

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_first_promotion_creates_production_files_and_card(self):
        promoted, reason = promote(
            str(self.candidate_model), str(self.candidate_metrics),
            str(self.production_model), str(self.production_metrics),
            str(self.model_card),
        )
        self.assertTrue(promoted)
        self.assertTrue(self.production_model.exists())
        self.assertTrue(self.production_metrics.exists())
        self.assertTrue(self.model_card.exists())
        self.assertIn("roc_auc | 0.8000", self.model_card.read_text())

    def test_worse_candidate_does_not_overwrite_production(self):
        promote(
            str(self.candidate_model), str(self.candidate_metrics),
            str(self.production_model), str(self.production_metrics),
            str(self.model_card),
        )
        original_production_bytes = self.production_model.read_text()

        self.candidate_model.write_text("fake model bytes v2 (worse)")
        self.candidate_metrics.write_text(json.dumps({
            "roc_auc": 0.60, "f1": 0.3, "precision": 0.3, "recall": 0.3,
            "accuracy": 0.6, "n_holdout": 100,
        }))

        promoted, reason = promote(
            str(self.candidate_model), str(self.candidate_metrics),
            str(self.production_model), str(self.production_metrics),
            str(self.model_card),
        )
        self.assertFalse(promoted)
        self.assertEqual(self.production_model.read_text(), original_production_bytes)

    def test_better_candidate_updates_card_history_and_keeps_prior_entry(self):
        promote(
            str(self.candidate_model), str(self.candidate_metrics),
            str(self.production_model), str(self.production_metrics),
            str(self.model_card),
        )
        first_card = self.model_card.read_text()

        self.candidate_model.write_text("fake model bytes v2 (better)")
        self.candidate_metrics.write_text(json.dumps({
            "roc_auc": 0.85, "f1": 0.6, "precision": 0.6, "recall": 0.6,
            "accuracy": 0.75, "n_holdout": 100,
        }))

        promoted, _ = promote(
            str(self.candidate_model), str(self.candidate_metrics),
            str(self.production_model), str(self.production_metrics),
            str(self.model_card),
        )
        second_card = self.model_card.read_text()

        self.assertTrue(promoted)
        self.assertIn("roc_auc | 0.8500", second_card)
        self.assertEqual(second_card.count("promoted."), 2)
        self.assertNotEqual(first_card, second_card)


if __name__ == "__main__":
    unittest.main()
