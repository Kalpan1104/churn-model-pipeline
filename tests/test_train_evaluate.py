import json
import tempfile
import unittest
from pathlib import Path

import joblib

from src.evaluate import evaluate_model
from src.generate_data import generate_customers, write_csv
from src.train import train_model


class TestTrainEvaluate(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)

        train_rows = generate_customers(1500, seed=101)
        holdout_rows = generate_customers(500, seed=909090)

        self.train_csv = self.tmp_path / "train.csv"
        self.holdout_csv = self.tmp_path / "holdout.csv"
        write_csv(train_rows, self.train_csv)
        write_csv(holdout_rows, self.holdout_csv)

        self.model_path = self.tmp_path / "model.joblib"

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_train_produces_loadable_bundle(self):
        model, feature_cols = train_model(str(self.train_csv))
        joblib.dump({"model": model, "feature_cols": feature_cols}, self.model_path)

        bundle = joblib.load(self.model_path)
        self.assertIn("model", bundle)
        self.assertIn("feature_cols", bundle)
        self.assertGreater(len(bundle["feature_cols"]), 0)

    def test_evaluate_returns_plausible_metrics(self):
        model, feature_cols = train_model(str(self.train_csv))
        joblib.dump({"model": model, "feature_cols": feature_cols}, self.model_path)

        metrics = evaluate_model(str(self.model_path), str(self.holdout_csv))

        for key in ("roc_auc", "f1", "precision", "recall", "accuracy"):
            self.assertIn(key, metrics)
            self.assertGreaterEqual(metrics[key], 0.0)
            self.assertLessEqual(metrics[key], 1.0)
        self.assertEqual(metrics["n_holdout"], 500)

    def test_evaluate_better_than_random_guessing(self):
        model, feature_cols = train_model(str(self.train_csv))
        joblib.dump({"model": model, "feature_cols": feature_cols}, self.model_path)

        metrics = evaluate_model(str(self.model_path), str(self.holdout_csv))
        self.assertGreater(metrics["roc_auc"], 0.55)


if __name__ == "__main__":
    unittest.main()
