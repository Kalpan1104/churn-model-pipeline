import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

PRIMARY_METRIC = "roc_auc"
MIN_IMPROVEMENT = 0.002

MODEL_CARD_HEADER = "# Model Card: Churn Prediction\n\n"
MODEL_CARD_INTRO = (
    "This file is updated automatically by `.github/workflows/retrain.yml` whenever a "
    "retrained candidate model beats the current production model on the fixed holdout set. "
    "It is not meant to be edited by hand.\n\n"
)


def load_metrics(path):
    if not Path(path).exists():
        return None
    with open(path) as f:
        return json.load(f)


def should_promote(candidate_metrics, production_metrics):
    if production_metrics is None:
        return True, "no production model exists yet"

    candidate_score = candidate_metrics[PRIMARY_METRIC]
    production_score = production_metrics[PRIMARY_METRIC]

    if candidate_score > production_score + MIN_IMPROVEMENT:
        return True, f"{PRIMARY_METRIC} improved {production_score:.4f} -> {candidate_score:.4f}"

    return False, f"{PRIMARY_METRIC} did not improve enough ({production_score:.4f} -> {candidate_score:.4f})"


def append_model_card(model_card_path, metrics, reason):
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    if not Path(model_card_path).exists():
        with open(model_card_path, "w") as f:
            f.write(MODEL_CARD_HEADER)
            f.write(MODEL_CARD_INTRO)
            f.write("## Current production metrics\n\n")
            f.write("| metric | value |\n|---|---|\n")
            for key in ("roc_auc", "f1", "precision", "recall", "accuracy", "n_holdout"):
                f.write(f"| {key} | {metrics[key]:.4f} |\n" if isinstance(metrics[key], float) else f"| {key} | {metrics[key]} |\n")
            f.write("\n## Promotion history\n\n")
    else:
        with open(model_card_path) as f:
            content = f.read()
        marker = "## Current production metrics\n\n"
        start = content.index(marker) + len(marker)
        end = content.index("\n## Promotion history", start)
        new_table = "| metric | value |\n|---|---|\n"
        for key in ("roc_auc", "f1", "precision", "recall", "accuracy", "n_holdout"):
            new_table += f"| {key} | {metrics[key]:.4f} |\n" if isinstance(metrics[key], float) else f"| {key} | {metrics[key]} |\n"
        content = content[:start] + new_table + content[end:]
        with open(model_card_path, "w") as f:
            f.write(content)

    with open(model_card_path, "a") as f:
        f.write(f"- **{timestamp}** — promoted. {reason}\n")


def promote(candidate_model_path, candidate_metrics_path, production_model_path, production_metrics_path, model_card_path):
    candidate_metrics = load_metrics(candidate_metrics_path)
    production_metrics = load_metrics(production_metrics_path)

    promote_flag, reason = should_promote(candidate_metrics, production_metrics)

    if promote_flag:
        Path(production_model_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(candidate_model_path, production_model_path)
        shutil.copyfile(candidate_metrics_path, production_metrics_path)
        append_model_card(model_card_path, candidate_metrics, reason)

    return promote_flag, reason


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-model", default="models/candidate_model.joblib")
    parser.add_argument("--candidate-metrics", default="models/candidate_metrics.json")
    parser.add_argument("--production-model", default="models/production_model.joblib")
    parser.add_argument("--production-metrics", default="models/production_metrics.json")
    parser.add_argument("--model-card", default="models/MODEL_CARD.md")
    args = parser.parse_args()

    promoted, reason = promote(
        args.candidate_model,
        args.candidate_metrics,
        args.production_model,
        args.production_metrics,
        args.model_card,
    )

    print(f"promoted={promoted}: {reason}")

    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with open(output_path, "a") as f:
            f.write(f"promoted={'true' if promoted else 'false'}\n")


if __name__ == "__main__":
    main()
