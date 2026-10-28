import argparse
import json

from src.promote import should_promote

METRIC_KEYS = ("roc_auc", "f1", "precision", "recall", "accuracy")


def render_summary(candidate_metrics, production_metrics):
    lines = ["## Candidate model evaluation", ""]
    lines.append("| metric | production | candidate |")
    lines.append("|---|---|---|")
    for key in METRIC_KEYS:
        cand_val = candidate_metrics[key]
        if production_metrics is None:
            prod_str = "n/a"
        else:
            prod_str = f"{production_metrics[key]:.4f}"
        lines.append(f"| {key} | {prod_str} | {cand_val:.4f} |")

    would_promote, reason = should_promote(candidate_metrics, production_metrics)
    verdict = "would promote" if would_promote else "would NOT promote"
    lines.append("")
    lines.append(f"**Verdict:** this candidate {verdict} in a real retrain run — {reason}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--production", default="models/production_metrics.json")
    args = parser.parse_args()

    with open(args.candidate) as f:
        candidate_metrics = json.load(f)

    production_metrics = None
    try:
        with open(args.production) as f:
            production_metrics = json.load(f)
    except FileNotFoundError:
        pass

    print(render_summary(candidate_metrics, production_metrics))


if __name__ == "__main__":
    main()
