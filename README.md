# Churn Model Pipeline

A customer-churn prediction model that retrains itself on a schedule, evaluates every candidate against a fixed holdout set, and only replaces the live model if it's actually better — with **GitHub Actions as the entire orchestration layer**. No Airflow, no external scheduler, nothing to keep running: the automation lives entirely in `.github/workflows/`.

This is a deliberate contrast with an Airflow-based pipeline: same "generate → train → evaluate → decide whether to promote" shape, built on CI/CD instead of a dedicated orchestrator. For a project this size — one model, weekly retrains — that's arguably the more honest choice; Airflow earns its complexity at a different scale.

## Prerequisites

- Python 3.10+
- A GitHub repo with Actions enabled (on by default for a new repo)
- No Docker, no separate scheduler, no OS-specific setup — this is the whole appeal of building it this way

## What it does

```
   generate  -- synthesizes this run's "new" customer data
          |     (seeded by ISO year+week, so it's different every
          |     week but reproducible if you re-run the same week)
          v
   train     -- trains a candidate RandomForest classifier
          |
          v
   evaluate  -- scores the candidate against data/holdout_test.csv,
          |     a FIXED set committed to the repo, so every week's
          |     candidate is judged on the exact same yardstick
          v
   promote   -- compares candidate vs. current production metrics;
          |     promotes only if roc_auc improves by a real margin
          |     (not just noise), and updates the model card either way
          v
   (if promoted) commit the new model + metrics + model card back to
   the repo, using the workflow's own GITHUB_TOKEN
```

The key design decision: **the holdout set never changes.** Training data regenerates every run to simulate new data arriving, but `data/holdout_test.csv` is generated once, committed, and never touched again — otherwise there'd be no fair way to tell if this week's model is actually better or if the test just got easier.

## Repo layout

```
churn-model-pipeline/
├── src/
│   ├── generate_data.py   synthetic subscription data (deterministic per seed)
│   ├── features.py        encodes categoricals, derives ratio features
│   ├── train.py            trains the candidate (class-weighted RandomForest)
│   ├── evaluate.py         scores a model against a holdout CSV
│   ├── promote.py          promotion decision + model card writer
│   ├── pr_summary.py       renders a candidate-vs-production comparison (reuses promote.py's logic)
│   └── predict.py          scores a batch of customers against the production model
├── models/
│   ├── production_model.joblib   the live model (committed)
│   ├── production_metrics.json   its metrics (committed)
│   └── MODEL_CARD.md             human-readable history, auto-updated
├── data/
│   └── holdout_test.csv   fixed evaluation set (committed, never regenerated)
├── tests/
└── .github/workflows/
    ├── ci.yml        tests on every push/PR
    ├── retrain.yml   the actual MLOps pipeline
    ├── pr-eval.yml   previews what a code change would do, before merge
    └── rollback.yml  manually revert to the previous promoted model
```

## The four workflows

| Workflow | Trigger | What it does | Can it push to the repo? |
|---|---|---|---|
| `ci.yml` | push / PR to `main` | installs deps, runs the test suite | no |
| `retrain.yml` | weekly cron (Mon 06:00 UTC) + manual | generates data, trains, evaluates, promotes if better | yes — commits the new model |
| `pr-eval.yml` | PR touching `src/`, `tests/`, or `requirements.txt` | trains a candidate from the PR's code, evaluates it, posts a metrics table to the run's **Job Summary** | no — read-only, just reports |
| `rollback.yml` | manual only (`workflow_dispatch`) | reverts `models/production_*` to a previous promoted commit | yes — commits the revert |

**`retrain.yml`** is the main pipeline, covered above. It needs `permissions: contents: write` because its last step commits the promoted model using the workflow's automatic `GITHUB_TOKEN`.

**`pr-eval.yml`** answers "if I merge this change to the training code, would it actually produce a better model?" *before* you merge it — trains a throwaway candidate using the PR's version of `src/`, evaluates it against the same fixed holdout, and writes a comparison table straight to the workflow run's Job Summary (the tab right there in the Actions UI — no extra permissions needed, unlike posting an actual PR comment, which would require `pull-requests: write`). It only triggers on PRs that touch code, not for something like a README-only change.

**`rollback.yml`** is a deliberately manual escape hatch — no schedule, `workflow_dispatch` only, triggered from the Actions tab. By default it finds the previous commit that touched `models/production_model.joblib` and restores that version; you can also pass a specific commit SHA. I tested both the normal path and the "only one promotion has ever happened, nothing to roll back to" edge case directly against real git repos before trusting this one — it correctly fails with a clear message in the second case rather than doing nothing silently.

All four need `permissions: contents: write` (`retrain.yml`, `rollback.yml`) or nothing extra at all (`ci.yml`, `pr-eval.yml`) — **if your repo's default is "Read repository contents permission,"** the two that need to commit will fail even with `contents: write` in the YAML. Check **Settings → Actions → General → Workflow permissions** and select **"Read and write permissions."** This is a repo-level setting the workflow file can't override by itself.

## Running it locally

```bash
git clone https://github.com/<your-username>/churn-model-pipeline.git
cd churn-model-pipeline
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run one retrain cycle by hand, exactly like `retrain.yml` does:
```bash
python -m src.generate_data --out data/train_latest.csv --n 4000 --seed 202610
python -m src.train --data data/train_latest.csv --out-model models/candidate_model.joblib
python -m src.evaluate --model models/candidate_model.joblib --holdout data/holdout_test.csv --out models/candidate_metrics.json
python -m src.promote
```
That last command prints `promoted=True: ...` or `promoted=False: ...` and says why. If it promoted, check `models/MODEL_CARD.md` — updated metrics table, new history line.

Preview what `pr-eval.yml` would report, without needing a PR:
```bash
python -m src.pr_summary --candidate models/candidate_metrics.json
```

Score customers against whatever's currently in production:
```bash
python -m src.predict --customers data/holdout_test.csv --out /tmp/scored.csv
```

## Testing

```bash
python -m unittest discover -s tests -v
```
24 tests. No Airflow-style "skip if not installed" complexity needed — everything runs with just `pandas`/`scikit-learn`, so CI and local runs behave identically. Covers the data generator (determinism, churn-rate sanity, that month-to-month actually churns more than two-year contracts), feature engineering (fixed dummy columns even when a category is absent from a batch), the train/evaluate round trip, the PR summary renderer, and all three promotion-decision branches (first-ever promotion, genuine improvement, correctly rejecting a worse candidate without touching production).

## Triggering things manually

Repo → **Actions** tab:
- **Retrain and Promote** → **Run workflow** — don't wait for Monday to see it work.
- **Rollback Production Model** → **Run workflow** — optionally fill in a specific commit SHA, or leave it blank to go back one promotion.

## Notes / things I'd change with more time

- `data/train_latest.csv` is regenerated synthetically every run rather than pulled from a real source — swap `generate_data.py`'s call for an actual data export step and everything downstream (train/evaluate/promote) stays the same.
- No post-deployment monitoring — promotion is based entirely on the one-time holdout score, not on how the model actually performs on live predictions afterward.
- `retrain.yml` pushes straight to `main`. For a team repo you'd want it opening a PR with the new model instead and requiring review before merge (`peter-evans/create-pull-request` is the common action for this) — left out here to keep the workflow readable as a first pass.
- `rollback.yml` only restores the model files, not the data the old model was trained on — fine for this project since training data isn't versioned per-model, but worth knowing if you extend this.

## License

MIT — see [LICENSE](LICENSE).
