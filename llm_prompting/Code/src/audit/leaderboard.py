"""The results table, built to sit beside the paper's Table 4.

Reads every prediction file for a scope, rescores it against the frozen labels, and
reports accuracy with a bootstrap confidence interval, per-class F1, and a
two-proportion z-test against the best published fine-tuned baseline.

Nothing here calls an API: it only reads files already on disk.

Run:  python -m src.audit.leaderboard [scope]
"""

import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.config import (
    LABEL_MAPPING,
    PUBLISHED_BASELINES,
    prompting_dir,
)
from src.common.data import load_bnsentmix, scope_indices
from src.common.io_utils import load_jsonl
from src.common.metrics import sentiment_metrics
from src.prompting.prompts import parse_predictions_name

# these audits analyse the BnSentMix Mixed label specifically
DATASET = "bnsentmix"
PREDICTIONS = prompting_dir(DATASET, "predictions")
PUBLISHED_TEST_ROWS = 3003


def bootstrap_ci(correct, resamples=4000, seed=42):
    rng = random.Random(seed)
    n = len(correct)
    if n == 0:
        return float("nan"), float("nan")
    samples = sorted(
        sum(correct[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples)
    )
    return samples[int(0.025 * resamples)], samples[int(0.975 * resamples) - 1]


def two_proportion_z(accuracy, rows, target, target_rows):
    """z-test of two independent proportions, treating the published figure as fixed."""
    successes = accuracy * rows
    target_successes = target * target_rows
    pooled = (successes + target_successes) / (rows + target_rows)
    standard_error = math.sqrt(pooled * (1 - pooled) * (1 / rows + 1 / target_rows))
    if standard_error == 0:
        return 0.0, 1.0
    z = (accuracy - target) / standard_error
    p = 0.5 * math.erfc(abs(z) / math.sqrt(2))
    return z, p


def collect(scope, dataset):
    rows_in_scope = scope_indices(scope, dataset)
    runs = []
    for path in sorted(PREDICTIONS.glob(f"*__{scope}.jsonl")):
        provider, model, prompt_name, parsed_scope = parse_predictions_name(path)
        records = {record["index"]: record["prediction"] for record in load_jsonl(path)}
        labelled = [i for i in rows_in_scope if records.get(i) is not None]
        if not labelled:
            continue
        runs.append({
            "provider": provider,
            "model": model,
            "prompt": prompt_name,
            "records": records,
            "rows": labelled,
            "complete": len(labelled) == len(rows_in_scope),
        })
    return runs, rows_in_scope


def score(run, dataset):
    labels = [dataset[i]["Label"] for i in run["rows"]]
    predicted = [run["records"][i] for i in run["rows"]]
    metrics = sentiment_metrics(labels, predicted)
    correct = [1 if p == g else 0 for p, g in zip(predicted, labels)]
    low, high = bootstrap_ci(correct)
    return metrics, low, high


def main():
    scope = sys.argv[1] if len(sys.argv) > 1 else "test"
    dataset = load_bnsentmix()
    runs, rows_in_scope = collect(scope, dataset)

    best_name, best = max(PUBLISHED_BASELINES.items(), key=lambda kv: kv[1]["test"]["accuracy"])
    target = best["test"]["accuracy"]

    print("=" * 118)
    print(f"LEADERBOARD - {scope} split ({len(rows_in_scope)} rows)")
    print("=" * 118)
    header = (f"{'model':<34}{'prompt':<22}{'n':>6}{'accuracy':>10}{'95% CI':>20}"
              f"{'precision':>11}{'recall':>8}{'f1':>8}{'MixedF1':>9}{'p vs best':>11}")
    print(header)
    print("-" * len(header))

    results = []
    for run in runs:
        metrics, low, high = score(run, dataset)
        weighted = metrics["weighted"]
        if run["complete"]:
            z, p = two_proportion_z(weighted["accuracy"], len(run["rows"]), target, PUBLISHED_TEST_ROWS)
            p_text = f"{p:.2e}" if p < 0.01 else f"{p:.3f}"
        else:
            z, p, p_text = float("nan"), float("nan"), "-"
        results.append({
            "provider": run["provider"], "model": run["model"], "prompt": run["prompt"],
            "rows": len(run["rows"]), "complete": run["complete"], "metrics": metrics,
            "ci": [low, high], "z": z, "p": p,
        })
        marker = "" if run["complete"] else "  (partial)"
        print(f"{run['model']:<34}{run['prompt']:<22}{len(run['rows']):>6}"
              f"{weighted['accuracy']:>10.4f}{f'[{low:.3f},{high:.3f}]':>20}"
              f"{weighted['precision']:>11.4f}{weighted['recall']:>8.4f}"
              f"{weighted['f1']:>8.4f}{metrics['per_class']['Mixed']['f1']:>9.3f}"
              f"{p_text:>11}{marker}")

    print()
    print(f"PUBLISHED BASELINES - fine-tuned on train, evaluated on the paper's test split")
    print("-" * len(header))
    for name, entry in sorted(PUBLISHED_BASELINES.items(), key=lambda kv: -kv[1]["test"]["accuracy"]):
        test = entry["test"]
        print(f"{entry['paper_name']:<34}{'fine-tuned':<22}{PUBLISHED_TEST_ROWS:>6}"
              f"{test['accuracy']:>10.4f}{'-':>20}{test['precision']:>11.4f}"
              f"{test['recall']:>8.4f}{test['f1']:>8.4f}{'-':>9}{'-':>11}")
    print()
    print(f"  reference for the p column: {best['paper_name']} at {target:.4f}")

    ladder = sorted(
        (r for r in results if r["complete"] and r["prompt"] == "zero_shot_v1"),
        key=lambda r: -r["metrics"]["weighted"]["accuracy"],
    )
    if ladder:
        print()
        print("=" * 118)
        print("SIZE LADDER - zero-shot only, ordered by accuracy")
        print("=" * 118)
        for entry in ladder:
            beats = "beats" if entry["metrics"]["weighted"]["accuracy"] > target else "does not beat"
            significance = "significant" if entry["p"] < 0.05 else "not significant"
            print(f"  {entry['model']:<34} acc {entry['metrics']['weighted']['accuracy']:.4f}"
                  f"   {beats} {target:.4f}   p={entry['p']:.2e}  {significance}")

    out_dir = prompting_dir(DATASET, "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"leaderboard__{scope}.json"
    out_path.write_text(
        json.dumps({
            "scope": scope,
            "rows_in_scope": len(rows_in_scope),
            "reference_baseline": {"name": best["paper_name"], "accuracy": target},
            "runs": results,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    print()
    print("saved:", out_path)


if __name__ == "__main__":
    main()
