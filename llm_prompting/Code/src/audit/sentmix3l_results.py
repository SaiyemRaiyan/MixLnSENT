"""SentMix-3L results table.

Reports each completed run against the baselines published for this dataset. Because
the BnSentMix prompt is applied verbatim it offers four labels while the corpus has
three, so a Mixed prediction cannot be correct. The table therefore reports both the
accuracy as measured and the accuracy over rows where the model gave a valid label,
so the cost of the label-set mismatch stays visible rather than buried.

Run:  python -m src.audit.sentmix3l_results
"""

import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common import datasets
from src.common.config import SENTMIX3L_BASELINES, prompting_dir
from src.common.io_utils import load_jsonl
from src.common.metrics import sentiment_metrics

DATASET = "sentmix3l"
PREDICTIONS = prompting_dir(DATASET, "predictions")
RESULTS = prompting_dir(DATASET, "results")

MIXED = 3  # never a gold label here, but the prompt offers it


def bootstrap_ci(correct, resamples=4000, seed=42):
    rng = random.Random(seed)
    n = len(correct)
    if n == 0:
        return float("nan"), float("nan")
    samples = sorted(
        sum(correct[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples)
    )
    return samples[int(0.025 * resamples)], samples[int(0.975 * resamples) - 1]


def collect(dataset, rows):
    runs = []
    for path in sorted(PREDICTIONS.glob("*.jsonl")):
        records = {record["index"]: record["prediction"] for record in load_jsonl(path)}
        labelled = [i for i in rows if records.get(i) is not None]
        if not labelled:
            continue
        runs.append({
            "path": path,
            "name": path.stem.replace("commandcode_", "").replace("groq_", "")
                    .replace("_sentiment_", " | ").replace("__full", ""),
            "records": records,
            "rows": labelled,
            "complete": len(labelled) == len(rows),
        })
    return runs


def main():
    dataset = datasets.load(DATASET)
    gold = dataset.gold
    rows = dataset.indices("full")
    runs = collect(dataset, rows)

    best_baseline_name, best_baseline = max(
        SENTMIX3L_BASELINES.items(), key=lambda kv: kv[1]["weighted_f1"]
    )
    target = best_baseline["weighted_f1"]

    print("=" * 108)
    print(f"SENTMIX-3L - {len(rows)} rows, {len(dataset.label_names())} labels, "
          f"prompt offers {len(dataset.label_names()) + 1}")
    print("=" * 108)
    print()
    header = (f"  {'model':<26}{'prompt':<24}{'n':>5}{'acc':>8}{'95% CI':>16}"
              f"{'f1':>8}{'invalid':>9}{'acc|valid':>11}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    payload = {}
    for run in sorted(runs, key=lambda r: -sum(
        1 for i in r["rows"] if r["records"][i] == gold[i]
    ) / len(r["rows"])):
        usable = run["rows"]
        labels = [gold[i] for i in usable]
        predicted = [run["records"][i] for i in usable]
        metrics = sentiment_metrics(labels, predicted)
        correct = [1 if p == g else 0 for p, g in zip(predicted, labels)]
        low, high = bootstrap_ci(correct)
        valid = [i for i in usable if run["records"][i] != MIXED]
        on_valid = sum(1 for i in valid if run["records"][i] == gold[i]) / len(valid)
        invalid_share = 1 - len(valid) / len(usable)

        model, _, prompt = run["name"].partition(" | ")
        flag = "" if run["complete"] else f"  ({len(usable)}/{len(rows)})"
        print(f"  {model:<26}{prompt:<24}{len(usable):>5}"
              f"{metrics['weighted']['accuracy']:>8.4f}{f'[{low:.3f},{high:.3f}]':>16}"
              f"{metrics['weighted']['f1']:>8.4f}{invalid_share:>8.1%}{on_valid:>11.4f}{flag}")

        payload[run["name"]] = {
            "rows": len(usable),
            "complete": run["complete"],
            "accuracy": round(metrics["weighted"]["accuracy"], 4),
            "f1": round(metrics["weighted"]["f1"], 4),
            "ci": [round(low, 4), round(high, 4)],
            "invalid_mixed_share": round(invalid_share, 4),
            "accuracy_on_valid_labels": round(on_valid, 4),
        }

    print()
    print("  PUBLISHED BASELINES - trained on synthetic data, tested on the natural set")
    print("  " + "-" * (len(header) - 2))
    for name, entry in sorted(SENTMIX3L_BASELINES.items(),
                              key=lambda kv: -kv[1]["weighted_f1"]):
        print(f"  {entry['paper_name']:<26}{'--':<24}{'--':>5}{'--':>8}{'--':>16}"
              f"{entry['weighted_f1']:>8.4f}{'--':>9}{'--':>11}")
    print()
    print(f"  best published: {best_baseline['paper_name']} at weighted f1 {target:.2f}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "sentmix3l_table.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print()
    print("saved:", out)


if __name__ == "__main__":
    main()
