"""The shot-count ablation, across prompt families and models.

For each model, zero / two / five examples per label are compared on identical rows
with a paired McNemar test. A correction for the number of comparisons is applied,
because running many tests on one dataset produces a nominal hit by chance alone.

The ablation is run once per prompt family:

  elaborate   label definitions plus seven numbered rules
  bare        the four label names only

That matters because the two families give different answers, so reporting only one
of them would be misleading.

Run:  python -m src.audit.shot_curve [scope]
"""

import json
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scipy.stats import chi2

from src.common.config import prompting_dir
from src.common.data import load_bnsentmix, scope_indices
from src.common.io_utils import load_jsonl
from src.prompting.prompts import parse_predictions_name

# these audits analyse the BnSentMix Mixed label specifically
DATASET = "bnsentmix"
PREDICTIONS = prompting_dir(DATASET, "predictions")

FAMILIES = {
    "elaborate": [("zero", "zero_shot_v1"), ("two", "two_shot_v1"), ("five", "five_shot_v1")],
    "bare": [
        ("zero", "zero_shot_minimal_v1"),
        ("two", "two_shot_minimal_v1"),
        ("five", "five_shot_minimal_v1"),
    ],
}

SHORT = {
    "qwen_qwen3.8-27b": "Qwen3.8-27B",
    "openai_gpt-oss-20b": "gpt-oss-20B",
    "openai_gpt-oss-120b": "gpt-oss-120B",
    "allam-2-7b": "ALLaM-7B",
    "google_gemini-3.8-flash": "Gemini-3.8-Flash",
    "meta_muse-spark-1.3-contributor": "Muse-Spark-1.3",
    "deepseek_deepseek-v4.1-flash": "DeepSeek-V4.1",
}


def load(provider, model, prompt, scope):
    safe = model.replace("/", "_").replace(":", "_")
    path = PREDICTIONS / f"{provider}_{safe}_sentiment_{prompt}__{scope}.jsonl"
    if not path.exists():
        return None
    return {record["index"]: record["prediction"] for record in load_jsonl(path)}


def mcnemar(first, second, gold, rows):
    """Paired test on identical rows. Returns (gained, lost, p)."""
    gained = sum(1 for i in rows if first[i] != gold[i] and second[i] == gold[i])
    lost = sum(1 for i in rows if first[i] == gold[i] and second[i] != gold[i])
    if gained + lost == 0:
        return gained, lost, 1.0
    statistic = (abs(gained - lost) - 1) ** 2 / (gained + lost)
    return gained, lost, float(chi2.sf(statistic, 1))


def discover(conditions, scope):
    """Every provider/model that has all three conditions for one family."""
    wanted = {name for _, name in conditions}
    found = {}
    for path in sorted(PREDICTIONS.glob(f"*__{scope}.jsonl")):
        provider, model, prompt_name, _ = parse_predictions_name(path)
        if prompt_name in wanted:
            found.setdefault((provider, model), set()).add(prompt_name)
    return sorted(key for key, names in found.items() if names >= wanted)


def run_family(family, conditions, scope, gold, rows_in_scope):
    complete = discover(conditions, scope)
    print("=" * 100)
    print(f"SHOT-COUNT ABLATION - {family} prompt - {scope} split ({len(rows_in_scope)} rows)")
    print("=" * 100)
    if not complete:
        print("  no model has all three conditions for this family")
        print()
        return None

    models = {}
    tested = 0
    for provider, model in complete:
        runs = {}
        for label, prompt_name in conditions:
            records = load(provider, model, prompt_name, scope)
            if records is None:
                runs = None
                break
            runs[label] = records
        if not runs:
            continue

        rows = [
            i for i in rows_in_scope
            if all(runs[label].get(i) is not None for label, _ in conditions)
        ]
        if not rows:
            continue

        accuracies = {
            label: sum(1 for i in rows if runs[label][i] == gold[i]) / len(rows)
            for label, _ in conditions
        }
        if accuracies["zero"] < accuracies["two"] < accuracies["five"]:
            shape = "rising"
        elif accuracies["zero"] > accuracies["two"] > accuracies["five"]:
            shape = "falling"
        else:
            shape = "non-monotone"

        curve = "   ".join(f"{label} {accuracies[label]:.4f}" for label, _ in conditions)
        print(f"  {SHORT.get(model, model):<18}{curve}   -> {shape}")

        entry = {"accuracy": {k: round(v, 4) for k, v in accuracies.items()},
                 "shape": shape, "pairs": {}}
        for (left_label, _), (right_label, _) in combinations(conditions, 2):
            gained, lost, p = mcnemar(runs[left_label], runs[right_label], gold, rows)
            same = sum(1 for i in rows if runs[left_label][i] == runs[right_label][i]) / len(rows)
            entry["pairs"][f"{left_label}->{right_label}"] = {
                "delta": round(accuracies[right_label] - accuracies[left_label], 4),
                "gained": gained, "lost": lost,
                "identical": round(same, 4), "p": round(p, 6),
            }
            tested += 1
            print(f"      {left_label:>4} -> {right_label:<4} "
                  f"{accuracies[right_label] - accuracies[left_label]:+.4f}   "
                  f"gained {gained:<4} lost {lost:<4} identical {same:.1%}   p={p:.4f}")
        models[model] = entry
        print()

    threshold = 0.05 / tested if tested else 0.05
    survivors = [
        (model, pair, data["p"], data["delta"])
        for model, entry in models.items()
        for pair, data in entry["pairs"].items()
        if data["p"] < threshold
    ]
    print(f"  {tested} comparisons, Bonferroni threshold {threshold:.4f}")
    if survivors:
        for model, pair, p, delta in survivors:
            print(f"    {SHORT.get(model, model)} {pair}  delta {delta:+.4f}  "
                  f"p={p:.2e}  survives")
    else:
        print("    none survive correction")

    shapes = sorted({entry["shape"] for entry in models.values()})
    print(f"  shapes across models: {shapes}")
    print()

    return {
        "family": family,
        "models": models,
        "comparisons": tested,
        "bonferroni_threshold": round(threshold, 4),
        "shapes": shapes,
    }


def main():
    scope = sys.argv[1] if len(sys.argv) > 1 else "test"
    dataset = load_bnsentmix()
    gold = list(dataset["Label"])
    rows_in_scope = scope_indices(scope, dataset)

    payload = {"scope": scope, "families": {}}
    for family, conditions in FAMILIES.items():
        result = run_family(family, conditions, scope, gold, rows_in_scope)
        if result:
            payload["families"][family] = result

    out_dir = prompting_dir(DATASET, "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"shot_curve__{scope}.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("saved:", out_path)


if __name__ == "__main__":
    main()
