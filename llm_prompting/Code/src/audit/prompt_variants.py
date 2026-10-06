"""Do the prompt's instructions help or hurt?

Compares the elaborate zero-shot prompt (seven numbered rules, Banglish lexicon
cues, explicit Mixed guidance) against variants that remove some or all of it, on
identical rows with a paired test.

The expectation was that a carefully written prompt would beat a bare one. If the
opposite holds, the instructions were not doing what they were written to do.

Run:  python -m src.audit.prompt_variants [scope]
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scipy.stats import chi2

from src.common.config import LABEL_MAPPING, prompting_dir
from src.common.data import load_bnsentmix, scope_indices
from src.common.io_utils import load_jsonl
from src.common.metrics import sentiment_metrics

# these audits analyse the BnSentMix Mixed label specifically
DATASET = "bnsentmix"
PREDICTIONS = prompting_dir(DATASET, "predictions")

# ordered from most to least instructive
VARIANTS = [
    ("zero_shot_v1", "elaborate", "7 rules + lexicon cues + Mixed guidance"),
    ("zero_shot_v3", "mixed-rule rewritten", "instruction 4 no longer requires a contrast word"),
    ("zero_shot_v2", "neutral-rule rewritten", "instruction 5 no longer offers a Neutral fallback"),
    ("zero_shot_minimal_v1", "bare", "label names only"),
]

MODELS = [
    ("groq", "qwen/qwen3.8-27b"),
    ("groq", "openai/gpt-oss-20b"),
    ("groq", "openai/gpt-oss-120b"),
]


def load(provider, model, prompt, scope):
    safe = model.replace("/", "_").replace(":", "_")
    path = PREDICTIONS / f"{provider}_{safe}_sentiment_{prompt}__{scope}.jsonl"
    if not path.exists():
        return None
    records = {record["index"]: record["prediction"] for record in load_jsonl(path)}
    return records if records else None


def mcnemar(first, second, gold, rows):
    gained = sum(1 for i in rows if first[i] != gold[i] and second[i] == gold[i])
    lost = sum(1 for i in rows if first[i] == gold[i] and second[i] != gold[i])
    if gained + lost == 0:
        return gained, lost, 1.0
    return gained, lost, float(chi2.sf((abs(gained - lost) - 1) ** 2 / (gained + lost), 1))


def main():
    scope = sys.argv[1] if len(sys.argv) > 1 else "test"
    dataset = load_bnsentmix()
    gold = list(dataset["Label"])
    rows_in_scope = scope_indices(scope, dataset)

    print("=" * 100)
    print(f"PROMPT VARIANTS - does the instruction text help?  ({scope} split)")
    print("=" * 100)
    print("  all variants use the same answer format and the same rows; only the")
    print("  instruction block differs")
    print()

    payload = {"scope": scope, "models": {}}
    comparisons = 0

    for provider, model in MODELS:
        runs = {}
        for prompt_name, short, _ in VARIANTS:
            records = load(provider, model, prompt_name, scope)
            if records is not None:
                runs[prompt_name] = records
        if len(runs) < 2:
            continue

        rows = [
            i for i in rows_in_scope
            if all(records.get(i) is not None for records in runs.values())
        ]
        if not rows:
            continue

        print(f"{model}")
        header = f"  {'variant':<24}{'accuracy':>10}{'f1':>9}{'Mixed P':>9}{'Mixed R':>9}{'Mixed f1':>10}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        entry = {"variants": {}, "pairs": {}}
        for prompt_name, short, _ in VARIANTS:
            if prompt_name not in runs:
                continue
            labels = [gold[i] for i in rows]
            predicted = [runs[prompt_name][i] for i in rows]
            metrics = sentiment_metrics(labels, predicted)
            weighted = metrics["weighted"]
            mixed = metrics["per_class"]["Mixed"]
            entry["variants"][short] = {
                "prompt": prompt_name,
                "accuracy": round(weighted["accuracy"], 4),
                "f1": round(weighted["f1"], 4),
                "mixed": {k: round(v, 4) for k, v in mixed.items()},
            }
            print(f"  {short:<24}{weighted['accuracy']:>10.4f}{weighted['f1']:>9.4f}"
                  f"{mixed['precision']:>9.3f}{mixed['recall']:>9.3f}{mixed['f1']:>10.3f}")

        base = "zero_shot_v1"
        if base in runs:
            print()
            print(f"  paired against the elaborate prompt:")
            for prompt_name, short, _ in VARIANTS:
                if prompt_name == base or prompt_name not in runs:
                    continue
                gained, lost, p = mcnemar(runs[base], runs[prompt_name], gold, rows)
                acc_base = sum(1 for i in rows if runs[base][i] == gold[i]) / len(rows)
                acc_other = sum(1 for i in rows if runs[prompt_name][i] == gold[i]) / len(rows)
                delta = acc_other - acc_base
                comparisons += 1
                entry["pairs"][short] = {
                    "delta": round(delta, 4), "gained": gained, "lost": lost, "p": round(p, 4)
                }
                sign = "BETTER" if delta > 0 else "worse"
                print(f"    {short:<24}{delta:+.4f}   gained {gained:<4} lost {lost:<4} "
                      f"p={p:.2e}   {sign}")

        payload["models"][model] = entry
        print()

    print("=" * 100)
    if comparisons:
        threshold = 0.05 / comparisons
        print(f"  {comparisons} comparisons; Bonferroni threshold {threshold:.4f}")
        survived = [
            (model, variant, data["p"], data["delta"])
            for model, entry in payload["models"].items()
            for variant, data in entry["pairs"].items()
            if data["p"] < threshold
        ]
        for model, variant, p, delta in survived:
            print(f"    {model} vs {variant}: delta {delta:+.4f} p={p:.2e}  survives")
        if not survived:
            print("    none survive correction")

        better = [
            (model, variant, data["delta"])
            for model, entry in payload["models"].items()
            for variant, data in entry["pairs"].items()
            if data["delta"] > 0 and data["p"] < threshold
        ]
        if better:
            print()
            print("  -> removing instructions improved accuracy significantly:")
            for model, variant, delta in better:
                print(f"     {model} without {variant}: {delta:+.4f}")
    payload["comparisons"] = comparisons

    out_dir = prompting_dir(DATASET, "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"prompt_variants__{scope}.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print()
    print("saved:", out_path)


if __name__ == "__main__":
    main()
