"""Analyze XAI Prompt Feature Ablation results on SentMix-3L.

Computes accuracy, macro F1, and invalid label rates across single-element ablations
to determine which prompt rules cause or suppress specific model behaviors.
"""

import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.datasets import load
from src.common.io_utils import load_jsonl
from src.common.metrics import sentiment_metrics
from src.common.config import prompting_dir

DATASET = "sentmix3l"
PREDICTIONS_DIR = prompting_dir(DATASET, "predictions")
RESULTS_DIR = prompting_dir(DATASET, "results")

ABLATION_NAMES = [
    ("zero_shot_v1", "Baseline (Full 7 Rules)", "Intact elaborate prompt"),
    ("zero_shot_ablate_drop_mixed_def_v1", "Drop Mixed Def", "Removes definition of Mixed"),
    ("zero_shot_ablate_drop_rule1_v1", "Drop Rule 1", "Removes bilingual understanding rule"),
    ("zero_shot_ablate_drop_rule2_v1", "Drop Rule 2", "Removes Banglish lexicon cues (bhalo, etc.)"),
    ("zero_shot_ablate_drop_rule3_v1", "Drop Rule 3", "Removes sentence-level interpretation rule"),
    ("zero_shot_ablate_drop_rule4_v1", "Drop Rule 4", "Removes 'Use Mixed ONLY when BOTH' rule"),
    ("zero_shot_ablate_drop_rule5_v1", "Drop Rule 5", "Removes 'If unclear, choose Neutral' rule"),
    ("zero_shot_ablate_drop_rule6_v1", "Drop Rule 6", "Removes 'Do not translate' rule"),
    ("zero_shot_ablate_drop_rule7_v1", "Drop Rule 7", "Removes 'Do not invent labels' rule"),
]


def load_run(provider, safe_model, prompt_name, scope="full"):
    filename = f"{provider}_{safe_model}_sentiment_{prompt_name}__{scope}.jsonl"
    path = PREDICTIONS_DIR / filename
    if not path.exists():
        return None
    records = {r["index"]: r["prediction"] for r in load_jsonl(path)}
    return records if records else None


def analyze(provider="groq", model="qwen/qwen3.8-27b", scope="full"):
    safe_model = model.replace("/", "_").replace(":", "_")
    dataset = load(DATASET)
    gold = dataset.gold
    indices = dataset.indices(scope)
    total_rows = len(indices)

    runs = {}
    for prompt_key, short_name, desc in ABLATION_NAMES:
        rec = load_run(provider, safe_model, prompt_key, scope)
        if rec and len(rec) >= total_rows:
            runs[prompt_key] = rec

    print("=" * 115)
    print(f"XAI PROMPT FEATURE ABLATION ANALYSIS: {model} on {DATASET.upper()} ({len(runs)}/{len(ABLATION_NAMES)} finished)")
    print("=" * 115)
    header = (
        f"  {'Ablation Variant':<32}{'Acc':>8}{'Delta Acc':>11}{'F1':>8}"
        f"{'Delta F1':>10}{'Invalid (Mixed)':>17}{'Delta Inv':>11}{'Acc|Valid':>11}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    base_key = "zero_shot_v1"
    base_metrics = None
    base_inv = 0.0

    if base_key in runs:
        labels = [gold[i] for i in indices]
        preds = [runs[base_key][i] for i in indices]
        base_metrics = sentiment_metrics(labels, preds)
        base_inv = sum(1 for p in preds if p == 3) / total_rows

    results = {}

    for prompt_key, short_name, desc in ABLATION_NAMES:
        if prompt_key not in runs:
            continue
        preds = [runs[prompt_key][i] for i in indices]
        labels = [gold[i] for i in indices]
        m = sentiment_metrics(labels, preds)
        acc = m["weighted"]["accuracy"]
        f1 = m["weighted"]["f1"]
        mixed_count = sum(1 for p in preds if p == 3)
        inv_rate = mixed_count / total_rows

        valid_idx = [i for i, p in enumerate(preds) if p != 3]
        acc_valid = sum(1 for i in valid_idx if preds[i] == labels[i]) / len(valid_idx) if valid_idx else 0.0

        d_acc = (acc - base_metrics["weighted"]["accuracy"]) if base_metrics else 0.0
        d_f1 = (f1 - base_metrics["weighted"]["f1"]) if base_metrics else 0.0
        d_inv = (inv_rate - base_inv) if base_metrics else 0.0

        d_acc_str = f"{d_acc:+.4f}" if prompt_key != base_key else "---"
        d_f1_str = f"{d_f1:+.4f}" if prompt_key != base_key else "---"
        d_inv_str = f"{d_inv:+.2%}" if prompt_key != base_key else "---"

        print(
            f"  {short_name:<32}{acc:>8.4f}{d_acc_str:>11}{f1:>8.4f}"
            f"{d_f1_str:>10}{inv_rate:>16.1%}{d_inv_str:>11}{acc_valid:>11.4f}"
        )

        results[prompt_key] = {
            "name": short_name,
            "description": desc,
            "accuracy": acc,
            "delta_acc": d_acc,
            "f1": f1,
            "delta_f1": d_f1,
            "invalid_rate": inv_rate,
            "delta_invalid": d_inv,
            "accuracy_on_valid": acc_valid,
            "per_class": m["per_class"],
        }

    out_file = RESULTS_DIR / f"xai_ablation_{safe_model}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nSaved XAI summary to:", out_file)


if __name__ == "__main__":
    analyze()
