"""Charts built from the saved prediction files.

Reads whatever is on disk and draws it. No API calls, so the figures can be
regenerated at any time.

Run:  python -m src.audit.charts [scope]
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
RESULTS = prompting_dir(DATASET, "results")

# readable short names for the axis; keys are the safe forms returned by
# parse_predictions_name, which replaces "/" with "_"
SHORT = {
    "qwen_qwen3.8-27b": "Qwen3.8-27B",
    "openai_gpt-oss-20b": "gpt-oss-20B",
    "openai_gpt-oss-120b": "gpt-oss-120B",
    "allam-2-7b": "ALLaM-7B",
    "meta-llama_Llama-3.3-70B-Instruct": "Llama-3.3-70B",
    "google_gemini-3.8-flash": "Gemini-3.8-Flash",
    "meta_muse-spark-1.3-contributor": "Muse-Spark-1.3",
    "deepseek_deepseek-v4.1-flash": "DeepSeek-V4.1",
    "gpt-5.6-sol": "GPT-5.6-Sol",
    "zai-org_GLM-5.3": "GLM-5.3",
}
PROMPT_SHORT = {
    "zero_shot_v1": "zero-shot",
    "zero_shot_v2": "zero-shot v2",
    "zero_shot_v3": "zero-shot v3",
    "zero_shot_minimal_v1": "bare",
    "two_shot_minimal_v1": "bare two-shot",
    "five_shot_minimal_v1": "bare five-shot",
    "two_shot_v1": "two-shot",
    "five_shot_v1": "five-shot",
    "few_shot_v1": "few-shot v1",
    "few_shot_v2": "few-shot v2",
    "annotation_definition_only_v1": "definition only",
}

COLOURS = {
    # bare family - greens, since it is the stronger prompt
    "bare": "#1a9850",
    "bare two-shot": "#66bd63",
    "bare five-shot": "#006837",
    # elaborate family - blues through reds
    "zero-shot": "#2c6fbb",
    "two-shot": "#f4a582",
    "five-shot": "#b2182b",
    "zero-shot v2": "#9970ab",
    "zero-shot v3": "#762a83",
    "few-shot v1": "#9970ab",
    "few-shot v2": "#762a83",
    "definition only": "#999999",
}


def collect(scope, dataset):
    rows_in_scope = scope_indices(scope, dataset)
    runs = []
    for path in sorted(PREDICTIONS.glob(f"*__{scope}.jsonl")):
        provider, model, prompt_name, _ = parse_predictions_name(path)
        if not path.name.startswith(("groq_", "huggingface_", "commandcode_", "gemini_")):
            continue
        if provider == "reannotation":
            continue
        records = {record["index"]: record["prediction"] for record in load_jsonl(path)}
        rows = [i for i in rows_in_scope if records.get(i) is not None]
        if len(rows) < len(rows_in_scope) * 0.95:
            continue
        labels = [dataset[i]["Label"] for i in rows]
        predicted = [records[i] for i in rows]
        metrics = sentiment_metrics(labels, predicted)
        weighted = metrics["weighted"]
        runs.append({
            "model": model, "prompt": prompt_name,
            "accuracy": weighted["accuracy"],
            # weighted recall equals accuracy by construction, so the informative
            # recall figures are the macro average and the per-class ones
            "recall": weighted["recall"],
            "f1": weighted["f1"],
            "macro_recall": metrics["macro"]["recall"],
            "macro_precision": metrics["macro"]["precision"],
            "macro_f1": metrics["macro"]["f1"],
            "per_class": {
                name: {
                    "precision": metrics["per_class"][name]["precision"],
                    "recall": metrics["per_class"][name]["recall"],
                    "f1": metrics["per_class"][name]["f1"],
                    "support": metrics["per_class"][name]["support"],
                }
                for name in LABEL_MAPPING.values()
            },
            "mixed_f1": metrics["per_class"]["Mixed"]["f1"],
            "rows": len(rows),
        })
    return runs


def best_baseline():
    name, entry = max(PUBLISHED_BASELINES.items(), key=lambda kv: kv[1]["test"]["accuracy"])
    return entry["paper_name"], entry["test"]["accuracy"]


def chart_all_runs(runs, baseline_name, baseline, out_path):
    ordered = sorted(runs, key=lambda r: r["accuracy"])
    # one line per bar: two-line labels collide once there are more than about
    # fifteen runs, which there now are
    labels = [
        f"{SHORT.get(r['model'], r['model'])}  -  {PROMPT_SHORT.get(r['prompt'], r['prompt'])}"
        for r in ordered
    ]
    values = [r["accuracy"] for r in ordered]
    colours = [COLOURS.get(PROMPT_SHORT.get(r["prompt"], ""), "#777777") for r in ordered]

    # scale the height with the number of bars so labels always have room
    height = max(9.0, 0.34 * len(ordered))
    figure, axes = plt.subplots(figsize=(13, height))
    bars = axes.barh(range(len(values)), values, color=colours, height=0.72)
    axes.set_yticks(range(len(values)))
    axes.set_yticklabels(labels, fontsize=8)
    axes.axvline(baseline, color="black", linestyle="--", linewidth=1.4,
                 label=f"best fine-tuned baseline ({baseline_name}) {baseline:.3f}")
    axes.set_xlabel("accuracy on the frozen test split")
    axes.set_title("Prompted LLMs vs fine-tuned baselines on BnSentMix\n"
                   f"({runs[0]['rows']:,} held-out rows; no model was trained)"
                   "   -   green = bare prompt, blue/orange/red = elaborate prompt",
                   fontsize=11)
    axes.set_xlim(0.30, max(0.88, max(values) + 0.05))
    axes.set_ylim(-0.7, len(values) - 0.3)
    axes.legend(loc="lower right", fontsize=9)
    axes.grid(axis="x", alpha=0.3)

    for bar, run in zip(bars, ordered):
        axes.text(bar.get_width() + 0.004, bar.get_y() + bar.get_height() / 2,
                  f"{run['accuracy']:.4f}", va="center", fontsize=7.5)

    figure.tight_layout()
    figure.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(figure)
    print("wrote", out_path.name)


def chart_shot_curves(runs, baseline, out_path):
    """Both prompt families: bare (solid) and elaborate (dashed)."""
    models = [
        ("google_gemini-3.8-flash", "Gemini-3.8-Flash", "#1a9850", "o"),
        ("deepseek_deepseek-v4.1-flash", "DeepSeek-V4.1", "#2c6fbb", "s"),
        ("meta_muse-spark-1.3-contributor", "Muse-Spark-1.3", "#9970ab", "D"),
        ("openai_gpt-oss-20b", "gpt-oss-20B", "#b2182b", "^"),
        ("qwen_qwen3.8-27b", "Qwen3.8-27B", "#f4a582", "v"),
        ("allam-2-7b", "ALLaM-7B", "#777777", "x"),
    ]
    families = {
        "bare": ("zero_shot_minimal_v1", "two_shot_minimal_v1", "five_shot_minimal_v1"),
        "elaborate": ("zero_shot_v1", "two_shot_v1", "five_shot_v1"),
    }
    lookup = {(r["model"], r["prompt"]): r["accuracy"] for r in runs}
    positions = [0, 2, 5]

    figure, axes = plt.subplots(figsize=(10, 6.5))
    plotted = False
    for family, prompts in families.items():
        style = "-" if family == "bare" else "--"
        for model, name, colour, marker in models:
            values = [lookup.get((model, prompt)) for prompt in prompts]
            if any(value is None for value in values):
                continue
            label = f"{name} ({family})"
            axes.plot(positions, values, marker=marker, color=colour,
                      linewidth=2, linestyle=style, markersize=7, label=label)
            plotted = True
    if not plotted:
        plt.close(figure)
        return

    axes.axhline(baseline, color="black", linestyle=":", linewidth=1.6,
                 label=f"best fine-tuned baseline {baseline:.3f}")
    axes.set_xticks(positions)
    axes.set_xlabel("examples per label")
    axes.set_ylabel("accuracy on the frozen test split")
    axes.set_title("Adding examples helps some models and hurts others\n"
                   "solid = bare prompt, dashed = elaborate prompt", fontsize=12)
    axes.grid(alpha=0.3)
    axes.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    print("wrote", out_path.name)


def chart_mixed(runs, out_path):
    """Wildcard: what the prompt does to the hardest class."""
    wanted = [
        ("qwen_qwen3.8-27b", "zero_shot_v1"),
        ("qwen_qwen3.8-27b", "zero_shot_minimal_v1"),
        ("openai_gpt-oss-20b", "zero_shot_v1"),
        ("openai_gpt-oss-20b", "zero_shot_minimal_v1"),
    ]
    lookup = {(r["model"], r["prompt"]): r for r in runs}
    present = [(m, p) for m, p in wanted if (m, p) in lookup]
    if len(present) < 2:
        return
    labels = [f"{SHORT.get(m, m)}\n{PROMPT_SHORT.get(p, p)}" for m, p in present]
    values = [lookup[key]["mixed_f1"] for key in present]

    figure, axes = plt.subplots(figsize=(9, 5))
    bars = axes.bar(range(len(values)), values, color="#2c6fbb", width=0.55)
    axes.set_xticks(range(len(values)))
    axes.set_xticklabels(labels, fontsize=9)
    axes.set_ylabel("f1 on the Mixed class")
    axes.set_title("The bare prompt trades Mixed away\n"
                   "overall accuracy rises while the hardest class is withheld", fontsize=12)
    axes.grid(axis="y", alpha=0.3)
    for bar, value in zip(bars, values):
        axes.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.3f}",
                  ha="center", fontsize=9)
    figure.tight_layout()
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    print("wrote", out_path.name)


def chart_per_class_recall(runs, out_path):
    """Recall per class for the zero-shot runs - accuracy alone hides the spread."""
    wanted = ["zero_shot_v1", "zero_shot_minimal_v1"]
    selected = [r for r in runs if r["prompt"] in wanted]
    if not selected:
        return
    names = list(LABEL_MAPPING.values())
    selected.sort(key=lambda r: -r["accuracy"])

    import numpy as np

    positions = np.arange(len(names))
    width = 0.8 / max(len(selected), 1)
    figure, axes = plt.subplots(figsize=(10, 5.5))
    palette = ["#2c6fbb", "#1a9850", "#b2182b", "#9970ab", "#f4a582"]
    for offset, run in enumerate(selected):
        values = [run["per_class"][name]["recall"] for name in names]
        label = f"{SHORT.get(run['model'], run['model'])} ({PROMPT_SHORT.get(run['prompt'], run['prompt'])})"
        axes.bar(positions + offset * width - 0.4 + width / 2, values, width,
                 label=label, color=palette[offset % len(palette)])
    axes.set_xticks(positions)
    axes.set_xticklabels(names)
    axes.set_ylabel("recall")
    axes.set_ylim(0, 1.0)
    axes.set_title("Recall by class - Mixed is where the predictions fail\n"
                   "zero-shot runs on the frozen test split", fontsize=12)
    axes.grid(axis="y", alpha=0.3)
    axes.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    print("wrote", out_path.name)


def main():
    scope = sys.argv[1] if len(sys.argv) > 1 else "test"
    dataset = load_bnsentmix()
    runs = collect(scope, dataset)
    if not runs:
        print("no runs found for scope", scope)
        return

    baseline_name, baseline = best_baseline()
    print("=" * 112)
    print(f"CHART DATA - {scope} split, {len(runs)} runs, baseline {baseline:.4f}")
    print("=" * 112)
    print("  weighted recall is not shown: it equals accuracy by construction")
    print()
    header = (f"  {'model':<14}{'prompt':<14}{'acc':>8}{'macroR':>8}{'f1':>8}"
              f"{'MixedP':>8}{'MixedR':>8}{'MixedF1':>9}   vs baseline")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for run in sorted(runs, key=lambda r: -r["accuracy"]):
        mixed = run["per_class"]["Mixed"]
        delta = run["accuracy"] - baseline
        note = f"beats {delta:+.4f}" if delta > 0 else f"below {delta:+.4f}"
        print(f"  {SHORT.get(run['model'], run['model']):<14}"
              f"{PROMPT_SHORT.get(run['prompt'], run['prompt']):<14}"
              f"{run['accuracy']:>8.4f}{run['macro_recall']:>8.4f}{run['f1']:>8.4f}"
              f"{mixed['precision']:>8.3f}{mixed['recall']:>8.3f}{mixed['f1']:>9.3f}   {note}")

    print()
    print("  per-class recall (accuracy alone hides which classes carry the errors)")
    print()
    names = list(LABEL_MAPPING.values())
    header = f"  {'model':<14}{'prompt':<14}" + "".join(f"{name:>10}" for name in names) + f"{'macro':>9}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for run in sorted(runs, key=lambda r: -r["accuracy"]):
        print(f"  {SHORT.get(run['model'], run['model']):<14}"
              f"{PROMPT_SHORT.get(run['prompt'], run['prompt']):<14}"
              + "".join(f"{run['per_class'][name]['recall']:>10.3f}" for name in names)
              + f"{run['macro_recall']:>9.3f}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    chart_all_runs(runs, baseline_name, baseline, RESULTS / f"chart__all_runs__{scope}.png")
    chart_shot_curves(runs, baseline, RESULTS / f"chart__shot_curves__{scope}.png")
    chart_mixed(runs, RESULTS / f"chart__mixed_tradeoff__{scope}.png")
    chart_per_class_recall(runs, RESULTS / f"chart__per_class_recall__{scope}.png")

    (RESULTS / f"chart_data__{scope}.json").write_text(
        json.dumps({"scope": scope, "baseline": baseline, "baseline_name": baseline_name,
                    "runs": runs}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
