"""Charts for SentMix-3L.

Separate from the BnSentMix charts because those are finalised figures and this dataset
needs a different headline plot: the relationship between the share of invalid-label
predictions and reported accuracy, which is the finding that the mismatch accounts for
the spread between models.

Reads the saved prediction and metrics files only. No API calls, so it can be re-run
after every new run lands.

Run:  python -m src.audit.sentmix3l_charts
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.common import datasets
from src.common.config import SENTMIX3L_BASELINES, prompting_dir
from src.common.io_utils import load_jsonl
from src.common.metrics import sentiment_metrics

DATASET = "sentmix3l"
PREDICTIONS = prompting_dir(DATASET, "predictions")
CHARTS = prompting_dir(DATASET, "charts")
MIXED = 3

SHOTS = {"zero_shot_minimal_v1": "zero-shot", "two_shot_minimal_v1": "two-shot",
         "five_shot_minimal_v1": "five-shot"}

MODEL_COLOURS = {
    "meta_muse-spark-1.3-contributor": "#1f77b4",
    "google_gemini-3.8-flash": "#ff7f0e",
    "zai-org_GLM-5.3": "#2ca02c",
    "openai_gpt-oss-20b": "#d62728",
    "deepseek_deepseek-v4.1-flash": "#9467bd",
    "gpt-5.6-sol": "#8c564b",
    "meta-llama_Llama-3.3-70B-Instruct": "#e377c2",
}


def prettify(model):
    name = model.replace("_", "/", 1).split("/", 1)[-1] if "_" in model else model
    name = model.replace("meta_muse-spark-1.3-contributor", "Muse-Spark")
    name = name.replace("google_gemini-3.8-flash", "Gemini-3.8-Flash")
    name = name.replace("zai-org_GLM-5.3", "GLM-5.3")
    name = name.replace("openai_gpt-oss-20b", "gpt-oss-20B")
    name = name.replace("deepseek_deepseek-v4.1-flash", "DeepSeek-v4.1")
    name = name.replace("gpt-5.6-sol", "GPT-5.6-Sol")
    name = name.replace("meta-llama_Llama-3.3-70B-Instruct", "Llama-3.3-70B")
    return name


def collect():
    dataset = datasets.load(DATASET)
    gold = dataset.gold
    rows = dataset.indices("full")
    runs = []
    for path in sorted(PREDICTIONS.glob("*.jsonl")):
        records = {r["index"]: r["prediction"] for r in load_jsonl(path)}
        complete = all(i in records for i in rows)
        usable = [i for i in rows if i in records]
        if len(usable) < 50:
            continue
        labels = [gold[i] for i in usable]
        predicted = [records[i] for i in usable]
        metrics = sentiment_metrics(labels, predicted)
        invalid = [i for i in usable if records[i] == MIXED]
        valid = [i for i in usable if records[i] != MIXED]
        stem = path.stem.replace("commandcode_", "").replace("groq_", "").replace(
            "huggingface_", "")
        model, _, prompt = stem.partition("_sentiment_")
        prompt = prompt.replace("__full", "")
        runs.append({
            "model": model,
            "prompt": prompt,
            "complete": complete,
            "n": len(usable),
            "accuracy": metrics["weighted"]["accuracy"],
            "f1": metrics["weighted"]["f1"],
            "invalid": len(invalid) / len(usable),
            "on_valid": (sum(1 for i in valid if records[i] == gold[i]) / len(valid))
                        if valid else float("nan"),
        })
    return runs


def chart_all_runs(runs, out_path):
    best = max(SENTMIX3L_BASELINES.values(), key=lambda e: e["weighted_f1"])
    ordered = sorted(runs, key=lambda r: r["f1"])
    fig, ax = plt.subplots(figsize=(11, max(6, 0.42 * len(ordered))))
    names = [f"{prettify(r['model'])} {SHOTS.get(r['prompt'], r['prompt'])}"
             + ("" if r["complete"] else " *") for r in ordered]
    values = [r["f1"] for r in ordered]
    colours = [MODEL_COLOURS.get(r["model"], "#7f7f7f") for r in ordered]
    bars = ax.barh(names, values, color=colours)
    for bar, run in zip(bars, ordered):
        ax.text(bar.get_width() + 0.006, bar.get_y() + bar.get_height() / 2,
                f"{run['f1']:.3f}", va="center", fontsize=9)
    ax.axvline(best["weighted_f1"], color="black", linestyle="--", linewidth=1.4)
    ax.text(best["weighted_f1"] + 0.004, len(ordered) - 0.6,
            f"{best['paper_name']} (best published) {best['weighted_f1']:.2f}",
            fontsize=9, va="top")
    ax.set_xlabel("weighted F1")
    ax.set_xlim(0, min(1.0, max(values) + 0.09))
    ax.set_title(f"SentMix-3L, prompt engineering only, no training ({len(runs)} runs)")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print("wrote", out_path.name)


def chart_invalid_vs_accuracy(runs, out_path):
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    for run in runs:
        marker = "o" if run["complete"] else "^"
        ax.scatter(run["invalid"] * 100, run["accuracy"], s=110,
                   color=MODEL_COLOURS.get(run["model"], "#7f7f7f"),
                   edgecolor="black", linewidth=0.7, marker=marker, zorder=3)
        ax.annotate(f"{prettify(run['model'])} {SHOTS.get(run['prompt'], '')}",
                    (run["invalid"] * 100, run["accuracy"]),
                    textcoords="offset points", xytext=(7, -3), fontsize=7.5)
    ax.set_xlabel("share of predictions using the label the corpus does not have (%)")
    ax.set_ylabel("reported accuracy")
    ax.set_title("SentMix-3L: the label-set mismatch tracks the reported score")
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print("wrote", out_path.name)


def chart_shot_curves(runs, out_path):
    order = ["zero_shot_minimal_v1", "two_shot_minimal_v1", "five_shot_minimal_v1"]
    models = []
    for run in runs:
        if run["model"] not in models:
            models.append(run["model"])
    fig, ax = plt.subplots(figsize=(9, 6))
    for model in models:
        points = {r["prompt"]: r for r in runs if r["model"] == model}
        if not all(p in points and points[p]["complete"] for p in order):
            continue
        ax.plot(range(3), [points[p]["f1"] for p in order], marker="o", linewidth=2,
                color=MODEL_COLOURS.get(model, "#7f7f7f"), label=prettify(model))
    ax.set_xticks(range(3))
    ax.set_xticklabels(["zero-shot", "two-shot", "five-shot"])
    ax.set_xlabel("demonstrations")
    ax.set_ylabel("weighted F1")
    ax.set_title("SentMix-3L: shot count against weighted F1")
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print("wrote", out_path.name)


def main():
    runs = collect()
    CHARTS.mkdir(parents=True, exist_ok=True)
    print(f"{len(runs)} runs, {sum(1 for r in runs if r['complete'])} complete")
    chart_all_runs(runs, CHARTS / "chart__all_runs.png")
    chart_invalid_vs_accuracy(runs, CHARTS / "chart__invalid_vs_accuracy.png")
    chart_shot_curves(runs, CHARTS / "chart__shot_curves.png")

    out = CHARTS / "chart_data.json"
    out.write_text(json.dumps(runs, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("wrote", out.name)


if __name__ == "__main__":
    main()
