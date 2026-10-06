import json
import re
from datetime import datetime, timezone

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from .config import LABEL_MAPPING, OUTPUTS_DIR, PUBLISHED_BASELINES

TABLE_EVAL_SETS = ("validation", "test")
TABLE_COLUMNS = ("accuracy", "precision", "recall", "f1")


def metrics_dir(experiment, dataset=None):
    """Metrics directory.

    Prompting runs are namespaced by dataset because the same prompt is applied to
    more than one corpus. The filtering experiment only ever runs on BnSentMix, so
    it passes no dataset and keeps its original layout.
    """
    base = OUTPUTS_DIR / experiment
    if dataset is not None:
        base = base / dataset
    return base / "metrics"


def _slug(value):
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "none"


def make_run_id(**parts):
    return "__".join(f"{_slug(key)}-{_slug(value)}" for key, value in parts.items())


def save_metrics(record, experiment, **parts):
    """Write a metrics record with a self-identifying file name."""
    directory = metrics_dir(experiment)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{make_run_id(**parts)}.json"
    payload = {
        "run_id": path.stem,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **parts,
        **record,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def save_table(table, experiment, **parts):
    """Write only the comparison table, with no extra keys."""
    directory = metrics_dir(experiment)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{make_run_id(**parts)}.json"
    path.write_text(json.dumps(table, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def sentiment_metrics(labels, predictions):
    report = classification_report(labels, predictions, output_dict=True, zero_division=0)

    weighted = {
        "accuracy": round(float(accuracy_score(labels, predictions)), 6),
        "precision": round(
            float(precision_score(labels, predictions, average="weighted", zero_division=0)), 6
        ),
        "recall": round(
            float(recall_score(labels, predictions, average="weighted", zero_division=0)), 6
        ),
        "f1": round(
            float(f1_score(labels, predictions, average="weighted", zero_division=0)), 6
        ),
    }
    macro = {
        "precision": round(
            float(precision_score(labels, predictions, average="macro", zero_division=0)), 6
        ),
        "recall": round(
            float(recall_score(labels, predictions, average="macro", zero_division=0)), 6
        ),
        "f1": round(
            float(f1_score(labels, predictions, average="macro", zero_division=0)), 6
        ),
    }
    per_class = {}
    for label_value, label_name in LABEL_MAPPING.items():
        stats = report.get(str(label_value), {})
        per_class[label_name] = {
            "precision": round(float(stats.get("precision", 0.0)), 6),
            "recall": round(float(stats.get("recall", 0.0)), 6),
            "f1": round(float(stats.get("f1-score", 0.0)), 6),
            "support": int(stats.get("support", 0)),
        }

    predicted_counts = {}
    for label_value, label_name in LABEL_MAPPING.items():
        predicted_counts[label_name] = int(sum(1 for value in predictions if value == label_value))
    unknown = int(sum(1 for value in predictions if value not in LABEL_MAPPING))

    return {
        "weighted": weighted,
        "macro": macro,
        "per_class": per_class,
        "predicted_counts": predicted_counts,
        "unparsed_or_unknown": unknown,
    }


def build_comparison_table(model_entries, eval_sets=TABLE_EVAL_SETS):
    """One entry per model from the reference table: raw_train, filtered_train, published."""
    eval_sets = tuple(eval_sets)
    table = {}
    for model_name, published in PUBLISHED_BASELINES.items():
        entry = model_entries.get(model_name)
        row = {
            "raw_train": None,
            "filtered_train": None,
            "published": {name: published.get(name) for name in eval_sets},
        }
        if entry:
            row["raw_train"] = {
                name: entry["conditions"][f"raw_train__{name}"]["weighted"]
                for name in eval_sets
            }
            row["filtered_train"] = {
                name: entry["conditions"][f"filtered_train__{name}"]["weighted"]
                for name in eval_sets
            }
        table[model_name] = row
    return table


def format_comparison_table(table, eval_sets=TABLE_EVAL_SETS, include_published=True):
    header = f"{'Model':<22}"
    for name in eval_sets:
        header += f"{name:^36}"
    lines = [header, "-" * len(header)]

    for model_name, row in table.items():
        lines.append(_format_row(model_name, row.get("raw_train"), eval_sets))
        lines.append(_format_row("  filtered", row.get("filtered_train"), eval_sets))
        if include_published:
            lines.append(_format_row("  published", row.get("published"), eval_sets))
        lines.append("")

    return "\n".join(lines)


def _format_row(label, blocks, eval_sets):
    line = f"{label:<22}"
    for name in eval_sets:
        block = (blocks or {}).get(name)
        if block:
            line += "  ".join(f"{block[column]:>8.3f}" for column in TABLE_COLUMNS) + "  "
        else:
            line += f"{'-':>8}" * len(TABLE_COLUMNS) + "  "
    return line


def save_model_metrics(record, experiment, model, prompt, scope="full", dataset=None):
    """One file per model, named experiment + model + prompt + scope."""
    directory = metrics_dir(experiment, dataset)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{experiment}__{_slug(model)}__{_slug(prompt)}__{_slug(scope)}.json"
    path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path
