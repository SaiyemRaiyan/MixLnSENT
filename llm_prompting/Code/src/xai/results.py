"""Hypothesis-level analysis that consumes complete, hash-verified run files only."""

from collections import Counter, defaultdict
from pathlib import Path
from .config import FIGURES_DIR, RESULTS_DIR
from .report import complete_runs, load_complete_run
from .sampling import load_sample
from .stats import mcnemar_exact, paired_stratified_bootstrap


def _consensus(records: list[dict]) -> tuple[dict[int, int], dict[int, list[int]]]:
    grouped = defaultdict(list)
    for record in records:
        if record.get("prediction") is not None:
            grouped[record["src_index"]].append(record["prediction"])
    consensus = {}
    for source, labels in grouped.items():
        counts = Counter(labels)
        best = counts.most_common()
        if len(best) == 1 or best[0][1] > best[1][1]:
            if best[0][1] >= (2 if len(labels) > 1 else 1):
                consensus[source] = best[0][0]
    return consensus, dict(grouped)


def _accuracy_summary(rows: list[dict], predictions: dict[int, int]) -> dict:
    scored = [row for row in rows if row["index"] in predictions]
    recalls = {}
    for label in range(4):
        candidates = [row for row in scored if row["gold"] == label]
        recalls[str(label)] = {
            "n": len(candidates),
            "recall": (
                sum(predictions[row["index"]] == label for row in candidates) / len(candidates)
                if candidates else None
            ),
        }
    return {
        "n": len(scored),
        "accuracy": (
            sum(predictions[row["index"]] == row["gold"] for row in scored) / len(scored)
            if scored else None
        ),
        "per_class_recall": recalls,
        "unstable_or_missing": len(rows) - len(scored),
    }


def _noise_floor(replicates: dict[int, list[int]]) -> dict[int, float]:
    return {
        source: (
            sum(first != second for i, first in enumerate(labels) for second in labels[i + 1:])
            / (len(labels) * (len(labels) - 1) / 2)
            if len(labels) > 1 else 0.0
        )
        for source, labels in replicates.items()
    }


def _paired_effect(differences: dict[int, float], rows_by_id: dict[int, dict]) -> dict:
    sources = sorted(index for index in differences if index in rows_by_id)
    return paired_stratified_bootstrap(
        [differences[index] for index in sources],
        [rows_by_id[index]["gold"] for index in sources],
    )


def _find_run(run_map, stage: str, sample: str, provider: str, model: str, prompt: str | None = None):
    for key, run in run_map.items():
        current_stage, current_sample, current_provider, current_model, current_prompt = key
        if (current_stage, current_sample, current_provider, current_model) != (
            stage, sample, provider, model,
        ):
            continue
        if prompt is None or current_prompt == prompt:
            return run
    return None


def _source_sample(path: Path) -> str:
    pieces = path.stem.split("__")
    if len(pieces) < 2:
        raise ValueError(f"unexpected XAI run filename: {path.name}")
    return pieces[1]


def analyze_complete_runs() -> dict:
    runs = complete_runs()
    run_map = {}
    for run in runs:
        path = run["path"]
        prompt = run["prompt"] or ""
        key = (
            run["stage"], _source_sample(path), run["provider"], run["model"], prompt,
        )
        run_map[key] = {
            **run,
            "records": load_complete_run(path),
        }

    e3_results = {}
    faithfulness_results = {}
    mixed_transitions = {}
    hypotheses = {f"P{number}": {} for number in range(1, 6)}
    for key, baseline_run in run_map.items():
        stage, sample_name, provider, model, _prompt = key
        if stage != "E3":
            continue
        rows = load_sample(sample_name)
        baseline, replicates = _consensus(baseline_run["records"])
        model_key = f"{provider}/{model}"
        e3_results[f"{model_key} [{sample_name}]"] = _accuracy_summary(rows, baseline)
        if sample_name not in ("xai800", "gemini300"):
            continue
        noise = _noise_floor(replicates)
        rows_by_id = {row["index"]: row for row in rows}
        e5_sample = "xai200" if sample_name == "xai800" else sample_name
        e6 = _find_run(run_map, "E6", sample_name, provider, model)
        e7 = _find_run(run_map, "E7", e5_sample, provider, model)
        e8 = _find_run(run_map, "E8", sample_name, provider, model)

        if sample_name == "xai800" and e7 is not None:
            records = e7["records"]
            outcomes = defaultdict(dict)
            interventions = defaultdict(list)
            xai200_rows = {row["index"]: row for row in load_sample("xai200")}
            for record in records:
                outcomes[record["src_index"]][record["kind"]] = record.get("prediction")
                source = record["src_index"]
                if source not in baseline or source not in xai200_rows:
                    continue
                prediction = record.get("prediction")
                if prediction is None:
                    continue
                if record["kind"].endswith("_comprehensiveness"):
                    value = float(prediction != baseline[source])
                    metric = "comprehensiveness"
                elif record["kind"].endswith("_sufficiency"):
                    value = float(prediction == baseline[source])
                    metric = "sufficiency"
                else:
                    continue
                interventions[record["kind"]].append({
                    "value": value,
                    "gold": xai200_rows[source]["gold"],
                })
            faithfulness_results[model_key] = {
                kind: {
                    "n": len(values),
                    "mean": sum(item["value"] for item in values) / len(values),
                    "ci": paired_stratified_bootstrap(
                        [item["value"] for item in values],
                        [item["gold"] for item in values],
                    ),
                    "per_class": {
                        str(label): {
                            "n": sum(item["gold"] == label for item in values),
                            "mean": (
                                sum(item["value"] for item in values if item["gold"] == label)
                                / sum(item["gold"] == label for item in values)
                                if any(item["gold"] == label for item in values) else None
                            ),
                        }
                        for label in sorted({item["gold"] for item in values})
                    },
                }
                for kind, values in interventions.items()
            }
            p1_differences = {}
            p1_labels = {}
            p2_differences = {}
            p2_labels = {}
            from .rationale import match_rationale_words

            rationale_sizes = {}
            if e6 is not None:
                for record in e6["records"]:
                    rationale = record.get("rationale", "NONE")
                    if rationale and rationale.upper() != "NONE":
                        rationale_sizes[record["src_index"]] = len(
                            match_rationale_words(record["text"], rationale)["matched"]
                        )
            row_by_id = {row["index"]: row for row in load_sample("xai200")}
            for source, values in outcomes.items():
                if source not in baseline or source not in row_by_id:
                    continue
                coalition = values.get("coalition_top_3_comprehensiveness")
                random_three = [
                    values.get(f"random_3_{draw}_comprehensiveness")
                    for draw in range(5)
                ]
                if coalition is not None and all(value is not None for value in random_three):
                    coalition_changed = coalition != baseline[source]
                    random_changed = [value != baseline[source] for value in random_three]
                    p1_differences[source] = (
                        float(coalition_changed)
                        - sum(random_changed) / len(random_changed)
                    )
                    p1_labels[source] = row_by_id[source]["gold"]
                rationale_prediction = values.get("rationale_comprehensiveness")
                size = rationale_sizes.get(source)
                if rationale_prediction is not None and size in (1, 2, 3):
                    random_values = [
                        values.get(f"random_{size}_{draw}_comprehensiveness")
                        for draw in range(5)
                    ]
                    if all(value is not None for value in random_values):
                        p2_differences[source] = (
                            float(rationale_prediction != baseline[source])
                            - sum(value != baseline[source] for value in random_values) / 5
                        )
                        p2_labels[source] = row_by_id[source]["gold"]
            for name, differences, labels in (
                ("P1", p1_differences, p1_labels),
                ("P2", p2_differences, p2_labels),
            ):
                sources = sorted(differences)
                interval = paired_stratified_bootstrap(
                    [differences[index] for index in sources],
                    [labels[index] for index in sources],
                )
                hypotheses[name][model_key] = {
                    **interval,
                    "outcome": (
                        "supported" if interval["ci_low"] is not None and interval["ci_low"] > 0
                        else "not supported" if interval["ci_high"] is not None and interval["ci_high"] <= 0
                        else "inconclusive"
                    ),
                }

        if e8 is not None and sample_name == "xai800":
            e8_by_source = defaultdict(list)
            transition_counts = defaultdict(Counter)
            for record in e8["records"]:
                e8_by_source[record["src_index"]].append(record)
                source = record["src_index"]
                if source in baseline and record.get("prediction") is not None:
                    transition_counts[record["kind"]][
                        (baseline[source], record["prediction"])
                    ] += 1
            mixed_transitions[model_key] = {
                kind: {
                    f"{before}->{after}": count
                    for (before, after), count in counts.items()
                }
                for kind, counts in transition_counts.items()
                if kind in ("clause_delete_left", "clause_delete_right")
            }
            class3 = {}
            polarity = {}
            for source, records in e8_by_source.items():
                if source not in baseline or source not in rows_by_id:
                    continue
                row = rows_by_id[source]
                for record in records:
                    kind = record["kind"]
                    prediction = record.get("prediction")
                    if kind in ("clause_delete_left", "clause_delete_right") and baseline[source] == 3:
                        if prediction is not None:
                            class3.setdefault(kind, {})[source] = float(prediction != 3) - noise.get(source, 0.0)
                    if kind == "antonym_swap" and row["gold"] in (0, 1) and prediction is not None:
                        polarity[source] = float(prediction == 1 - row["gold"]) - noise.get(source, 0.0)
            for kind, differences in class3.items():
                interval = _paired_effect(differences, rows_by_id)
                hypotheses["P3"][f"{model_key} ({kind})"] = {
                    **interval,
                    "outcome": (
                        "supported" if interval["ci_low"] is not None and interval["ci_low"] >= 0.10
                        else "not supported" if interval["ci_high"] is not None and interval["ci_high"] < 0.10
                        else "inconclusive"
                    ),
                }
            if polarity:
                interval = _paired_effect(polarity, rows_by_id)
                hypotheses["P4"][model_key] = {
                    **interval,
                    "outcome": (
                        "supported" if interval["ci_low"] is not None and interval["ci_low"] >= 0.10
                        else "not supported" if interval["ci_high"] is not None and interval["ci_high"] < 0.10
                        else "inconclusive"
                    ),
                }

    p5_by_model = defaultdict(list)
    for key, primary in run_map.items():
        stage, sample_name, provider, model, prompt = key
        if stage != "E9" or sample_name != "xai800" or prompt != "sentiment_zero_shot_v1":
            continue
        base_records = {record["src_index"]: record for record in primary["records"]}
        rows = {row["index"]: row for row in load_sample("xai800")}
        model_key = f"{provider}/{model}"
        for ablation_key, ablation in run_map.items():
            if ablation_key[:4] != ("E9", "xai800", provider, model):
                continue
            name = ablation_key[4].removeprefix("sentiment_")
            if not name.startswith("zero_shot_ablate_drop_"):
                continue
            ablation_records = {record["src_index"]: record for record in ablation["records"]}
            common = sorted(base_records.keys() & ablation_records.keys() & rows.keys())
            first = [
                base_records[index]["prediction"] == rows[index]["gold"]
                for index in common
            ]
            second = [
                ablation_records[index]["prediction"] == rows[index]["gold"]
                for index in common
            ]
            test = mcnemar_exact(first, second)
            differences = [float(after) - float(before) for before, after in zip(first, second)]
            interval = paired_stratified_bootstrap(
                differences, [rows[index]["gold"] for index in common]
            )
            p5_by_model[model_key].append({
                "prompt": name,
                "n": len(common),
                "accuracy_delta": test["accuracy_delta"],
                "ci_low": interval["ci_low"],
                "ci_high": interval["ci_high"],
                "pvalue": test["pvalue"],
            })
    from .stats import holm_adjust

    for model_key, comparisons in p5_by_model.items():
        adjusted = holm_adjust([row["pvalue"] for row in comparisons])
        for row, p_adjusted in zip(comparisons, adjusted):
            row["pvalue_holm"] = p_adjusted
            row["outcome"] = "supported" if p_adjusted < 0.05 else "not supported"
        hypotheses["P5"][model_key] = comparisons

    result = {
        "complete_runs": [
            {
                "stage": run["stage"],
                "sample": _source_sample(run["path"]),
                "provider": run["provider"],
                "model": run["model"],
                "prompt": run["prompt"],
                "rows": run["rows"],
            }
            for run in runs
        ],
        "baseline_metrics": e3_results,
        "faithfulness_metrics": faithfulness_results,
        "mixed_transitions": mixed_transitions,
        "hypotheses": hypotheses,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    import json

    output = RESULTS_DIR / "analysis.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def write_figures(analysis: dict) -> list[Path]:
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    effect_rows = []
    for hypothesis in ("P1", "P2"):
        for model, result in analysis["hypotheses"][hypothesis].items():
            if result.get("estimate") is not None:
                effect_rows.append((f"{hypothesis}: {model}", result))
    if effect_rows:
        figure, axis = plt.subplots(figsize=(max(8, len(effect_rows) * 2), 5))
        names = [item[0] for item in effect_rows]
        means = [item[1]["estimate"] for item in effect_rows]
        lows = [mean - item[1]["ci_low"] for mean, item in zip(means, effect_rows)]
        highs = [item[1]["ci_high"] - mean for mean, item in zip(means, effect_rows)]
        axis.errorbar(range(len(names)), means, yerr=[lows, highs], fmt="o", capsize=4)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(range(len(names)), names, rotation=45, ha="right")
        axis.set_ylabel("Paired comprehensiveness gap (95% CI)")
        axis.set_title("Behavioral evidence faithfulness")
        figure.tight_layout()
        path = FIGURES_DIR / "faithfulness_gap_by_model.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        written.append(path)

    curves = defaultdict(lambda: defaultdict(list))
    for model, metrics in analysis["faithfulness_metrics"].items():
        for kind, value in metrics.items():
            if not kind.startswith(("coalition_top_", "occlusion_top_")):
                continue
            pieces = kind.split("_")
            metric_name = pieces[-1]
            try:
                size = int(pieces[-2])
            except ValueError:
                continue
            curves[(model, metric_name)][size] = value["mean"]
    if curves:
        figure, axis = plt.subplots(figsize=(8, 5))
        for (model, metric), values in sorted(curves.items()):
            if values:
                axis.plot(
                    sorted(values),
                    [values[size] for size in sorted(values)],
                    marker="o",
                    label=f"{model} {metric}",
                )
        axis.set_xticks((1, 2, 3))
        axis.set_xlabel("Evidence size k")
        axis.set_ylabel("Intervention rate")
        axis.set_ylim(0, 1)
        axis.legend(fontsize="small")
        axis.set_title("Comprehensiveness and sufficiency by evidence size")
        figure.tight_layout()
        path = FIGURES_DIR / "comprehensiveness_sufficiency_curves.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        written.append(path)

    prompt_rows = [
        (model, item)
        for model, comparisons in analysis["hypotheses"]["P5"].items()
        for item in comparisons
    ]
    if prompt_rows:
        figure, axis = plt.subplots(figsize=(max(8, len(prompt_rows) * 0.7), 5))
        labels = [f"{model}\n{item['prompt']}" for model, item in prompt_rows]
        deltas = [item["accuracy_delta"] for _, item in prompt_rows]
        axis.bar(range(len(labels)), deltas)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=7)
        axis.set_ylabel("Accuracy change vs elaborate prompt")
        axis.set_title("Prompt-ablation correctness changes")
        figure.tight_layout()
        path = FIGURES_DIR / "prompt_ablation_deltas.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        written.append(path)

    transitions = analysis["mixed_transitions"]
    if transitions:
        labels = ("Positive", "Negative", "Neutral", "Mixed")
        figure, axes = plt.subplots(
            len(transitions), 2,
            figsize=(10, max(4, len(transitions) * 3)),
            squeeze=False,
        )
        for row_index, (model, operators) in enumerate(sorted(transitions.items())):
            for column_index, operator in enumerate(
                ("clause_delete_left", "clause_delete_right")
            ):
                counts = operators.get(operator, {})
                matrix = [[0] * 4 for _ in range(4)]
                for transition, count in counts.items():
                    before, after = (int(part) for part in transition.split("->"))
                    matrix[before][after] = count
                axis = axes[row_index][column_index]
                image = axis.imshow(matrix, cmap="Blues")
                axis.set_xticks(range(4), labels, rotation=30, ha="right")
                axis.set_yticks(range(4), labels)
                axis.set_xlabel("Variant prediction")
                axis.set_ylabel("Baseline")
                axis.set_title(f"{model}: {operator}")
                figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        figure.tight_layout()
        path = FIGURES_DIR / "mixed_transition_matrices.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        written.append(path)
    return written
