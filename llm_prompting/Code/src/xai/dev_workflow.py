"""Development-only operations required to complete G1 and G2."""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from .analysis import coalition_seed_stability, make_coalition_attributions
from .config import (
    COALITION_BUDGETS, LABELS_DIR, PROJECT_ROOT, RESULTS_DIR, XAI_OUTPUTS_DIR,
)
from .instrument import agreement
from .sampling import load_sample
from .validity import (
    compare_occlusion_to_lr, fit_lr_oracle, planted_cue_rows,
    planted_cue_top1_rate, select_lr_validation_rows,
)
from .variants import (
    coalition_variants, make_variant, occlusion_variants,
)
from .oracle import run_variants
from .report import load_complete_run


def model_run_path(stage: str, provider: str, model: str, sample: str = "dev") -> Path:
    safe_model = model.replace("/", "_").replace(":", "_")
    return LABELS_DIR / stage / f"dev__{sample}__{provider}__{safe_model}.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def baseline_consensus(records: list[dict]) -> tuple[dict[int, int], list[int]]:
    grouped = defaultdict(list)
    for record in records:
        if record.get("prediction") is not None:
            grouped[record["src_index"]].append(record["prediction"])
    result = {}
    unstable = []
    for source, predictions in grouped.items():
        counts = Counter(predictions)
        most_common = counts.most_common()
        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            unstable.append(source)
        elif most_common[0][1] >= 2:
            result[source] = most_common[0][0]
        else:
            unstable.append(source)
    return result, unstable


def run_lr_gate() -> dict:
    rows = select_lr_validation_rows(load_sample("dev"), 300)
    result = compare_occlusion_to_lr(fit_lr_oracle(), rows)
    path = RESULTS_DIR / "E1" / "lr_validation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def run_planted_cue_gate(provider: str, model: str, coalitions: int, sender=None) -> dict:
    if (provider, model) != ("groq", "qwen/qwen3.8-27b"):
        raise ValueError("the planted-cue gate is preregistered on groq/qwen/qwen3.8-27b")
    from dotenv import load_dotenv

    from .analysis import _write_jsonl
    from .variants import baseline_variants

    load_dotenv(override=True)
    rows = load_sample("dev")
    baseline_records = read_jsonl(model_run_path("E3", provider, model))
    baseline, unstable = baseline_consensus(baseline_records)
    planted = planted_cue_rows(rows)
    cue_variants = [
        make_variant(
            row,
            f"planted_{row['planted_cue']}",
            [],
            row["sentence"],
        )
        for row in planted
    ]
    cue_path = model_run_path("E1_planted", provider, model)
    cue_records = run_variants(
        cue_variants, provider, model, cue_path, "dev", sender=sender,
        prompt_name="zero_shot_minimal_v1",
    )
    cue_predictions = {
        record["src_index"]: record["prediction"] for record in cue_records
    }
    changed = [
        row for row in planted
        if row["index"] in cue_predictions
        and cue_predictions[row["index"]] != baseline.get(row["index"])
    ]
    if not changed:
        raise ValueError("no planted-cue rows changed the model label; G2 cannot be measured")
    coalition_records = []
    for replicate in (1, 2):
        variants = [
            variant
            for position, row in enumerate(changed)
            for variant in coalition_variants(
                row,
                coalitions,
                seed=71_000 + replicate * 10_000 + position,
                rep=replicate,
            )
        ]
        output_path = model_run_path(
            f"E1_planted_coalitions_M{coalitions}_rep{replicate}",
            provider,
            model,
        )
        coalition_records.extend(run_variants(
            variants, provider, model, output_path, "dev", sender=sender,
            prompt_name="zero_shot_minimal_v1",
        ))
    scores = make_coalition_attributions(changed, cue_predictions, coalition_records)
    score_by_source = {row["src_index"]: row["scores"] for row in scores}
    result = {
        "coalition_budget": coalitions,
        "planted_cue": planted_cue_top1_rate(
            planted, cue_predictions, score_by_source, baseline
        ),
        "changed_rows": len(changed),
        "unstable_baselines": len(unstable),
        "cue_predictions": len(cue_predictions),
    }
    path = RESULTS_DIR / "E1" / f"planted_cue_M{coalitions}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def choose_dev_coalition_budget(provider: str, model: str) -> dict:
    rows = load_sample("dev")
    baseline_records = load_complete_run(model_run_path("E3", provider, model))
    baseline, _ = baseline_consensus(baseline_records)
    overlap_by_budget = {}
    detail = {}
    for budget in COALITION_BUDGETS:
        run_path = model_run_path(f"E5_M{budget}", provider, model)
        records = load_complete_run(run_path)
        result = coalition_seed_stability(rows, baseline, records)
        detail[budget] = result
        overlap_by_budget[budget] = result["mean_top3_jaccard"] or 0.0
    selected = next(
        (budget for budget in COALITION_BUDGETS if overlap_by_budget[budget] >= 0.8),
        COALITION_BUDGETS[-1],
    )
    result = {"selected_budget": selected, "budgets": detail}
    path = RESULTS_DIR / "E2" / "dev_coalition_budget.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def finalize_dev_gate(provider: str = "groq", model: str = "qwen/qwen3.8-27b") -> dict:
    g1 = {}
    models = (
        ("groq", "qwen/qwen3.8-27b"),
        ("groq", "openai/gpt-oss-20b"),
        ("huggingface", "meta-llama/Llama-3.1-8B-Instruct"),
    )
    for current_provider, current_model in models:
        slug = current_model.replace("/", "_")
        path = LABELS_DIR / "E0" / f"dev__{current_provider}__{slug}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = payload["G1"]
        result["noisy"] = result["agreement_A_A2"]["agreement"] < 0.90
        g1[f"{current_provider}/{current_model}"] = result
    lr = json.loads((RESULTS_DIR / "E1" / "lr_validation.json").read_text(encoding="utf-8"))
    planted_path = RESULTS_DIR / "E1" / "planted_cue_M*.json"
    planted_files = sorted(planted_path.parent.glob(planted_path.name))
    if not planted_files:
        raise FileNotFoundError("run the Qwen planted-cue gate before finalizing G2")
    planted_results = [
        json.loads(path.read_text(encoding="utf-8")) for path in planted_files
    ]
    planted_results.sort(key=lambda item: item["coalition_budget"])
    passing_planted = [
        item for item in planted_results
        if item["planted_cue"]["top1_rate"] is not None
        and item["planted_cue"]["top1_rate"] >= 0.8
    ]
    if passing_planted:
        planted = passing_planted[0]
    else:
        planted = planted_results[-1]
    stability = json.loads(
        (RESULTS_DIR / "E2" / "dev_coalition_budget.json").read_text(encoding="utf-8")
    )
    selected = max(
        stability["selected_budget"],
        planted["coalition_budget"],
    )
    pipeline_pass = (
        lr["median_spearman"] is not None
        and lr["median_spearman"] >= 0.6
        and lr["top1_agreement"] >= 0.7
        and planted["planted_cue"]["top1_rate"] is not None
        and planted["planted_cue"]["top1_rate"] >= 0.8
    )
    gate = {
        "G1": g1,
        "G2": {
            "lr_median_spearman": lr["median_spearman"],
            "lr_top1_agreement": lr["top1_agreement"],
            "planted_cue_top1_rate": planted["planted_cue"]["top1_rate"],
            "coalition_budget": selected,
            "pipeline_limit": "full" if pipeline_pass else "occlusion_only",
            "planted_cue_budget_tested": planted["coalition_budget"],
        },
    }
    path = XAI_OUTPUTS_DIR / "dev_gate.json"
    path.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    return gate
