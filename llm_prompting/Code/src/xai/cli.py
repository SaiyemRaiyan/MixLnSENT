"""Command-line entry points for local preparation and resumable XAI runs."""

import argparse
import json
from pathlib import Path

from src.prompting.models import model_spec

from .config import (
    ATTRIBUTIONS_DIR, LABELS_DIR, RESULTS_DIR, SAMPLES_DIR, ensure_output_dirs,
)
from .counterfactual import generate_counterfactuals
from .gates import check_prereg, write_prereg
from .oracle import run_variants
from .rationale import run_rationales
from .sampling import load_sample, prepare_dev, prepare_test
from .variants import baseline_variants, coalition_variants, occlusion_variants


def _model_path(
    provider: str, model: str, stage: str, scope: str, sample: str
) -> Path:
    safe_model = model.replace("/", "_").replace(":", "_")
    return LABELS_DIR / stage / f"{scope}__{sample}__{provider}__{safe_model}.jsonl"


def _rationale_path(provider: str, model: str, scope: str, sample: str) -> Path:
    import hashlib

    from .config import XAI_PROMPTS_DIR

    prompt_path = XAI_PROMPTS_DIR / "rationale_extractive_v1.txt"
    prompt_hash = hashlib.sha256(prompt_path.read_bytes()).hexdigest()[:8]
    base_path = _model_path(provider, model, "E6", scope, sample)
    return base_path.with_name(f"{base_path.stem}__p-{prompt_hash}.jsonl")


def _read_jsonl(path: Path) -> list[dict]:
    from .report import load_complete_run

    return load_complete_run(path)


def _baseline_consensus(path: Path, allow_single: bool = False) -> tuple[dict[int, int], list[int]]:
    from collections import Counter, defaultdict

    records = _read_jsonl(path)
    predictions = defaultdict(list)
    for record in records:
        if record.get("prediction") is not None:
            predictions[record["src_index"]].append(record["prediction"])
    consensus = {}
    unstable = []
    for source, labels in predictions.items():
        counts = Counter(labels)
        maximum = max(counts.values())
        winners = [label for label, count in counts.items() if count == maximum]
        if len(winners) == 1 and (maximum >= 2 or (allow_single and maximum == 1)):
            consensus[source] = winners[0]
        else:
            unstable.append(source)
    return consensus, unstable


def _run(args) -> None:
    from dotenv import load_dotenv

    load_dotenv(override=True)
    ensure_output_dirs()
    if (args.scope == "dev") != (args.sample == "dev"):
        raise ValueError("dev runs must use the dev sample and test runs must use a test sample")
    prereg = None
    if args.scope == "test":
        prereg = check_prereg()
    if args.coalitions is None:
        args.coalitions = (
            prereg["dev_gate"]["G2"]["coalition_budget"]
            if prereg is not None else 96
        )
    elif prereg is not None and args.coalitions != prereg["dev_gate"]["G2"]["coalition_budget"]:
        raise ValueError("test coalition budget is frozen by PREREG.md")
    rows = load_sample(args.sample)
    spec = model_spec(args.provider, args.model)
    batch_size = args.batch_size or spec["batch_size"]
    baseline_sample = "xai800" if args.sample == "xai200" else args.sample
    allow_single = (
        args.provider == "commandcode"
        and args.model == "google/gemini-3.8-flash"
    )

    if args.stage == "baseline":
        variants = baseline_variants(rows, args.replicates)
    elif args.stage == "prompt":
        variants = baseline_variants(rows, replicates=1)
    elif args.stage == "occlusion":
        baseline, _ = _baseline_consensus(
            _model_path(args.provider, args.model, "E3", args.scope, baseline_sample),
            allow_single=allow_single,
        )
        variants = [
            variant
            for row in rows if row["index"] in baseline
            for variant in occlusion_variants(row)
        ]
    elif args.stage == "coalition":
        baseline_sample = "dev" if args.sample == "dev" else "xai800"
        baseline, _ = _baseline_consensus(
            _model_path(args.provider, args.model, "E3", args.scope, baseline_sample),
            allow_single=allow_single,
        )
        if args.sample not in ("dev", "xai200"):
            raise ValueError("coalition attribution is restricted to dev or xai200")
        variants = []
        for position, row in enumerate(rows):
            if row["index"] not in baseline:
                continue
            variants.extend(coalition_variants(
                row, args.coalitions, args.seed + position, rep=1
            ))
            variants.extend(coalition_variants(
                row, args.coalitions, args.seed + 10_000 + position, rep=2
            ))
    elif args.stage == "counterfactual":
        baseline, unstable = _baseline_consensus(
            _model_path(args.provider, args.model, "E3", args.scope, baseline_sample)
        )
        variants, qualified = generate_counterfactuals(rows, baseline)
        print(json.dumps({"unstable_source_rows": len(unstable), "qualified": qualified}, indent=2))
    elif args.stage == "rationale":
        baseline, _ = _baseline_consensus(
            _model_path(args.provider, args.model, "E3", args.scope, baseline_sample),
            allow_single=allow_single,
        )
        path = _rationale_path(args.provider, args.model, args.scope, args.sample)
        results = run_rationales(
            [row for row in rows if row["index"] in baseline],
            baseline,
            args.provider,
            args.model,
            path,
            args.scope,
            sender=None,
        )
        if args.scope == "dev":
            from .rationale import validate_rationales

            validation = validate_rationales(rows, results)
            validation.pop("records", None)
            safe_model = args.model.replace("/", "_").replace(":", "_")
            validation_path = (
                RESULTS_DIR / "E6"
                / f"dev_rationale_validation__{args.provider}__{safe_model}.json"
            )
            validation_path.parent.mkdir(parents=True, exist_ok=True)
            validation_path.write_text(
                json.dumps(validation, indent=2) + "\n", encoding="utf-8"
            )
            print(json.dumps(validation, indent=2))
        print(f"Completed {len(results)} rationale records at {path}")
        return
    elif args.stage == "langtag":
        from .run_langtag import run_langtags

        path = _model_path(args.provider, args.model, "E10", args.scope, args.sample)
        results = run_langtags(
            rows, args.provider, args.model, path, args.scope, sender=None
        )
        print(f"Completed {len(results)} token tags at {path}")
        return
    elif args.stage == "faithfulness":
        if args.sample != "xai200":
            raise ValueError("E7 is evaluated on xai200")
        from .analysis import (
            make_coalition_attributions, make_occlusion_attributions, _write_jsonl,
        )
        from .faithfulness import build_faithfulness_variants
        from .rationale import match_rationale_words

        baseline, _ = _baseline_consensus(
            _model_path(args.provider, args.model, "E3", args.scope, "xai800"),
            allow_single=allow_single,
        )
        e4_path = _model_path(args.provider, args.model, "E4", args.scope, "xai800")
        e5_path = _model_path(
            args.provider, args.model, f"E5_M{args.coalitions}", args.scope, "xai200"
        )
        e6_path = _rationale_path(args.provider, args.model, args.scope, "xai800")
        e4_records = _read_jsonl(e4_path)
        e5_records = _read_jsonl(e5_path)
        e6_records = _read_jsonl(e6_path)
        attributions_e4 = make_occlusion_attributions(rows, baseline, e4_records)
        attributions_e5 = make_coalition_attributions(rows, baseline, e5_records)
        attr_dir = ATTRIBUTIONS_DIR / "E7"
        attr_dir.mkdir(parents=True, exist_ok=True)
        safe_model = args.model.replace("/", "_")
        _write_jsonl(attr_dir / f"{args.scope}__{safe_model}__occlusion.jsonl", attributions_e4)
        _write_jsonl(attr_dir / f"{args.scope}__{safe_model}__coalition.jsonl", attributions_e5)
        rationale_indices = {}
        for record in e6_records:
            rationale = record.get("rationale", "NONE")
            if rationale and rationale.upper() != "NONE":
                match = match_rationale_words(record["text"], rationale)
                rationale_indices[record["src_index"]] = [
                    item["token_index"] for item in match["matched"]
                ]
        coalition_scores = {
            record["src_index"]: record["scores"] for record in attributions_e5
        }
        occlusion_scores = {
            record["src_index"]: record["scores"] for record in attributions_e4
        }
        variants = build_faithfulness_variants(
            [row for row in rows if row["index"] in baseline],
            coalition_scores, occlusion_scores, rationale_indices, args.seed
        )
    elif args.stage == "instrument":
        if args.scope != "dev":
            raise ValueError("E0 instrument checks are development-only")
        from .instrument import instrument_conditions

        conditions = instrument_conditions(rows)
        results = {}
        for condition in ("A", "A2", "B", "C", "D"):
            if condition == "A2":
                condition_variants = conditions["A2"]
                seed = None
                batch_size = args.batch_size or spec["batch_size"]
            else:
                condition_variants = conditions[condition]
                seed = None if condition == "A" else args.seed
                batch_size = 1 if condition == "C" else args.batch_size or spec["batch_size"]
            output = _model_path(
                args.provider, args.model, f"E0_{condition}", args.scope, args.sample
            )
            results[condition] = run_variants(
                condition_variants, args.provider, args.model, output, args.scope,
                sender=None, batch_size=batch_size, seed=seed, prompt_name=args.prompt,
            )
        from .instrument import agreement

        summary = {
            "G1": {
                "agreement_A_A2": agreement(results["A"], results["A2"]),
                "agreement_A_B": agreement(results["A"], results["B"]),
                "agreement_A_C": agreement(results["A"], results["C"]),
                "epsilon": 1 - agreement(results["A"], results["A2"])["agreement"],
            }
        }
        result_path = LABELS_DIR / "E0" / f"{args.scope}__{args.provider}__{args.model.replace('/', '_')}.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return
    else:
        raise ValueError(f"unsupported stage: {args.stage}")

    stage_name = {
        "baseline": "E3",
        "prompt": "E9",
        "occlusion": "E4",
        "coalition": f"E5_M{args.coalitions}",
        "counterfactual": "E8",
        "faithfulness": "E7",
    }[args.stage]
    output = _model_path(
        args.provider, args.model, stage_name, args.scope, args.sample
    )
    if args.stage in ("prompt",):
        prompt_slug = args.prompt.replace("/", "_")
        output = output.with_name(output.stem + f"__{prompt_slug}" + output.suffix)
    results = run_variants(
        variants,
        args.provider,
        args.model,
        output,
        args.scope,
        batch_size=batch_size,
        seed=args.seed,
        prompt_name=args.prompt,
    )
    if args.stage == "faithfulness":
        from .report import write_report

        print(f"Updated report: {write_report()}")
    print(f"Completed {len(results)} records at {output}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m src.xai.cli")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare-dev", help="create the train-only dev sample")
    commands.add_parser("prepare-test", help="create frozen test and subset samples")
    annotations = commands.add_parser("prepare-annotations", help="create human annotation sheets")
    commands.add_parser("preregister", help="freeze measured dev gates and current hashes")
    commands.add_parser("finalize-dev-gate", help="combine measured G1/G2 results")
    validate = commands.add_parser("validate-dev", help="run a development-only E1 gate")
    validate.add_argument("--mode", required=True, choices=("lr", "planted-cue"))
    validate.add_argument("--provider", default="groq")
    validate.add_argument("--model", default="qwen/qwen3.8-27b")
    validate.add_argument("--coalitions", type=int, default=96)
    select_budget = commands.add_parser(
        "select-coalition-budget", help="choose M from completed two-seed dev runs"
    )
    select_budget.add_argument("--provider", required=True)
    select_budget.add_argument("--model", required=True)
    commands.add_parser(
        "make-demo-ablation-prompts",
        help="generate one leave-one-demo-out copy per demonstration in the source prompt",
    )
    check = commands.add_parser("check-prereg", help="verify test-scope gate and hashes")
    commands.add_parser("report", help="regenerate the report from complete runs only")
    run = commands.add_parser("run", help="run one resumable behavioural-XAI stage")
    run.add_argument("--stage", required=True, choices=(
        "baseline", "occlusion", "coalition", "counterfactual", "rationale",
        "instrument", "prompt", "faithfulness", "langtag",
    ))
    run.add_argument("--sample", required=True, choices=(
        "dev", "xai800", "xai200", "gemini300", "gemini_noise100",
    ))
    run.add_argument("--scope", required=True, choices=("dev", "test"))
    run.add_argument("--provider", required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--batch-size", type=int)
    run.add_argument("--coalitions", type=int)
    run.add_argument("--seed", type=int, default=43)
    run.add_argument("--prompt", default="zero_shot_minimal_v1")
    run.add_argument("--replicates", type=int, default=3)

    args = parser.parse_args()
    if args.command == "prepare-dev":
        print(json.dumps(prepare_dev(), indent=2))
    elif args.command == "prepare-test":
        print(json.dumps(prepare_test(), indent=2))
    elif args.command == "prepare-annotations":
        from .annotations import prepare_annotation_templates

        result = prepare_annotation_templates(
            load_sample("dev"), load_sample("xai800")
        )
        print(json.dumps(result, indent=2))
    elif args.command == "preregister":
        print(write_prereg())
    elif args.command == "finalize-dev-gate":
        from .dev_workflow import finalize_dev_gate

        print(json.dumps(finalize_dev_gate(), indent=2))
    elif args.command == "validate-dev":
        from .dev_workflow import run_lr_gate, run_planted_cue_gate

        if args.mode == "lr":
            result = run_lr_gate()
        else:
            result = run_planted_cue_gate(
                args.provider, args.model, args.coalitions
            )
        print(json.dumps(result, indent=2))
    elif args.command == "select-coalition-budget":
        from .dev_workflow import choose_dev_coalition_budget

        print(json.dumps(
            choose_dev_coalition_budget(args.provider, args.model), indent=2
        ))
    elif args.command == "make-demo-ablation-prompts":
        from .prompt_variants import make_leave_one_out_prompts

        paths = make_leave_one_out_prompts()
        print(f"Created {len(paths)} leave-one-demonstration-out prompts")
        for path in paths:
            print(path)
    elif args.command == "check-prereg":
        payload = check_prereg()
        print(f"Valid preregistration at commit {payload['commit']}")
    elif args.command == "report":
        from .report import write_report

        print(write_report())
    else:
        _run(args)


if __name__ == "__main__":
    main()
