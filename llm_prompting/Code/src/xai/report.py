"""Completeness-checked report index; partial checkpoints never become results."""

import hashlib
import json
from pathlib import Path

from .config import FULL_MODELS, LABELS_DIR, RESULTS_DIR, XAI_OUTPUTS_DIR, ensure_output_dirs


def load_complete_run(jsonl_path: Path) -> list[dict]:
    manifest_path = jsonl_path.with_suffix(jsonl_path.suffix + ".manifest.json")
    if not manifest_path.is_file():
        raise ValueError(f"run has no completeness manifest: {jsonl_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("complete"):
        raise ValueError(f"partial run cannot be analyzed: {jsonl_path}")
    with jsonl_path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if len(rows) != manifest.get("expected_rows"):
        raise ValueError(
            f"run row count mismatch for {jsonl_path}: "
            f"{len(rows)} != {manifest.get('expected_rows')}"
        )
    ids = sorted(record["variant_id"] for record in rows)
    actual = hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()
    if actual != manifest.get("variant_id_sha256"):
        raise ValueError(f"run variant identifiers do not match manifest: {jsonl_path}")
    if any(record.get("prediction") is None for record in rows):
        raise ValueError(f"run contains unparsed labels: {jsonl_path}")
    return rows


def complete_runs() -> list[dict]:
    runs = []
    for manifest_path in sorted(LABELS_DIR.rglob("*.jsonl.manifest.json")):
        jsonl_path = Path(str(manifest_path)[:-len(".manifest.json")])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not manifest.get("complete"):
            continue
        records = load_complete_run(jsonl_path)
        runs.append({
            "path": jsonl_path,
            "rows": len(records),
            "provider": manifest.get("provider"),
            "model": manifest.get("model"),
            "prompt": manifest.get("prompt"),
            "scope": manifest.get("scope"),
            "stage": jsonl_path.parent.name,
        })
    return runs


def write_report() -> Path:
    ensure_output_dirs()
    from .results import analyze_complete_runs

    analysis = analyze_complete_runs()
    runs = complete_runs()
    if not runs:
        completed_table = "| Stage | Model | Scope | Prompt | Complete rows |\n|---|---|---|---|---:|\n| — | — | — | — | 0 |\n"
    else:
        completed_table = (
            "| Stage | Model | Scope | Prompt | Complete rows |\n"
            "|---|---|---|---|---:|\n"
            + "\n".join(
                f"| {run['stage']} | {run['provider']}/{run['model']} | "
                f"{run['scope']} | {run['prompt']} | {run['rows']} |"
                for run in runs
            )
            + "\n"
        )
    def outcome_table(items: dict) -> str:
        rows = ["| Model/condition | n | Effect | 95% CI | Outcome |", "|---|---:|---:|---|---|"]
        for name, values in items.items():
            if isinstance(values, list):
                for value in values:
                    interval = value.get("ci_low")
                    if interval is None:
                        ci = "—"
                    else:
                        ci = f"[{interval:.3f}, {value['ci_high']:.3f}]"
                    adjustment = (
                        f"; Holm p={value['pvalue_holm']:.4g}"
                        if value.get("pvalue_holm") is not None else ""
                    )
                    rows.append(
                        f"| {value.get('prompt', name)} | {value.get('n', 0)} | "
                        f"{value.get('accuracy_delta', value.get('estimate', 0)):.3f} | "
                        f"{ci}{adjustment} | {value.get('outcome', 'inconclusive')} |"
                    )
            else:
                interval = values.get("ci_low")
                ci = "—" if interval is None else f"[{interval:.3f}, {values['ci_high']:.3f}]"
                effect = values.get("estimate")
                rows.append(
                    f"| {name} | {values.get('n', 0)} | "
                    f"{'—' if effect is None else f'{effect:.3f}'} | {ci} | "
                    f"{values.get('outcome', 'inconclusive')} |"
                )
        if len(rows) == 2:
            rows.append("| — | 0 | — | — | Inconclusive (not run) |")
        return "\n".join(rows)

    baseline_rows = [
        f"| {name} | {value['n']} | "
        f"{'—' if value['accuracy'] is None else f'{value['accuracy']:.3f}'} | "
        f"{', '.join(f'{label}:{recall['recall']:.3f}' for label, recall in value['per_class_recall'].items() if recall['recall'] is not None)} |"
        for name, value in analysis["baseline_metrics"].items()
    ] or ["| — | 0 | — | — |"]

    calibration_rows = []
    unmeasured_models = []
    for provider, model in FULL_MODELS:
        safe_model = model.replace("/", "_").replace(":", "_")
        result_path = (
            LABELS_DIR / "E0" / f"dev__{provider}__{safe_model}.json"
        )
        if not result_path.is_file():
            unmeasured_models.append(f"{provider}/{model}")
            continue
        g1 = json.loads(result_path.read_text(encoding="utf-8")).get("G1", {})
        repeat = g1.get("agreement_A_A2", {}).get("agreement")
        shuffled = g1.get("agreement_A_B", {}).get("agreement")
        single = g1.get("agreement_A_C", {}).get("agreement")
        epsilon = g1.get("epsilon")
        calibration_rows.append(
            f"| {provider}/{model} | "
            f"{'—' if repeat is None else f'{repeat:.3f}'} | "
            f"{'—' if epsilon is None else f'{epsilon:.3f}'} | "
            f"{'—' if shuffled is None else f'{shuffled:.3f}'} | "
            f"{'—' if single is None else f'{single:.3f}'} |"
        )
    calibration_table = "\n".join(calibration_rows) or "| — | — | — | — | — |"
    lr_gate_path = RESULTS_DIR / "E1" / "lr_validation.json"
    lr_gate = (
        json.loads(lr_gate_path.read_text(encoding="utf-8"))
        if lr_gate_path.is_file() else None
    )
    lr_gate_summary = (
        f"LR-oracle E1: n={lr_gate['n']}, identifiable="
        f"{lr_gate.get('identifiable_n', lr_gate['n'])}, "
        f"unidentifiable={lr_gate.get('unidentifiable_n', 0)}, median Spearman="
        f"{lr_gate['median_spearman']:.3f} (gate ≥0.600), top-1 agreement="
        f"{lr_gate['top1_agreement']:.3f} (gate ≥0.700)."
        if lr_gate else "LR-oracle E1 has not been measured."
    )
    missing_model_summary = (
        ", ".join(unmeasured_models) if unmeasured_models else "none"
    )

    faithfulness_rows = []
    for model, metrics in analysis["faithfulness_metrics"].items():
        for kind, value in metrics.items():
            faithfulness_rows.append(
                f"| {model} | {kind} | {value['n']} | {value['mean']:.3f} | "
                f"[{value['ci']['ci_low']:.3f}, {value['ci']['ci_high']:.3f}] |"
            )
    faithfulness_table = "\n".join(faithfulness_rows) or "| — | — | 0 | — | — |"
    mixed_rows = []
    for model, operators in analysis["mixed_transitions"].items():
        for operator, transitions in operators.items():
            mixed_rows.append(
                f"| {model} | {operator} | "
                + ", ".join(f"{transition}: {count}" for transition, count in sorted(transitions.items()))
                + " |"
            )
    mixed_table = "\n".join(mixed_rows) or "| — | — | — |"

    prereg_path = XAI_OUTPUTS_DIR / "PREREG.md"
    report_status = (
        "PREREGISTERED / RUNS INCOMPLETE"
        if prereg_path.exists() else "DEVELOPMENT / NOT FROZEN"
    )
    report = f"""# Behavioral explainability of BnSentMix sentiment classification

**Report status: {report_status}** — regenerated from complete run manifests only. A run is included only when its manifest says complete and its JSONL row count and variant-ID digest both match.

## Setup and instrument checks (E0, E1)

Full G1/G2 preregistration status: {json.dumps((json.loads((XAI_OUTPUTS_DIR / "dev_gate.json").read_text(encoding="utf-8")) if (XAI_OUTPUTS_DIR / "dev_gate.json").exists() else {"status": "incomplete; no dev_gate.json"}), ensure_ascii=False)}.

Partial G1 development instrument results (agreement; ε = 1 − repeat agreement):

| Model | A–A2 repeat | Noise ε | A–B shuffled | A–C single |
|---|---:|---:|---:|---:|
{calibration_table}

Required but unmeasured model(s): {missing_model_summary}.

{lr_gate_summary}

E1 correlations and top-1 agreement are evaluated only where hard-label occlusion produces a non-constant token ranking; constant rows are unidentifiable rather than assigned a fabricated zero correlation or first-token tie-break. E1 thresholds require median Spearman ≥0.600 and top-1 agreement ≥0.700. Because the local LR-oracle top-1 threshold failed, coalition/KernelSHAP claims are not validated. No planted-cue gate, rationale freeze, preregistration, or test-scope inference is recorded.

| Model / sample | n | Accuracy | Per-class recall (label ids: 0 Positive, 1 Negative, 2 Neutral, 3 Mixed) |
|---|---:|---:|---|
{chr(10).join(baseline_rows)}

## Registered hypotheses

### P1 Attribution validity

{outcome_table(analysis["hypotheses"]["P1"])}

### P2 Self-rationale faithfulness

{outcome_table(analysis["hypotheses"]["P2"])}

### P3 Mixed evidence grounding

{outcome_table(analysis["hypotheses"]["P3"])}

### P4 Polarity sensitivity

{outcome_table(analysis["hypotheses"]["P4"])}

### P5 Prompt components (McNemar, Holm-adjusted within model)

{outcome_table(analysis["hypotheses"]["P5"])}

## Per-model faithfulness with confidence intervals

| Model | Evidence intervention | n | Rate | 95% CI |
|---|---|---:|---:|---|
{faithfulness_table}

## Mixed analysis (E8 C1–C3)

Clause deletion transition counts (baseline label to variant label):

| Model | Operator | Transitions |
|---|---|---|
{mixed_table}

## Prompt-component results (E9)

P5 outcome table above is derived from common source rows in complete primary and ablation runs. Each recorded comparison includes the per-sentence correctness delta and Holm-adjusted p-value.

## Exploratory results

Language/position, demonstration, and cross-method analyses are exploratory. Language-tag findings are not validated unless the 300-token hand-label gate G4 reaches 0.90 accuracy.

## Complete runs

{completed_table}
## Limitations

Hosted models expose hard labels in this pipeline. Results are behavioural intervention evidence, not access to internal representations or proof of causal mechanisms. The optional white-box arm is not claimed unless its GPU dependencies and local/API agreement gate are satisfied.

## Deviations

See [DEVIATIONS.md](./DEVIATIONS.md). No hypothesis outcome or metric is filled from incomplete checkpoints.
"""
    path = XAI_OUTPUTS_DIR / "REPORT.md"
    path.write_text(report, encoding="utf-8")
    from .results import write_figures

    write_figures(analysis)
    return path
