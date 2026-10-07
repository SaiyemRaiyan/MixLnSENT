"""Fail-closed preregistration and test-scope integrity checks."""

import hashlib
import json
import math
import re
import subprocess
import csv
from pathlib import Path

from .config import (
    FULL_MODELS, PROJECT_ROOT, PROMPT_HASH_FILES, XAI_OUTPUTS_DIR,
    ensure_output_dirs,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prereg_prompt_files() -> list[Path]:
    root = PROJECT_ROOT / "prompts"
    used = set(PROMPT_HASH_FILES)
    used.update((root / "prompting").glob("zero_shot_ablate_drop_*_v1.txt"))
    used.add(root / "prompting" / "zero_shot_v1.txt")
    used.add(root / "prompting" / "five_shot_minimal_v1.txt")
    used.update((root / "xai").glob("five_shot_minimal_leave_out_*_v1.txt"))
    return sorted(path for path in used if path.exists())


def _validate_gate(gate: dict) -> None:
    g1 = gate.get("G1")
    if not isinstance(g1, dict):
        raise ValueError("dev_gate.json needs measured G1 results")
    required_models = {
        "groq/qwen/qwen3.8-27b",
        "groq/openai/gpt-oss-20b",
        "huggingface/meta-llama/Llama-3.1-8B-Instruct",
    }
    missing_models = required_models - set(g1)
    if missing_models:
        raise ValueError(f"G1 results missing models: {sorted(missing_models)}")
    for model, result in g1.items():
        if not isinstance(result, dict):
            raise ValueError(f"G1 entry for {model} must be an object")
        for key in ("agreement_A_A2", "agreement_A_B", "agreement_A_C"):
            value = result.get(key)
            if not isinstance(value, dict) or not isinstance(value.get("agreement"), (int, float)):
                raise ValueError(f"G1 {model}.{key}.agreement must be measured")
    g2 = gate.get("G2")
    if not isinstance(g2, dict):
        raise ValueError("dev_gate.json needs measured G2 results")
    for key in (
        "lr_median_spearman", "lr_top1_agreement", "planted_cue_top1_rate",
        "coalition_budget",
    ):
        if not isinstance(g2.get(key), (int, float)):
            raise ValueError(f"G2 {key} must be measured")
    if g2["coalition_budget"] not in (32, 64, 96, 192):
        raise ValueError("G2 coalition_budget must be one of 32, 64, 96, 192")


def _rationale_validation_path(provider: str, model: str) -> Path:
    safe_model = model.replace("/", "_").replace(":", "_")
    return (
        XAI_OUTPUTS_DIR / "results" / "E6"
        / f"dev_rationale_validation__{provider}__{safe_model}.json"
    )


def _validate_rationale_prompt_freeze(validation_by_model: dict) -> None:
    for provider, model in FULL_MODELS:
        key = f"{provider}/{model}"
        result = validation_by_model.get(key)
        if not isinstance(result, dict):
            raise ValueError(f"development rationale validation is missing for {key}")
        for metric, threshold in (
            ("parse_rate", 0.95),
            ("verbatim_match_rate", 0.90),
        ):
            value = result.get(metric)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not threshold <= value <= 1
            ):
                raise ValueError(
                    f"development rationale {metric} must be between "
                    f"{threshold} and 1 for {key}"
                )


def build_prereg_payload() -> dict:
    sample_manifest = XAI_OUTPUTS_DIR / "samples" / "manifest.json"
    if not sample_manifest.exists():
        raise FileNotFoundError("prepare samples before preregistering")
    manifest = json.loads(sample_manifest.read_text(encoding="utf-8"))
    hashed_files = {}
    for entry in manifest.values():
        if isinstance(entry, dict) and entry.get("file"):
            path = (XAI_OUTPUTS_DIR / entry["file"]).resolve()
            hashed_files[str(path.relative_to(PROJECT_ROOT))] = sha256(path)
    hashed_files[str(sample_manifest.relative_to(PROJECT_ROOT))] = sha256(sample_manifest)
    for path in _prereg_prompt_files():
        hashed_files[str(path.relative_to(PROJECT_ROOT))] = sha256(path)
    for provider, model in FULL_MODELS:
        rationale_validation = _rationale_validation_path(provider, model)
        if rationale_validation.is_file():
            hashed_files[str(rationale_validation.relative_to(PROJECT_ROOT))] = (
                sha256(rationale_validation)
            )
    signoff = XAI_OUTPUTS_DIR / "annotation" / "operator_signoff.csv"
    if signoff.is_file():
        hashed_files[str(signoff.relative_to(PROJECT_ROOT))] = sha256(signoff)
    for path in sorted((PROJECT_ROOT / "src" / "xai").glob("*.py")):
        hashed_files[str(path.relative_to(PROJECT_ROOT))] = sha256(path)
    plan_path = PROJECT_ROOT / (
        "XAI Experiment Plan Behavioural Explainability of LLM Sentiment Classification on Code-Mixed Bengali-English Text.md"
    )
    if plan_path.is_file():
        hashed_files[str(plan_path.relative_to(PROJECT_ROOT))] = sha256(plan_path)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("could not record the current git commit") from error
    return {
        "version": 1,
        "commit": commit,
        "seed": manifest["seed"],
        "sample_manifest": manifest,
        "models": {
            "full_analysis": [
                {"provider": provider, "model": model}
                for provider, model in (
                    ("groq", "qwen/qwen3.8-27b"),
                    ("groq", "openai/gpt-oss-20b"),
                    ("huggingface", "meta-llama/Llama-3.1-8B-Instruct"),
                )
            ],
            "restricted": {
                "provider": "commandcode",
                "model": "google/gemini-3.8-flash",
            },
        },
        "primary_prompt": "prompts/prompting/zero_shot_minimal_v1.txt",
        "hashed_files": hashed_files,
        "hypotheses": {
            "P1": "Coalition top-3 comprehensiveness exceeds 3-random-word comprehensiveness; paired bootstrap CI excludes 0.",
            "P2": "Self-rationale evidence comprehensiveness exceeds size-matched random evidence; paired bootstrap CI excludes 0.",
            "P3": "For baseline-Mixed rows with a strict marker, deleting either clause moves away from Mixed at least 0.10 above the noise floor.",
            "P4": "On single-polarity rows with one lexicon hit, antonym swapping flips to the opposite polarity at least 0.10 above noise.",
            "P5": "Each of eight prompt ablations changes correctness; paired McNemar tests, Holm-corrected within family.",
        },
    }


def write_prereg() -> Path:
    ensure_output_dirs()
    gate_path = XAI_OUTPUTS_DIR / "dev_gate.json"
    if not gate_path.exists():
        raise FileNotFoundError(
            "dev_gate.json must contain measured G1 and G2 results before test freeze"
        )
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    _validate_gate(gate)
    signoff_path = XAI_OUTPUTS_DIR / "annotation" / "operator_signoff.csv"
    if not signoff_path.is_file():
        raise FileNotFoundError(
            "prepare annotation sheets and obtain operator sign-off before freezing"
        )
    with signoff_path.open(encoding="utf-8", newline="") as stream:
        signoff = list(csv.DictReader(stream))
    if not signoff or any(
        row.get("approved", "").strip().lower() not in ("yes", "approved", "true")
        for row in signoff
    ):
        raise ValueError("every operator-signoff row must be approved before freezing")
    rationale_validation = {}
    for provider, model in FULL_MODELS:
        validation_path = _rationale_validation_path(provider, model)
        if not validation_path.is_file():
            raise FileNotFoundError(
                f"complete the development rationale-prompt freeze check for {provider}/{model}"
            )
        rationale_validation[f"{provider}/{model}"] = json.loads(
            validation_path.read_text(encoding="utf-8")
        )
    _validate_rationale_prompt_freeze(rationale_validation)
    payload = build_prereg_payload()
    payload["dev_gate"] = gate
    payload["rationale_prompt_validation"] = rationale_validation
    path = XAI_OUTPUTS_DIR / "PREREG.md"
    path.write_text(
        "# Frozen XAI preregistration\n\n"
        "This file is generated from the frozen sample manifest, config, prompts, and git state.\n\n"
        "```json\n" + json.dumps(payload, ensure_ascii=False, indent=2) + "\n```\n",
        encoding="utf-8",
    )
    return path


def check_prereg() -> dict:
    path = XAI_OUTPUTS_DIR / "PREREG.md"
    if not path.exists():
        raise RuntimeError("test scope refused: outputs/xai/PREREG.md is missing")
    text = path.read_text(encoding="utf-8")
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if not match:
        raise RuntimeError("test scope refused: PREREG.md has no valid JSON payload")
    payload = json.loads(match.group(1))
    gate = payload.get("dev_gate", {})
    try:
        _validate_gate(gate)
    except ValueError as error:
        raise RuntimeError(f"test scope refused: {error}") from error
    try:
        _validate_rationale_prompt_freeze(
            payload.get("rationale_prompt_validation", {})
        )
    except ValueError as error:
        raise RuntimeError(f"test scope refused: {error}") from error
    for relative, expected in payload.get("hashed_files", {}).items():
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"test scope refused: preregistered file is missing: {relative}")
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"test scope refused: preregistered hash changed: {relative}")
    return payload
