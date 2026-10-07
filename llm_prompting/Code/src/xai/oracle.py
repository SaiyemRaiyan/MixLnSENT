"""Resumable variant classifier built on the repository's shared prompting path."""

import json
import hashlib
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from src.common.io_utils import append_jsonl, load_jsonl
from src.prompting.models import model_spec
from src.prompting.prompts import (
    build_sentiment_batch_prompt, parse_sentiment_batch, prompt_label,
)
from src.prompting.providers import build_sender
from src.prompting.runner import classify_batch

from .config import LABELS_DIR, PROJECT_ROOT, VARIANT_SEED, XAI_PROMPTS_DIR, ensure_output_dirs
from .gates import check_prereg


SENTIMENT_TASK = SimpleNamespace(
    build=build_sentiment_batch_prompt,
    parse=parse_sentiment_batch,
    label=prompt_label,
    name="sentiment",
)


def _task_label(task, prompt_name: str) -> str:
    try:
        return task.label(prompt_name)
    except TypeError:
        return task.label()


def _run_metadata(task, prompt_name: str, provider: str, model: str, scope: str, batch_size: int, seed):
    if getattr(task, "name", "sentiment") == "sentiment":
        prompt_path = PROJECT_ROOT / "prompts" / "prompting" / f"{prompt_name}.txt"
        batch_path = PROJECT_ROOT / "prompts" / "prompting" / "batch_format_v1.txt"
    else:
        prompt_path = XAI_PROMPTS_DIR / f"{prompt_name}.txt"
        batch_name = "rationale" if task.name == "rationale" else prompt_name.removesuffix("_v1")
        batch_path = XAI_PROMPTS_DIR / f"{batch_name}_batch_format_v1.txt"
    if not prompt_path.is_file() or not batch_path.is_file():
        raise FileNotFoundError(
            f"prompt or batch format missing: {prompt_path}, {batch_path}"
        )
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("could not record the current git commit") from error
    return {
        "provider": provider,
        "model": model,
        "prompt": _task_label(task, prompt_name),
        "scope": scope,
        "batch_size": batch_size,
        "shuffle_seed": seed,
        "prompt_sha256": hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
        "batch_format_sha256": hashlib.sha256(batch_path.read_bytes()).hexdigest(),
        "git_commit": commit,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _batches(records: list[dict], batch_size: int, seed: int | None):
    pending = list(records)
    if seed is not None:
        random.Random(seed).shuffle(pending)
    while pending:
        batch = []
        sources = set()
        deferred = []
        for record in pending:
            source = record["src_index"]
            if source not in sources and len(batch) < batch_size:
                batch.append(record)
                sources.add(source)
            else:
                deferred.append(record)
        if not batch:
            raise RuntimeError("P_batch cannot make progress on the pending variants")
        yield batch
        pending = deferred


def run_variants(
    variants: list[dict],
    provider: str,
    model: str,
    output_path: Path,
    scope: str,
    prompt_name: str = "zero_shot_minimal_v1",
    sender=None,
    batch_size: int | None = None,
    seed: int | None = VARIANT_SEED,
    log=print,
    task=SENTIMENT_TASK,
) -> list[dict]:
    """Classify variants through classify_batch, preserving the original prompt bytes."""
    if scope == "test":
        check_prereg()
    ensure_output_dirs()
    spec = model_spec(provider, model)
    batch_size = batch_size or spec["batch_size"]
    send_prompt = sender or build_sender(provider, model)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_jsonl(path)
    done = {record["variant_id"]: record for record in existing}
    by_id = {}
    for variant in variants:
        identity = variant["variant_id"]
        if identity in by_id and by_id[identity] != variant:
            raise ValueError(f"variant id collision for {identity}")
        by_id[identity] = variant
    variant_digest = hashlib.sha256(
        "\n".join(sorted(by_id)).encode("utf-8")
    ).hexdigest()
    input_digest = hashlib.sha256(
        "\n".join(
            json.dumps(by_id[key], sort_keys=True, ensure_ascii=False)
            for key in sorted(by_id)
        ).encode("utf-8")
    ).hexdigest()
    metadata = _run_metadata(task, prompt_name, provider, model, scope, batch_size, seed)
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if manifest_path.exists():
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous_manifest.get("variant_id_sha256") != variant_digest:
            raise ValueError(f"run input changed since checkpoint: {manifest_path}")
        if previous_manifest.get("input_sha256") != input_digest:
            raise ValueError(f"variant content changed since checkpoint: {manifest_path}")
        for key in (
            "provider", "model", "prompt", "scope", "batch_size", "shuffle_seed",
            "prompt_sha256", "batch_format_sha256",
        ):
            if previous_manifest.get(key) != metadata[key]:
                raise ValueError(f"run settings changed since checkpoint ({key}): {manifest_path}")
    manifest = {
        **metadata,
        "expected_rows": len(by_id),
        "variant_id_sha256": variant_digest,
        "input_sha256": input_digest,
        "complete": False,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    remaining = [
        record for key, record in by_id.items()
        if key not in done or done[key].get("prediction") is None
    ]
    log(f"{model} [{scope}] {len(done)} on disk, {len(remaining)} variants to classify")

    for batch_number, batch in enumerate(_batches(remaining, batch_size, seed), start=1):
        labels = classify_batch(
            send_prompt,
            [record["text"] for record in batch],
            task,
            prompt_name,
            log=log,
        )
        records = []
        for variant, prediction in zip(batch, labels):
            records.append({
                **variant,
                "prediction": prediction,
                "provider": provider,
                "model": model,
                "prompt": _task_label(task, prompt_name),
                "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            })
        append_jsonl(records, path)
        done.update((record["variant_id"], record) for record in records)
        log(f"  batch {batch_number}: {sum(r['prediction'] is not None for r in records)}/{len(records)} parsed")

    expected = set(by_id)
    completed = {key for key in expected if key in done and done[key].get("prediction") is not None}
    if completed != expected:
        missing = sorted(expected - completed)
        raise RuntimeError(f"variant run incomplete: {len(missing)} missing/unparsed rows")
    ordered = [done[key] for key in sorted(expected)]
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in ordered) + "\n",
        encoding="utf-8",
    )
    manifest["complete"] = True
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return [done[key] for key in by_id]


def run_primary(rows, variants, provider, model, stage, scope, sender=None, log=print):
    """Use the primary prompt and the standard output tree for an experiment stage."""
    path = LABELS_DIR / stage / f"{scope}__{provider}__{model.replace('/', '_')}.jsonl"
    return run_variants(
        variants, provider, model, path, scope, sender=sender, log=log,
    )


def read_jsonl(path: Path) -> list[dict]:
    return load_jsonl(path)
