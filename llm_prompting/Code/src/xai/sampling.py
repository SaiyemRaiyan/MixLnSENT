"""Seeded, stratified BnSentMix sample creation with content hashes."""

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from src.common.data import dev_indices, duplicate_summary, load_bnsentmix, normalize_sentence, split_indices

from .config import (
    ATTRIBUTION_PER_CLASS, GEMINI_PER_CLASS, MAX_TOKENS, SAMPLE_PER_CLASS,
    SAMPLE_SEED, SAMPLES_DIR, ensure_output_dirs,
)
from .tokenize import tokenize


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _conflicting_duplicate_indices(
    texts: list[str], labels: list[int], indices: list[int]
) -> set[int]:
    groups: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for index in indices:
        groups[normalize_sentence(texts[index])].append((index, labels[index]))
    return {
        index
        for members in groups.values()
        if len({label for _, label in members}) > 1
        for index, _ in members
    }


def _sample_rows(
    dataset,
    indices: list[int],
    limit_per_class: int | None,
    seed: int,
    conflict_indices: list[int] | None = None,
) -> list[dict]:
    texts = list(dataset["Sentence"])
    labels = list(dataset["Label"])
    conflicted = _conflicting_duplicate_indices(
        texts, labels, indices if conflict_indices is None else conflict_indices
    )
    eligible = []
    for index in indices:
        text = texts[index]
        count = len(tokenize(text))
        if index in conflicted or count > MAX_TOKENS:
            continue
        eligible.append((index, text, labels[index], count))

    rng = random.Random(seed)
    by_label: dict[int, list[tuple[int, str, int, int]]] = defaultdict(list)
    for row in eligible:
        by_label[row[2]].append(row)

    selected = []
    for label in sorted(by_label):
        rows = by_label[label]
        if limit_per_class is not None and len(rows) < limit_per_class:
            raise ValueError(
                f"Need {limit_per_class} eligible rows for label {label}; found {len(rows)}"
            )
        rng.shuffle(rows)
        selected.extend(rows if limit_per_class is None else rows[:limit_per_class])
    rng.shuffle(selected)
    return [
        {"index": index, "sentence": text, "gold": label, "token_count": count}
        for index, text, label, count in selected
    ]


def _write_sample(name: str, rows: list[dict], details: dict) -> dict:
    path = SAMPLES_DIR / f"{name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "file": str(path.relative_to(path.parents[1])),
        "sha256": sha256_file(path),
        "rows": len(rows),
        "class_counts": dict(sorted(Counter(row["gold"] for row in rows).items())),
        **details,
    }


def _write_manifest(manifest: dict) -> Path:
    path = SAMPLES_DIR / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def prepare_dev() -> dict:
    """Create only the train-derived development sample."""
    ensure_output_dirs()
    dataset = load_bnsentmix()
    indices = dev_indices(dataset)
    rows = _sample_rows(dataset, indices, limit_per_class=None, seed=SAMPLE_SEED)
    neutral_count = sum(row["gold"] == 2 for row in rows)
    extra_neutral_count = max(0, 200 - neutral_count)
    if extra_neutral_count:
        train_indices = split_indices(dataset)["train"]
        remaining_train = [index for index in train_indices if index not in set(indices)]
        extra_rows = _sample_rows(
            dataset,
            remaining_train,
            limit_per_class=None,
            seed=SAMPLE_SEED + 1,
        )
        neutral_rows = [row for row in extra_rows if row["gold"] == 2]
        if len(neutral_rows) < extra_neutral_count:
            raise ValueError(
                f"need {extra_neutral_count} additional neutral train rows for the planted-cue gate"
            )
        rows.extend(neutral_rows[:extra_neutral_count])
    sample = _write_sample(
        "dev",
        rows,
        {
            "source": "filtered 600-row train-only dev set plus extra train neutral rows for planted-cue validation",
            "extra_neutral_rows": extra_neutral_count,
        },
    )
    manifest = {
        "seed": SAMPLE_SEED,
        "split_seed": 42,
        "max_tokens": MAX_TOKENS,
        "dev": sample,
    }
    previous_path = SAMPLES_DIR / "manifest.json"
    if previous_path.exists():
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
        for name in ("xai800", "xai200", "gemini300", "gemini_noise100", "test_source_rows"):
            if name in previous:
                manifest[name] = previous[name]
    _write_manifest(manifest)
    return manifest


def prepare_test() -> dict:
    """Create XAI-800, XAI-200 and restricted Gemini samples from the fixed test split."""
    ensure_output_dirs()
    dataset = load_bnsentmix()
    indices = split_indices(dataset)["test"]
    duplicate_stats = duplicate_summary(dataset)
    texts = list(dataset["Sentence"])
    labels = list(dataset["Label"])
    conflicted = _conflicting_duplicate_indices(
        texts, labels, list(range(len(texts)))
    )
    eligible_count = sum(
        index not in conflicted and len(tokenize(texts[index])) <= MAX_TOKENS
        for index in indices
    )
    sample_rows = _sample_rows(
        dataset,
        indices,
        limit_per_class=SAMPLE_PER_CLASS,
        seed=SAMPLE_SEED,
        conflict_indices=list(range(len(texts))),
    )
    xai800 = _write_sample(
        "xai800",
        sample_rows,
        {
            "source": "fixed stratified test split",
            "eligible_excluded_fraction": round(1 - eligible_count / len(indices), 6),
            "eligible_rows_before_class_cap": eligible_count,
            "rows_not_sampled_after_class_cap": eligible_count - len(sample_rows),
            "duplicate_summary": duplicate_stats,
        },
    )
    by_label: dict[int, list[dict]] = defaultdict(list)
    for row in sample_rows:
        by_label[row["gold"]].append(row)

    def subset(per_class: int, seed: int, name: str) -> dict:
        local_rng = random.Random(seed)
        selected = []
        for label in sorted(by_label):
            candidates = list(by_label[label])
            local_rng.shuffle(candidates)
            selected.extend(candidates[:per_class])
        local_rng.shuffle(selected)
        return _write_sample(name, selected, {"source": "seeded subset of xai800"})

    # XAI-200 is 200 rows total, 50 per class; the generic sampler's naming in
    # the plan means a 50-per-class stratified subset, not 200 rows per class.
    xai200 = subset(ATTRIBUTION_PER_CLASS, SAMPLE_SEED + 20, "xai200")
    gemini = subset(GEMINI_PER_CLASS, SAMPLE_SEED + 30, "gemini300")
    gemini_noise = subset(25, SAMPLE_SEED + 31, "gemini_noise100")
    manifest = {
        "seed": SAMPLE_SEED,
        "split_seed": 42,
        "max_tokens": MAX_TOKENS,
        "test_source_rows": len(indices),
        "xai800": xai800,
        "xai200": xai200,
        "gemini300": gemini,
        "gemini_noise100": gemini_noise,
    }
    # Preserve dev's already-written manifest entry without loading test data from it.
    previous_path = SAMPLES_DIR / "manifest.json"
    if previous_path.exists():
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
        if "dev" in previous:
            manifest["dev"] = previous["dev"]
    _write_manifest(manifest)
    return manifest


def load_sample(name: str) -> list[dict]:
    path = SAMPLES_DIR / f"{name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Sample {name!r} has not been prepared: {path}")
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]
