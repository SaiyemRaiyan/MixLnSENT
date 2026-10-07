"""Preparation, validation and scoring for the two human annotation tasks."""

import csv
import random
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

from .config import ANNOTATION_DIR, SAMPLE_SEED, ensure_output_dirs
from .tokenize import tokenize


def prepare_annotation_templates(dev_rows: list[dict], test_rows: list[dict]) -> dict:
    ensure_output_dirs()
    rng = random.Random(SAMPLE_SEED + 40)
    dev_tokens = [
        (row, token)
        for row in dev_rows
        for token in tokenize(row["sentence"])
    ]
    rng.shuffle(dev_tokens)
    if len(dev_tokens) < 300:
        raise ValueError(f"need 300 dev tokens for G4; found {len(dev_tokens)}")
    lang_path = ANNOTATION_DIR / "language_tags_300.csv"
    with lang_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("source_index", "token_index", "token", "tag"))
        for row, token in dev_tokens[:300]:
            writer.writerow((row["index"], token.index, token.text, ""))

    selected_by_class = {}
    for label in range(4):
        candidates = [row for row in test_rows if row["gold"] == label]
        rng.shuffle(candidates)
        if len(candidates) < 30:
            raise ValueError(f"need 30 XAI-800 rows for class {label}")
        selected_by_class[label] = candidates[:30]
    evidence_rows = [
        row
        for label in range(4)
        for row in selected_by_class[label]
    ]
    rng.shuffle(evidence_rows)
    overlap = set()
    for label in range(4):
        overlap.update(row["index"] for row in selected_by_class[label][:10])
    evidence_path = ANNOTATION_DIR / "human_evidence_120.csv"
    with evidence_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow((
            "source_index", "sentence", "gold_label", "annotator",
            "evidence_token_indices", "evidence_tokens",
        ))
        for row in evidence_rows:
            writer.writerow((
                row["index"], row["sentence"], row["gold"], "annotator_1", "", "",
            ))
            if row["index"] in overlap:
                writer.writerow((
                    row["index"], row["sentence"], row["gold"], "annotator_2", "", "",
                ))
    from src.audit.definition_compliance import CONTRAST_STRICT
    from .variants import ANTONYM_PAIRS

    signoff_path = ANNOTATION_DIR / "operator_signoff.csv"
    markers = (
        CONTRAST_STRICT.pattern.partition("(")[2]
        .partition(")")[0]
        .split("|")
    )
    with signoff_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("operator", "source", "replacement", "approved", "notes"))
        for positive, negative in ANTONYM_PAIRS:
            writer.writerow(("antonym", positive, negative, "", ""))
            writer.writerow(("antonym", negative, positive, "", ""))
        for token in ("nai", "nei"):
            writer.writerow(("negation", token, "", "", ""))
        for marker in markers:
            writer.writerow(("contrast_marker", marker, "", "", ""))
    return {
        "language_tags": str(lang_path),
        "human_evidence": str(evidence_path),
        "operator_signoff": str(signoff_path),
    }


def validate_language_annotations(path: Path) -> dict:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 300:
        raise ValueError(f"language tag sheet must have 300 rows, found {len(rows)}")
    allowed = {"EN", "BN-roman", "OTHER"}
    invalid = [
        row["source_index"] for row in rows
        if row.get("tag", "").strip() not in allowed
    ]
    if invalid:
        raise ValueError(f"missing or invalid tags for {len(invalid)} rows")
    return {"rows": len(rows), "valid": True}


def validate_evidence_annotations(path: Path) -> dict:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    annotators = {row["annotator"] for row in rows}
    first = {row["source_index"] for row in rows if row["annotator"] == "annotator_1"}
    second = {row["source_index"] for row in rows if row["annotator"] == "annotator_2"}
    if len(first) != 120:
        raise ValueError(f"expected 120 primary annotations, found {len(first)}")
    if len(first & second) != 40:
        raise ValueError(f"expected 40 overlap annotations, found {len(first & second)}")
    if annotators - {"annotator_1", "annotator_2"}:
        raise ValueError(f"unexpected annotator ids: {sorted(annotators)}")
    return {"primary": len(first), "overlap": len(first & second), "valid": True}


def token_f1(human_indices: set[int], model_indices: set[int]) -> float:
    if not human_indices and not model_indices:
        return 1.0
    if not human_indices or not model_indices:
        return 0.0
    overlap = len(human_indices & model_indices)
    precision = overlap / len(model_indices)
    recall = overlap / len(human_indices)
    return 2 * precision * recall / (precision + recall)


def score_language_annotations(path: Path, predicted_tags: dict[tuple[int, int], str]) -> dict:
    validate_language_annotations(path)
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    gold = [row["tag"].strip() for row in rows]
    predicted = [
        predicted_tags.get((int(row["source_index"]), int(row["token_index"])))
        for row in rows
    ]
    missing = sum(tag is None for tag in predicted)
    if missing:
        raise ValueError(f"language-tag predictions missing for {missing} annotated tokens")
    correct = sum(expected == actual for expected, actual in zip(gold, predicted))
    return {
        "n": len(rows),
        "accuracy": correct / len(rows),
        "gate_g4": correct / len(rows) >= 0.90,
    }


def _annotation_indices(row: dict) -> set[int]:
    raw = row.get("evidence_token_indices", "").strip()
    if not raw:
        return set()
    try:
        indices = {int(value.strip()) for value in raw.split(",") if value.strip()}
    except ValueError as error:
        raise ValueError(
            f"invalid token index list for source {row.get('source_index')}"
        ) from error
    token_count = len(tokenize(row["sentence"]))
    if any(index < 0 or index >= token_count for index in indices):
        raise ValueError(
            f"token index outside source {row.get('source_index')} sentence"
        )
    return indices


def score_evidence_annotations(path: Path, model_evidence: dict[int, set[int]]) -> dict:
    validate_evidence_annotations(path)
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_annotator = {
        name: {
            row["source_index"]: (row, _annotation_indices(row))
            for row in rows if row["annotator"] == name
        }
        for name in ("annotator_1", "annotator_2")
    }
    overlap = sorted(
        set(by_annotator["annotator_1"]) & set(by_annotator["annotator_2"])
    )
    kappas = []
    for source in overlap:
        first_row, first = by_annotator["annotator_1"][source]
        _, second = by_annotator["annotator_2"][source]
        token_count = len(tokenize(first_row["sentence"]))
        kappas.append(cohen_kappa_score(
            [index in first for index in range(token_count)],
            [index in second for index in range(token_count)],
        ))
    primary_scores = []
    missing_predictions = 0
    for source, (_row, human) in by_annotator["annotator_1"].items():
        if source not in model_evidence:
            missing_predictions += 1
            continue
        primary_scores.append(token_f1(human, model_evidence[source]))
    return {
        "primary_n": len(primary_scores),
        "missing_model_predictions": missing_predictions,
        "mean_token_f1": sum(primary_scores) / len(primary_scores) if primary_scores else None,
        "overlap_n": len(kappas),
        "mean_interannotator_kappa": (
            sum(value for value in kappas if value == value) / sum(value == value for value in kappas)
            if any(value == value for value in kappas) else None
        ),
    }
