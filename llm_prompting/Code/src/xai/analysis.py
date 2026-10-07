"""Reproducible score construction from complete behavioral-XAI run files."""

import json
from collections import defaultdict
from pathlib import Path

from .attribution import (
    coalition_values, fit_kernel_attribution, occlusion_scores, top_indices,
    top_k_jaccard,
)
from .config import ATTRIBUTIONS_DIR, ensure_output_dirs
from .gates import check_prereg
from .sampling import load_sample
from .tokenize import tokenize


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def make_occlusion_attributions(
    rows: list[dict], baseline: dict[int, int], occlusion_records: list[dict]
) -> list[dict]:
    grouped = defaultdict(dict)
    for record in occlusion_records:
        if record.get("kind") != "occlusion_delete":
            continue
        mask = record.get("mask", [])
        if mask:
            grouped[record["src_index"]][mask[0]] = record.get("prediction")
    result = []
    for row in rows:
        source = row["index"]
        if source not in baseline:
            continue
        tokens = tokenize(row["sentence"])
        predictions = [
            grouped[source].get(token.index)
            for token in tokens if token.index in grouped[source]
        ]
        indices = sorted(grouped[source])
        scores = occlusion_scores(baseline[source], predictions)
        if len(indices) != len(tokens):
            aligned = [0.0] * len(tokens)
            for index, score in zip(indices, scores):
                aligned[index] = score
            scores = aligned
        result.append({
            "src_index": source,
            "gold": row["gold"],
            "baseline": baseline[source],
            "token_texts": [token.text for token in tokens],
            "scores": scores,
            "top_indices": top_indices(scores, min(3, len(scores))),
        })
    return result


def make_coalition_attributions(
    rows: list[dict],
    baseline: dict[int, int],
    coalition_records: list[dict],
) -> list[dict]:
    row_by_index = {row["index"]: row for row in rows}
    grouped = defaultdict(lambda: defaultdict(dict))
    for record in coalition_records:
        if record.get("kind") != "coalition":
            continue
        mask = tuple(sorted(record.get("mask", [])))
        grouped[record["src_index"]][record.get("rep", 0)][mask] = record.get("prediction")
    result = []
    for source, replicate_predictions in grouped.items():
        if source not in baseline or source not in row_by_index:
            continue
        n = len(tokenize(row_by_index[source]["sentence"]))
        full = tuple(range(n))
        seed_scores = {}
        for replicate, predictions in replicate_predictions.items():
            predictions[full] = baseline[source]
            values = coalition_values(predictions, baseline[source], n)
            seed_scores[str(replicate)] = fit_kernel_attribution(values, n)
        if not seed_scores:
            continue
        averaged = [
            sum(scores[index] for scores in seed_scores.values()) / len(seed_scores)
            for index in range(n)
        ]
        result.append({
            "src_index": source,
            "gold": row_by_index[source]["gold"],
            "baseline": baseline[source],
            "token_texts": [token.text for token in tokenize(row_by_index[source]["sentence"])],
            "scores": averaged,
            "scores_by_seed": seed_scores,
            "top_indices": top_indices(averaged, min(3, n)),
        })
    return sorted(result, key=lambda record: record["src_index"])


def coalition_seed_stability(
    rows: list[dict],
    baseline: dict[int, int],
    coalition_records: list[dict],
) -> dict:
    row_by_index = {row["index"]: row for row in rows}
    grouped = defaultdict(lambda: defaultdict(dict))
    for record in coalition_records:
        if record.get("kind") == "coalition":
            grouped[record["src_index"]][record.get("rep", 0)][
                tuple(sorted(record.get("mask", [])))
            ] = record.get("prediction")
    overlaps = []
    for source, seeds in grouped.items():
        if source not in baseline or source not in row_by_index:
            continue
        n = len(tokenize(row_by_index[source]["sentence"]))
        full = tuple(range(n))
        rankings = []
        for predictions in seeds.values():
            predictions[full] = baseline[source]
            values = coalition_values(predictions, baseline[source], n)
            scores = fit_kernel_attribution(values, n)
            rankings.append(top_indices(scores, min(3, n)))
        if len(rankings) >= 2:
            overlaps.append(top_k_jaccard(rankings[0], rankings[1]))
    return {
        "n": len(overlaps),
        "mean_top3_jaccard": sum(overlaps) / len(overlaps) if overlaps else None,
        "per_sentence": overlaps,
    }


def write_attributions(
    rows: list[dict],
    baseline: dict[int, int],
    occlusion_records: list[dict],
    coalition_records: list[dict],
    model_key: str,
    scope: str,
) -> dict:
    if scope == "test":
        check_prereg()
    ensure_output_dirs()
    safe_model = model_key.replace("/", "_").replace(":", "_")
    occlusion = make_occlusion_attributions(rows, baseline, occlusion_records)
    coalition = make_coalition_attributions(rows, baseline, coalition_records)
    return {
        "occlusion": _write_jsonl(
            ATTRIBUTIONS_DIR / "E4" / f"{scope}__{safe_model}.jsonl", occlusion
        ),
        "coalition": _write_jsonl(
            ATTRIBUTIONS_DIR / "E5" / f"{scope}__{safe_model}.jsonl", coalition
        ),
    }
