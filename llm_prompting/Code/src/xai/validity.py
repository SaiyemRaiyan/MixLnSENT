"""E1 ground-truth pipeline checks and planted-cue measurement helpers."""

from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

from src.common.data import load_bnsentmix, split_indices
from src.filtering.classifiers import logistic_regression

from .attribution import occlusion_scores, top_indices
from .tokenize import delete_tokens, tokenize


def fit_lr_oracle():
    dataset = load_bnsentmix()
    train_indices = split_indices(dataset)["train"]
    model = logistic_regression()
    model.fit(
        [dataset[index]["Sentence"] for index in train_indices],
        [dataset[index]["Label"] for index in train_indices],
    )
    return model


def compare_occlusion_to_lr(model, rows: list[dict]) -> dict:
    """Compare hard-label occlusion ranking with exact predict_proba drops."""
    per_sentence = []
    for row in rows:
        tokens = tokenize(row["sentence"])
        if len(tokens) < 4:
            continue
        baseline_prediction = int(model.predict([row["sentence"]])[0])
        baseline_prob = model.predict_proba([row["sentence"]])[0]
        class_position = list(model.classes_).index(baseline_prediction)
        hard_scores = []
        probability_drops = []
        for token in tokens:
            changed = delete_tokens(row["sentence"], [token.index])
            if not changed:
                continue
            prediction = int(model.predict([changed])[0])
            probability = model.predict_proba([changed])[0][class_position]
            hard_scores.append(float(prediction != baseline_prediction))
            probability_drops.append(float(baseline_prob[class_position] - probability))
        if len(hard_scores) < 2:
            continue
        identifiable = len(set(hard_scores)) > 1
        rho = None
        top1_agreement = None
        if identifiable:
            observed_rho = float(spearmanr(hard_scores, probability_drops).statistic)
            if np.isfinite(observed_rho):
                rho = observed_rho
            hard_top = top_indices(hard_scores, 1)
            probability_top = top_indices(probability_drops, 1)
            top1_agreement = hard_top[0] == probability_top[0]
        per_sentence.append({
            "index": row["index"],
            "identifiable": identifiable,
            "spearman": rho,
            "top1_agreement": top1_agreement,
        })
    identifiable_rows = [
        row for row in per_sentence if row["spearman"] is not None
    ]
    return {
        "n": len(per_sentence),
        "identifiable_n": len(identifiable_rows),
        "unidentifiable_n": len(per_sentence) - len(identifiable_rows),
        "median_spearman": float(
            np.median([row["spearman"] for row in identifiable_rows])
        ) if identifiable_rows else None,
        "top1_agreement": sum(
            row["top1_agreement"] for row in identifiable_rows
        ) / len(identifiable_rows) if identifiable_rows else None,
        "per_sentence": per_sentence,
    }


def select_lr_validation_rows(rows: list[dict], count: int = 300, seed: int = 43) -> list[dict]:
    import random

    candidates = [
        row for row in rows if len(tokenize(row["sentence"])) >= 4
    ]
    if len(candidates) < count:
        raise ValueError(f"need {count} dev rows of at least four tokens; found {len(candidates)}")
    random.Random(seed).shuffle(candidates)
    return candidates[:count]


def planted_cue_rows(rows: list[dict], per_cue: int = 100, seed: int = 43) -> list[dict]:
    """Construct cue insertions at seeded token positions without changing labels."""
    import random

    neutral = [row for row in rows if row["gold"] == 2]
    if len(neutral) < per_cue * 2:
        raise ValueError(f"need {per_cue * 2} neutral dev rows; found {len(neutral)}")
    rng = random.Random(seed)
    rng.shuffle(neutral)
    result = []
    for cue, candidates in (("bhalo", neutral[:per_cue]), ("kharap", neutral[per_cue:2 * per_cue])):
        for row in candidates:
            words = row["sentence"].split()
            position = rng.randrange(len(words) + 1)
            words.insert(position, cue)
            result.append({
                **row,
                "index": row["index"],
                "sentence": " ".join(words),
                "planted_cue": cue,
                "planted_position": position,
            })
    return result


def planted_cue_top1_rate(
    rows: list[dict], labels: dict[int, int], scores: dict[int, list[float]], baseline: dict[int, int]
) -> dict:
    eligible = []
    for row in rows:
        source = row["index"]
        if source not in labels or source not in scores:
            continue
        if labels[source] == baseline.get(source):
            continue
        tokens = tokenize(row["sentence"])
        planted_index = row["planted_position"]
        if planted_index >= len(tokens) or tokens[planted_index].text.casefold() != row["planted_cue"]:
            continue
        ranked = top_indices(scores[source], 1)
        eligible.append(ranked[0] == planted_index)
    return {
        "n": len(eligible),
        "top1_rate": sum(eligible) / len(eligible) if eligible else None,
    }
