"""Seeded bootstrap intervals and paired tests used by the preregistered analysis."""

import math
import random

from scipy.stats import binomtest

from .config import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED


def bootstrap_mean(
    values: list[float],
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
    confidence: float = 0.95,
) -> dict:
    if not values:
        return {"estimate": None, "ci_low": None, "ci_high": None, "n": 0}
    rng = random.Random(seed)
    n = len(values)
    estimates = [
        sum(values[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(resamples)
    ]
    estimates.sort()
    tail = (1 - confidence) / 2
    low_index = int(tail * resamples)
    high_index = min(resamples - 1, math.ceil((1 - tail) * resamples) - 1)
    return {
        "estimate": sum(values) / n,
        "ci_low": estimates[low_index],
        "ci_high": estimates[high_index],
        "n": n,
    }


def stratified_bootstrap(
    values_by_class: dict[int, list[float]],
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict:
    rng = random.Random(seed)
    classes = sorted(values_by_class)
    if not classes or any(not values_by_class[label] for label in classes):
        raise ValueError("every requested class needs at least one observation")
    estimates = []
    for _ in range(resamples):
        sampled = []
        for label in classes:
            values = values_by_class[label]
            sampled.extend(values[rng.randrange(len(values))] for _ in values)
        estimates.append(sum(sampled) / len(sampled))
    estimates.sort()
    return {
        "estimate": sum(sum(values) for values in values_by_class.values())
        / sum(map(len, values_by_class.values())),
        "ci_low": estimates[int(0.025 * resamples)],
        "ci_high": estimates[min(resamples - 1, math.ceil(0.975 * resamples) - 1)],
        "n": sum(map(len, values_by_class.values())),
    }


def paired_stratified_bootstrap(
    differences: list[float],
    labels: list[int],
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict:
    if len(differences) != len(labels):
        raise ValueError("differences and class labels must have equal lengths")
    by_class = {
        label: [value for value, observed in zip(differences, labels) if observed == label]
        for label in sorted(set(labels))
    }
    if not by_class:
        return {"estimate": None, "ci_low": None, "ci_high": None, "n": 0}
    rng = random.Random(seed)
    estimates = []
    for _ in range(resamples):
        sampled = []
        for values in by_class.values():
            sampled.extend(values[rng.randrange(len(values))] for _ in values)
        estimates.append(sum(sampled) / len(sampled))
    estimates.sort()
    return {
        "estimate": sum(differences) / len(differences),
        "ci_low": estimates[int(0.025 * resamples)],
        "ci_high": estimates[min(resamples - 1, math.ceil(0.975 * resamples) - 1)],
        "n": len(differences),
    }


def mcnemar_exact(first_correct: list[bool], second_correct: list[bool]) -> dict:
    if len(first_correct) != len(second_correct):
        raise ValueError("paired correctness arrays must have equal lengths")
    first_only = sum(a and not b for a, b in zip(first_correct, second_correct))
    second_only = sum(not a and b for a, b in zip(first_correct, second_correct))
    discordant = first_only + second_only
    pvalue = (
        float(binomtest(min(first_only, second_only), discordant, 0.5).pvalue)
        if discordant else 1.0
    )
    return {
        "first_only": first_only,
        "second_only": second_only,
        "discordant": discordant,
        "pvalue": pvalue,
        "accuracy_delta": (
            sum(second_correct) / len(second_correct) - sum(first_correct) / len(first_correct)
            if first_correct else 0.0
        ),
    }


def holm_adjust(pvalues: list[float]) -> list[float]:
    indexed = sorted(enumerate(pvalues), key=lambda pair: pair[1])
    adjusted_sorted = []
    running = 0.0
    count = len(pvalues)
    for rank, (_, pvalue) in enumerate(indexed):
        running = max(running, min(1.0, (count - rank) * pvalue))
        adjusted_sorted.append(running)
    adjusted = [0.0] * count
    for (original_index, _), value in zip(indexed, adjusted_sorted):
        adjusted[original_index] = value
    return adjusted
