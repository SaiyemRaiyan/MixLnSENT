"""Hard-label occlusion and sampled Shapley-kernel attribution."""

import itertools
import math

import numpy as np

from .config import COALITION_ALPHA
from .variants import exact_shapley


def occlusion_scores(base_label: int, token_labels: list[int | None]) -> list[float]:
    """Positive scores indicate that deleting the token changes the label."""
    return [float(label is not None and label != base_label) for label in token_labels]


def _shapley_kernel(n: int, size: int) -> float:
    return (n - 1) / (math.comb(n, size) * size * (n - size))


def fit_kernel_attribution(
    values: dict[tuple[int, ...], float],
    n: int,
    alpha: float = COALITION_ALPHA,
) -> list[float]:
    """Fit the plan's weighted ridge regression with an intercept."""
    if n < 2:
        return [0.0] * n
    rows = []
    outcomes = []
    weights = []
    for coalition, value in values.items():
        if not coalition or len(coalition) == n:
            continue
        rows.append([1.0] + [1.0 if index in coalition else 0.0 for index in range(n)])
        outcomes.append(float(value))
        weights.append(_shapley_kernel(n, len(coalition)))
    if not rows:
        raise ValueError("at least one non-empty, non-full coalition is required")
    design = np.asarray(rows, dtype=float)
    target = np.asarray(outcomes, dtype=float)
    root_weight = np.sqrt(np.asarray(weights, dtype=float))
    weighted_design = design * root_weight[:, None]
    weighted_target = target * root_weight
    penalty = np.eye(n + 1, dtype=float) * alpha
    penalty[0, 0] = 0.0
    solution = np.linalg.solve(
        weighted_design.T @ weighted_design + penalty,
        weighted_design.T @ weighted_target,
    )
    return solution[1:].tolist()


def coalition_values(
    predictions: dict[tuple[int, ...], int],
    base_label: int,
    n: int,
) -> dict[tuple[int, ...], float]:
    """Convert predictions to agreement values, including empty/full endpoints."""
    full = tuple(range(n))
    if full not in predictions:
        raise ValueError("full-coalition baseline prediction is missing")
    values = {tuple(sorted(mask)): float(label == base_label)
              for mask, label in predictions.items()}
    values[()] = 0.0
    values[full] = float(predictions[full] == base_label)
    return values


def enumerate_coalition_values(game, n: int) -> dict[tuple[int, ...], float]:
    """Evaluate a callable game on every coalition, including empty and full."""
    return {
        coalition: float(game(coalition))
        for size in range(n + 1)
        for coalition in itertools.combinations(range(n), size)
    }


def top_indices(scores: list[float], k: int, tie_break: list[float] | None = None) -> list[int]:
    tie_break = tie_break or [0.0] * len(scores)
    return sorted(
        range(len(scores)),
        key=lambda index: (-scores[index], -tie_break[index], index),
    )[:k]


def top_k_jaccard(first: list[int], second: list[int]) -> float:
    left, right = set(first), set(second)
    union = left | right
    return len(left & right) / len(union) if union else 1.0
