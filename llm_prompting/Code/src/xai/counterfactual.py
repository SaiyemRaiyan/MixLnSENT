"""Counterfactual operators and qualification counts."""

from collections import Counter

from .variants import antonym_variant, contrast_variants, negation_variants


def generate_counterfactuals(rows: list[dict], baseline_labels: dict[int, int]):
    variants = []
    qualified = Counter()
    for row in rows:
        baseline = baseline_labels.get(row["index"])
        contrast = contrast_variants(row, baseline)
        contrast = [variant for variant in contrast if variant["text"].strip()]
        variants.extend(contrast)
        for variant in contrast:
            qualified[variant["kind"]] += 1
        antonym = antonym_variant(row)
        if antonym is not None and antonym["text"].strip():
            variants.append(antonym)
            qualified[antonym["kind"]] += 1
        negations = negation_variants(row)
        negations = [variant for variant in negations if variant["text"].strip()]
        variants.extend(negations)
        qualified["negation_deletion"] += len(negations)
    return variants, dict(qualified)


def expected_change(kind: str, gold: int, baseline: int | None = None, prediction: int | None = None) -> bool:
    if prediction is None:
        return False
    if kind == "clause_order_swap":
        return baseline is not None and prediction == baseline
    if kind in ("clause_delete_left", "clause_delete_right"):
        return baseline == 3 and prediction != 3
    if kind == "antonym_swap" and gold in (0, 1):
        return prediction == 1 - gold
    return False
