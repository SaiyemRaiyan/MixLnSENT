"""E0 instrument-order, replicate and batch-sensitivity conditions."""

from collections import defaultdict

from sklearn.metrics import cohen_kappa_score

from .tokenize import tokenize
from .variants import make_variant


def instrument_conditions(rows: list[dict]) -> dict[str, list[dict]]:
    original = [
        make_variant(row, "E0", [], row["sentence"], rep=0)
        for row in rows
    ]
    shuffled = [
        make_variant(row, "E0", [], row["sentence"], rep=1)
        for row in rows
    ]
    single = [
        make_variant(row, "E0", [], row["sentence"], rep=2)
        for row in rows
    ]
    paired = []
    for row in rows:
        paired.append(make_variant(row, "E0", [], row["sentence"], rep=3))
        tokens = tokenize(row["sentence"])
        if len(tokens) > 1:
            paired.append(
                make_variant(
                    row, "E0_delete", [0],
                    " ".join(token.text for token in tokens[1:]),
                )
            )
    return {"A": original, "A2": original, "B": shuffled, "C": single, "D": paired}


def agreement(first: list[dict], second: list[dict]) -> dict:
    left = {record["src_index"]: record.get("prediction") for record in first}
    right = {record["src_index"]: record.get("prediction") for record in second}
    common = sorted(
        index for index in left.keys() & right.keys()
        if left[index] is not None and right[index] is not None
    )
    a = [left[index] for index in common]
    b = [right[index] for index in common]
    return {
        "n": len(common),
        "agreement": sum(x == y for x, y in zip(a, b)) / len(common) if common else None,
        "cohen_kappa": float(cohen_kappa_score(a, b)) if common else None,
    }


def majority_labels(replicates: list[list[dict]]) -> tuple[dict[int, int], list[int]]:
    predictions = defaultdict(list)
    for records in replicates:
        for record in records:
            if record.get("prediction") is not None:
                predictions[record["src_index"]].append(record["prediction"])
    labels = {}
    unstable = []
    for source, outcomes in predictions.items():
        counts = {label: outcomes.count(label) for label in set(outcomes)}
        best = max(counts.values())
        winners = [label for label, count in counts.items() if count == best]
        if len(winners) == 1 and best >= 2:
            labels[source] = winners[0]
        else:
            unstable.append(source)
    return labels, unstable
