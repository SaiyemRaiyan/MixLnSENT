"""Language-tag prompt parsing and the hand-label quality gate."""

import re

TAGS = {"EN", "BN-roman", "OTHER"}
_LINE = re.compile(r"^\s*(\d+)\s*[:.)-]\s*(EN|BN-roman|OTHER)\s*$", re.IGNORECASE)


def parse_langtag_batch(response: str, expected_count: int):
    found = {}
    for line in response.splitlines():
        match = _LINE.match(line)
        if match:
            found[int(match.group(1))] = match.group(2).upper()
    # Preserve the canonical spelling for the mixed-script class.
    parsed = [
        ("BN-roman" if found.get(index) == "BN-ROMAN" else found.get(index))
        for index in range(1, expected_count + 1)
    ]
    missing = [index for index, label in enumerate(parsed, 1) if label is None]
    return parsed, missing


def tagging_accuracy(gold: list[str], predicted: list[str]) -> dict:
    if len(gold) != len(predicted):
        raise ValueError("gold and predicted token tags must have equal lengths")
    unknown = sorted((set(gold) | set(predicted)) - TAGS)
    if unknown:
        raise ValueError(f"unknown language tags: {unknown}")
    correct = sum(left == right for left, right in zip(gold, predicted))
    return {
        "n": len(gold),
        "accuracy": correct / len(gold) if gold else None,
        "gate_g4": bool(gold) and correct / len(gold) >= 0.90,
    }
