"""Deterministic input perturbations used by behavioural attribution."""

import hashlib
import itertools
import json
import math
import random
from collections.abc import Iterable

from src.audit.definition_compliance import CONTRAST_STRICT, NEGATIVE_CUES, POSITIVE_CUES

from .config import COALITION_BUDGETS, RANDOM_DRAWS, SAMPLE_SEED
from .tokenize import delete_tokens, keep_tokens, tokenize


def variant_id(src_index: int, kind: str, mask: Iterable[int], rep: int = 0) -> str:
    identity = {"mask": sorted(mask), "rep": rep}
    digest = hashlib.sha256(
        json.dumps(identity, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:8]
    return f"{src_index}:{kind}:{digest}"


def make_variant(row: dict, kind: str, mask: Iterable[int], text: str, rep: int = 0) -> dict:
    mask = sorted(mask)
    return {
        "variant_id": variant_id(row["index"], kind, mask, rep),
        "src_index": row["index"],
        "kind": kind,
        "mask": mask,
        "rep": rep,
        "text": text,
    }


def baseline_variants(rows: list[dict], replicates: int = 3) -> list[dict]:
    return [
        make_variant(row, f"baseline_rep_{rep}", [], row["sentence"], rep)
        for row in rows
        for rep in range(1, replicates + 1)
    ]


def occlusion_variants(row: dict, masking: str = "delete") -> list[dict]:
    tokens = tokenize(row["sentence"])
    result = []
    for token in tokens:
        if len(tokens) == 1:
            continue
        changed = (
            delete_tokens(row["sentence"], [token.index])
            if masking == "delete"
            else " ".join(
                "[MASK]" if item.index == token.index else item.text for item in tokens
            )
        )
        result.append(make_variant(row, f"occlusion_{masking}", [token.index], changed))
    if masking not in ("delete", "mask"):
        raise ValueError("masking must be 'delete' or 'mask'")
    return result


def _weighted_size(n: int, rng: random.Random) -> int:
    sizes = list(range(1, n))
    weights = [(n - 1) / (size * (n - size)) for size in sizes]
    return rng.choices(sizes, weights=weights, k=1)[0]


def sample_coalitions(n: int, budget: int, seed: int) -> list[tuple[int, ...]]:
    if n < 2:
        return []
    all_masks = [
        mask
        for size in range(1, n)
        for mask in itertools.combinations(range(n), size)
    ]
    if len(all_masks) <= budget:
        return all_masks
    rng = random.Random(seed)
    masks: set[tuple[int, ...]] = set()
    target = min(budget, len(all_masks))
    target -= target % 2
    while len(masks) < target:
        size = _weighted_size(n, rng)
        selected = tuple(sorted(rng.sample(range(n), size)))
        complement = tuple(index for index in range(n) if index not in selected)
        additions = {selected, complement} - masks
        if len(masks) + len(additions) <= target:
            masks.update(additions)
    return sorted(masks)


def coalition_variants(
    row: dict, budget: int, seed: int, rep: int = 0
) -> list[dict]:
    tokens = tokenize(row["sentence"])
    variants = []
    for mask in sample_coalitions(len(tokens), budget, seed):
        variants.append(
            make_variant(
                row,
                "coalition",
                mask,
                keep_tokens(row["sentence"], mask),
                rep=rep,
            )
        )
    return variants


def evidence_variants(row: dict, evidence: Iterable[int], kind: str) -> tuple[dict | None, dict | None]:
    tokens = tokenize(row["sentence"])
    evidence = sorted(set(evidence))
    if not evidence or len(evidence) >= len(tokens):
        return None, None
    remove = make_variant(
        row, f"{kind}_comprehensiveness", evidence,
        delete_tokens(row["sentence"], evidence),
    )
    keep = make_variant(
        row, f"{kind}_sufficiency", evidence,
        keep_tokens(row["sentence"], evidence),
    )
    return remove, keep


def random_evidence(
    row: dict, size: int, draws: int = RANDOM_DRAWS, seed: int = SAMPLE_SEED
) -> list[list[int]]:
    n = len(tokenize(row["sentence"]))
    if size < 1 or size >= n:
        return []
    rng = random.Random(seed + row["index"] * 1009 + size)
    return [sorted(rng.sample(range(n), size)) for _ in range(draws)]


def _marker_token_indices(sentence: str) -> list[int]:
    return [
        token.index for token in tokenize(sentence)
        if CONTRAST_STRICT.search(token.text)
    ]


def _split_around_first_marker(sentence: str) -> tuple[list[str], list[str], str] | None:
    tokens = tokenize(sentence)
    for marker in tokens:
        match = CONTRAST_STRICT.search(marker.text)
        if match:
            return (
                [token.text for token in tokens[:marker.index]],
                [token.text for token in tokens[marker.index + 1:]],
                marker.text,
            )
    return None


def contrast_variants(row: dict, baseline: int | None = None) -> list[dict]:
    parts = _split_around_first_marker(row["sentence"])
    if parts is None:
        return []
    left, right, marker = parts
    tokens = tokenize(row["sentence"])
    marker_indices = _marker_token_indices(row["sentence"])
    results = [
        make_variant(
            row, "contrast_marker_deletion", marker_indices,
            delete_tokens(row["sentence"], marker_indices),
        ),
        make_variant(
            row, "clause_order_swap", [tokens[marker_indices[0]].index],
            " ".join(right + [marker] + left).strip(),
        ),
    ]
    if baseline == 3:
        left_indices = list(range(0, marker_indices[0]))
        right_indices = list(range(marker_indices[0] + 1, len(tokens)))
        left_with_marker = sorted(set(left_indices) | set(marker_indices))
        right_with_marker = sorted(set(right_indices) | set(marker_indices))
        if len(left_with_marker) < len(tokens):
            results.append(
                make_variant(
                    row, "clause_delete_left", left_with_marker,
                    delete_tokens(row["sentence"], left_with_marker),
                )
            )
        if len(right_with_marker) < len(tokens):
            results.append(
                make_variant(
                    row, "clause_delete_right", right_with_marker,
                    delete_tokens(row["sentence"], right_with_marker),
                )
            )
    return results


ANTONYM_PAIRS = (
    ("bhalo", "kharap"), ("valo", "baje"), ("vlo", "baje"),
    ("balo", "kharap"), ("sundor", "kharap"), ("shundor", "kharap"),
    ("darun", "joghonno"), ("khusi", "dukkho"), ("khushi", "dukkho"),
    ("best", "worst"), ("good", "bad"), ("great", "terrible"),
    ("excellent", "awful"), ("awesome", "awful"), ("nice", "awful"),
    ("love", "hate"), ("loved", "hated"), ("happy", "sad"),
    ("perfect", "broken"), ("satisfied", "disappointed"),
)


def antonym_variant(row: dict) -> dict | None:
    hits = []
    pair_by_word = {}
    for positive, negative in ANTONYM_PAIRS:
        pair_by_word[positive] = negative
        pair_by_word[negative] = positive
    for token in tokenize(row["sentence"]):
        normalized = token.text.strip(".,!?;:()[]{}\"'").lower()
        if normalized in pair_by_word:
            hits.append((token, pair_by_word[normalized]))
    if len(hits) != 1 or row.get("gold") not in (0, 1):
        return None
    token, antonym = hits[0]
    text_tokens = tokenize(row["sentence"])
    changed = [
        antonym if item.index == token.index else item.text for item in text_tokens
    ]
    return make_variant(row, "antonym_swap", [token.index], " ".join(changed))


def negation_variants(row: dict) -> list[dict]:
    indices = [
        token.index for token in tokenize(row["sentence"])
        if token.text.strip(".,!?;:()[]{}\"'").lower() in ("nai", "nei")
    ]
    if not indices:
        return []
    return [
        make_variant(
            row, "negation_deletion", [index],
            delete_tokens(row["sentence"], [index]),
        )
        for index in indices
        if len(tokenize(row["sentence"])) > 1
    ]


def exact_shapley(values: dict[tuple[int, ...], float], n: int) -> list[float]:
    """Calculate exact Shapley values from a value for every coalition."""
    if n < 1:
        return []
    empty, full = (), tuple(range(n))
    if empty not in values or full not in values:
        raise ValueError("values must include the empty and full coalitions")
    result = []
    for feature in range(n):
        others = [index for index in range(n) if index != feature]
        score = 0.0
        for size in range(n):
            weight = 1 / (n * math.comb(n - 1, size))
            for coalition in itertools.combinations(others, size):
                with_feature = tuple(sorted((*coalition, feature)))
                score += weight * (values[with_feature] - values[coalition])
        result.append(score)
    return result


def choose_coalition_budget(overlaps: dict[int, float]) -> int:
    for budget in COALITION_BUDGETS:
        if overlaps.get(budget, 0.0) >= 0.8:
            return budget
    return COALITION_BUDGETS[-1]
