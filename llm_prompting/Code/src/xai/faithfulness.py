"""Comprehensiveness, sufficiency and matched evidence-vs-random summaries."""

from collections import defaultdict

from .tokenize import tokenize
from .variants import evidence_variants, random_evidence


def intervention_metrics(
    baseline_by_source: dict[int, int],
    intervention_records: list[dict],
    source_rows: dict[int, dict],
) -> dict:
    groups = defaultdict(dict)
    for record in intervention_records:
        groups[record["src_index"]][record["kind"]] = record.get("prediction")
    by_kind = defaultdict(list)
    for source, records in groups.items():
        baseline = baseline_by_source.get(source)
        if baseline is None:
            continue
        row = source_rows[source]
        for kind, prediction in records.items():
            by_kind[kind].append({
                "source": source,
                "gold": row["gold"],
                "changed": prediction is not None and prediction != baseline,
                "retained": prediction is not None and prediction == baseline,
            })
    result = {}
    for kind, observations in by_kind.items():
        result[kind] = {
            "n": len(observations),
            "comprehensiveness": (
                sum(item["changed"] for item in observations) / len(observations)
                if observations else None
            ),
            "sufficiency": (
                sum(item["retained"] for item in observations) / len(observations)
                if observations else None
            ),
            "per_class": {
                str(label): {
                    "n": sum(item["gold"] == label for item in observations),
                    "change_rate": (
                        sum(item["changed"] for item in observations if item["gold"] == label)
                        / sum(item["gold"] == label for item in observations)
                        if any(item["gold"] == label for item in observations) else None
                    ),
                }
                for label in sorted({item["gold"] for item in observations})
            },
        }
    return result


def paired_faithfulness_gap(evidence_changed: dict[int, bool], random_changed: dict[int, list[bool]]) -> dict:
    paired = [
        float(evidence_changed[index]) - sum(random_changed[index]) / len(random_changed[index])
        for index in evidence_changed
        if index in random_changed and random_changed[index]
    ]
    return {
        "n": len(paired),
        "gap": sum(paired) / len(paired) if paired else None,
        "paired_differences": paired,
    }


def build_faithfulness_variants(
    rows: list[dict],
    coalition_scores: dict[int, list[float]],
    occlusion_scores: dict[int, list[float]],
    rationale_indices: dict[int, list[int]],
    seed: int,
) -> list[dict]:
    variants = []
    for row in rows:
        source = row["index"]
        n_tokens = len(tokenize(row["sentence"]))
        evidence_sets = []
        for method, scores in (
            ("coalition", coalition_scores.get(source)),
            ("occlusion", occlusion_scores.get(source)),
        ):
            if scores is None:
                continue
            for size in (1, 2, 3):
                if n_tokens < size + 1:
                    continue
                tie_break = (
                    occlusion_scores.get(source)
                    if method == "coalition" else None
                )
                evidence_sets.append(
                    (f"{method}_top_{size}", sorted(
                        sorted(
                            range(len(scores)),
                            key=lambda i: (
                                -scores[i],
                                -(tie_break[i] if tie_break is not None and i < len(tie_break) else 0.0),
                                i,
                            ),
                        )[:size]
                    ))
                )
        rationale = rationale_indices.get(source, [])
        if rationale:
            evidence_sets.append(("rationale", rationale))
        for size in (1, 2, 3):
            if n_tokens < size + 1:
                continue
            for draw, selected in enumerate(
                random_evidence(row, size, draws=5, seed=seed + source * 101 + size)
            ):
                evidence_sets.append((f"random_{size}_{draw}", selected))
            evidence_sets.append(
                (f"last_{size}", list(range(n_tokens - size, n_tokens)))
            )
        for kind, selected in evidence_sets:
            remove, keep = evidence_variants(row, selected, kind)
            if remove is not None and remove["text"]:
                variants.append(remove)
            if keep is not None and keep["text"]:
                variants.append(keep)
    return variants
