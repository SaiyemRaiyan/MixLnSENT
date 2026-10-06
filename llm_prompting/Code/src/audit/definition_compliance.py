"""Definition-compliance audit.

Every label in BnSentMix comes with a stated definition. This module writes a rule
that implements each definition, then measures how well that rule reproduces the
gold labels. A label whose definition-rule scores well is definition-compliant: the
annotators were applying the definition they wrote down. A label whose rule scores
poorly is not.

Two marker families are used, because they detect different sub-cases of a
contrastive sentence:

  sentiment cues   an opposing pair of positive and negative words
  contrast markers explicit connectives such as kintu, but, bt, tobe

No model is involved, so this cannot leak from the evaluation sets and it reproduces
from code alone.

Run:  python -m src.audit.definition_compliance
"""

import json
import re
import sys
from collections import Counter
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# this module analyses a specific dataset
DATASET = "bnsentmix"

from src.common.config import LABEL_MAPPING, prompting_dir
from src.common.data import load_bnsentmix

# ---------------------------------------------------------------------------
# Lexicons. Deliberately high-confidence: only words whose sentiment is
# unambiguous in Banglish. A loose set was tried first and gave the same
# direction, which is recorded in the results.
# ---------------------------------------------------------------------------

POSITIVE_CUES = set("""
bhalo valo vlo balo sundor shundor darun ostad osadharon khusi khushi anondo
moja mast mojah best excellent awesome superb great nice good perfect
love loved proud dhonnobad thanks thank sohoj friendly recommend satisfied
""".split())

NEGATIVE_CUES = set("""
kharap karap baje joghonno faltu bakwas bokwas nosto boring borring waste
dukkho dukkhito kosto khoti loss slow late delay problem jhamela
fail failed broken damage damaged spoiled fake worst bad terrible awful
hate hated dislike angry raag bekar useless worthless
""".split())

CONTRAST_LOOSE = re.compile(
    r"\b(kintu|kinto|kindo|kin2|kinthu|but|bt|butt|however|though|although|"
    r"tobe|tabe|jodio|ar|o)\b",
    re.IGNORECASE,
)

CONTRAST_STRICT = re.compile(
    r"\b(kintu|kinto|kindo|kin2|kinthu|but|bt|butt|however|though|although|tobe|tabe|jodio)\b",
    re.IGNORECASE,
)

TOKEN = re.compile(r"[a-zA-Z]+")


def cue_profile(text):
    """Return (has_positive_cue, has_negative_cue, has_contrast_strict, has_contrast_loose)."""
    tokens = {token.lower() for token in TOKEN.findall(text)}
    return (
        bool(tokens & POSITIVE_CUES),
        bool(tokens & NEGATIVE_CUES),
        bool(CONTRAST_STRICT.search(text)),
        bool(CONTRAST_LOOSE.search(text)),
    )


def rule_predictions(sentences):
    """Label every row purely from the stated definitions.

    Positive  a positive cue and no negative cue
    Negative  a negative cue and no positive cue
    Mixed     both cues
    Neutral   neither cue
    """
    predicted = []
    for sentence in sentences:
        positive, negative, _, _ = cue_profile(sentence)
        if positive and negative:
            predicted.append(3)
        elif positive:
            predicted.append(0)
        elif negative:
            predicted.append(1)
        else:
            predicted.append(2)
    return predicted


def prf(gold, predicted, label_value):
    true_positive = sum(
        1 for g, p in zip(gold, predicted) if g == label_value and p == label_value
    )
    false_positive = sum(
        1 for g, p in zip(gold, predicted) if g != label_value and p == label_value
    )
    false_negative = sum(
        1 for g, p in zip(gold, predicted) if g == label_value and p != label_value
    )
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "support": sum(1 for g in gold if g == label_value),
        "fired": true_positive + false_positive,
    }


def audit(sentences, gold):
    predicted = rule_predictions(sentences)
    per_label = {
        LABEL_MAPPING[value]: prf(gold, predicted, value) for value in LABEL_MAPPING
    }
    accuracy = sum(1 for g, p in zip(gold, predicted) if g == p) / len(gold)
    return {
        "rows": len(gold),
        "rule_accuracy": round(accuracy, 4),
        "per_label": per_label,
    }


def marker_audit(sentences, gold):
    """How well does each marker family predict the Mixed label on its own?"""
    results = {}
    for name, detector in (
        ("contrast_marker_strict",
         lambda s: cue_profile(s)[2]),
        ("contrast_marker_loose",
         lambda s: cue_profile(s)[3]),
        ("opposing_sentiment_cues",
         lambda s: cue_profile(s)[0] and cue_profile(s)[1]),
    ):
        fired = [detector(sentence) for sentence in sentences]
        predicted = [3 if flag else 0 for flag in fired]  # 0 is a placeholder non-Mixed
        # precision/recall of the marker as a Mixed detector
        hits = sum(1 for flag, g in zip(fired, gold) if flag and g == 3)
        precision = hits / sum(fired) if sum(fired) else 0.0
        recall = hits / sum(1 for g in gold if g == 3)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        results[name] = {
            "fires": sum(fired),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
    return results


def main():
    dataset = load_bnsentmix()
    sentences = list(dataset["Sentence"])
    gold = list(dataset["Label"])

    print("=" * 88)
    print("DEFINITION-COMPLIANCE AUDIT - can each label's stated definition reproduce it?")
    print("=" * 88)
    print("  Every rule below implements the definition the dataset states for its label.")
    print("  No model is involved, so nothing here can leak from an evaluation split.")
    print()

    overall = audit(sentences, gold)
    print(f"  rows: {overall['rows']}    the four rules together agree with gold "
          f"{overall['rule_accuracy']:.4f} of the time")
    print()
    print(f"  {'label':<12}{'precision':>11}{'recall':>9}{'f1':>9}{'fired':>9}{'gold n':>9}")
    print("  " + "-" * 57)
    for value, name in LABEL_MAPPING.items():
        stats = overall["per_label"][name]
        print(f"  {name:<12}{stats['precision']:>11.4f}{stats['recall']:>9.4f}"
              f"{stats['f1']:>9.4f}{stats['fired']:>9}{stats['support']:>9}")

    print()
    print("  Same audit restricted to the held-out test split:")
    from src.common.data import scope_indices

    test_rows = scope_indices("test", dataset)
    test_audit = audit([sentences[i] for i in test_rows], [gold[i] for i in test_rows])
    print(f"  {'label':<12}{'precision':>11}{'recall':>9}{'f1':>9}")
    print("  " + "-" * 42)
    for value, name in LABEL_MAPPING.items():
        stats = test_audit["per_label"][name]
        print(f"  {name:<12}{stats['precision']:>11.4f}{stats['recall']:>9.4f}{stats['f1']:>9.4f}")

    print()
    print("=" * 88)
    print("WHICH MARKER BEST PREDICTS Mixed?")
    print("=" * 88)
    markers = marker_audit(sentences, gold)
    print(f"  {'marker':<28}{'fires':>8}{'precision':>11}{'recall':>9}{'f1':>8}")
    print("  " + "-" * 64)
    for name, stats in markers.items():
        print(f"  {name:<28}{stats['fires']:>8}{stats['precision']:>11.4f}"
              f"{stats['recall']:>9.4f}{stats['f1']:>8.4f}")

    print()
    print("  Rows matched by exactly one marker family:")
    strict = {i for i, s in enumerate(sentences) if cue_profile(s)[2]}
    opposing = {i for i, s in enumerate(sentences) if cue_profile(s)[0] and cue_profile(s)[1]}
    print(f"    contrast word only    {len(strict - opposing)}")
    print(f"    opposing cues only    {len(opposing - strict)}")
    print(f"    both agree            {len(strict & opposing)}")

    payload = {
        "experiment": "definition_compliance_audit",
        "rows": overall["rows"],
        "rule_accuracy_all": overall["rule_accuracy"],
        "per_label_all": overall["per_label"],
        "per_label_test": test_audit["per_label"],
        "markers": markers,
        "marker_overlap": {
            "contrast_only": len(strict - opposing),
            "opposing_only": len(opposing - strict),
            "both": len(strict & opposing),
        },
    }
    out_dir = prompting_dir(DATASET, "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "definition_compliance_audit.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print()
    print("saved:", out_path)


if __name__ == "__main__":
    main()
