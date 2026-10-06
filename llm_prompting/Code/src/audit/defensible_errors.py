"""Are the model's errors actually errors?

If the labels are applied inconsistently, some of what counts as a model error is the
model giving a defensible reading that the annotation disagrees with. This measures
how much of the reported error is of that kind.

Two views, one of which needs no sentiment lexicon at all:

  cue-based       a row's definition-implied label comes from its sentiment cues
  marker-based    a row containing kintu/but/bt is explicitly contrastive, which the
                  Mixed definition says should make it Mixed

Run:  python -m src.audit.defensible_errors
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.audit.definition_compliance import (
    CONTRAST_STRICT,
    cue_profile,
    rule_predictions,
)
from src.common.config import LABEL_MAPPING, prompting_dir
from src.common.data import load_bnsentmix, scope_indices
from src.common.io_utils import load_jsonl

# these audits analyse the BnSentMix Mixed label specifically
DATASET = "bnsentmix"
PREDICTIONS = prompting_dir(DATASET, "predictions")

# the strongest run on the held-out split
RUN = ("groq", "qwen/qwen3.8-27b", "zero_shot_v1")


def load_run(provider, model, prompt, scope):
    safe = model.replace("/", "_").replace(":", "_")
    path = PREDICTIONS / f"{provider}_{safe}_sentiment_{prompt}__{scope}.jsonl"
    return {record["index"]: record["prediction"] for record in load_jsonl(path)}


def main():
    dataset = load_bnsentmix()
    sentences = list(dataset["Sentence"])
    gold = list(dataset["Label"])

    test_rows = sorted(scope_indices("test", dataset))
    provider, model, prompt = RUN
    predictions = load_run(provider, model, prompt, "test")

    rows = [i for i in test_rows if predictions.get(i) is not None]
    implied = rule_predictions([sentences[i] for i in rows])

    correct = [predictions[i] == gold[i] for i in rows]
    accuracy = sum(correct) / len(rows)

    print("=" * 92)
    print(f"ARE THE ERRORS ACTUALLY ERRORS?  ({provider} / {model} / {prompt}, test split)")
    print("=" * 92)
    print(f"  rows: {len(rows)}    raw accuracy: {accuracy:.4f}")

    print()
    print("=" * 92)
    print("1. CUE-BASED - does the model's answer follow the stated definitions?")
    print("=" * 92)
    print("   CAVEAT: the cue rule is a lexicon proxy for 'expresses sentiment', and its")
    print("   coverage is partial, so 'gold contradicts the definition' below includes")
    print("   rows where the lexicon simply missed a cue. Read it as an upper bound.")
    print("   Section 2 is lexicon-free and is the stronger evidence.")
    print()

    consistent = [i for position, i in enumerate(rows) if implied[position] == gold[i]]
    violating = [i for position, i in enumerate(rows) if implied[position] != gold[i]]

    def acc(selection):
        if not selection:
            return float("nan")
        return sum(1 for i in selection if predictions[i] == gold[i]) / len(selection)

    print(f"  {'row type':<34}{'n':>7}{'model acc':>11}")
    print("  " + "-" * 52)
    print(f"  {'gold agrees with the definition':<34}{len(consistent):>7}{acc(consistent):>11.4f}")
    print(f"  {'gold contradicts the definition':<34}{len(violating):>7}{acc(violating):>11.4f}")

    print()
    print("  Among the model's errors, how many are it following the definition")
    print("  while gold contradicts it? Those are defensible, not wrong.")
    errors = [i for i in rows if predictions[i] != gold[i]]
    position_of = {index: position for position, index in enumerate(rows)}
    defensible = [
        i for i in errors
        if predictions[i] == implied[position_of[i]] and implied[position_of[i]] != gold[i]
    ]
    print(f"    total errors                       {len(errors)}")
    print(f"    of which definition-defensible     {len(defensible)}"
          f"  ({len(defensible) / len(errors):.1%} of errors)")
    adjusted = (len(rows) - len(errors) + len(defensible)) / len(rows)
    print(f"    raw accuracy                       {accuracy:.4f}")
    print(f"    counting defensible errors as correct      {adjusted:.4f}")

    print()
    print("=" * 92)
    print("2. MARKER-BASED - no sentiment lexicon needed")
    print("=" * 92)
    marker_rows = [i for i in rows if CONTRAST_STRICT.search(sentences[i])]
    print(f"  test rows containing an explicit contrast marker: {len(marker_rows)}")
    gold_mixed = sum(1 for i in marker_rows if gold[i] == 3)
    model_mixed = sum(1 for i in marker_rows if predictions[i] == 3)
    print(f"    gold labels them Mixed      {gold_mixed:>5}  ({gold_mixed / len(marker_rows):.1%})")
    print(f"    model labels them Mixed     {model_mixed:>5}  ({model_mixed / len(marker_rows):.1%})")
    print()
    print("  On these explicitly contrastive rows, a Mixed prediction is defensible")
    print("  under the dataset's own definition. Where the model says Mixed and gold")
    print("  does not, the disagreement is about the annotation rule, not the text.")
    disputed = [i for i in marker_rows if predictions[i] == 3 and gold[i] != 3]
    print(f"    rows the model calls Mixed where gold disagrees: {len(disputed)}")
    counts = Counter(LABEL_MAPPING[gold[i]] for i in disputed)
    for value, name in LABEL_MAPPING.items():
        if value == 3:
            continue
        share = counts.get(name, 0) / len(disputed) if disputed else 0
        print(f"      gold says {name:<10}{counts.get(name, 0):>5}  {share:>6.1%}")

    print()
    print("=" * 92)
    print("3. WHERE THE MODEL AND GOLD DISAGREE ON Mixed, WHAT IS THE TEXT LIKE?")
    print("=" * 92)
    false_negative = [i for i in rows if gold[i] == 3 and predictions[i] != 3]
    false_positive = [i for i in rows if gold[i] != 3 and predictions[i] == 3]
    for name, group in (("gold Mixed, model missed", false_negative),
                        ("model said Mixed, gold did not", false_positive)):
        if not group:
            continue
        with_marker = sum(1 for i in group if CONTRAST_STRICT.search(sentences[i]))
        print(f"  {name:<34} n={len(group):<5} contains a contrast marker: "
              f"{with_marker / len(group):.1%}")

    payload = {
        "experiment": "defensible_errors",
        "run": {"provider": provider, "model": model, "prompt": prompt, "scope": "test"},
        "rows": len(rows),
        "raw_accuracy": round(accuracy, 4),
        "accuracy_gold_agrees_with_definition": round(acc(consistent), 4),
        "accuracy_gold_contradicts_definition": round(acc(violating), 4),
        "rows_definition_consistent": len(consistent),
        "rows_definition_violating": len(violating),
        "errors": len(errors),
        "defensible_errors": len(defensible),
        "defensible_share_of_errors": round(len(defensible) / len(errors), 4),
        "accuracy_if_defensible_counted_correct": round(adjusted, 4),
        "marker_rows": len(marker_rows),
        "marker_gold_mixed": gold_mixed,
        "marker_model_mixed": model_mixed,
        "marker_disputed": len(disputed),
    }
    out_dir = prompting_dir(DATASET, "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "defensible_errors.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print()
    print("saved:", out_path)


if __name__ == "__main__":
    main()
