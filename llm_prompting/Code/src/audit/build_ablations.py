"""Generate single-element ablations of the elaborate prompt.

If models emit a label the corpus does not contain, the explanation should be findable
in the prompt: one of its rules either causes the behaviour or suppresses it. So each
variant removes exactly one element and changes nothing else, and the script proves
that by diffing against the original and reporting the diff.

Removing one numbered rule leaves the numbering intact except for the gap, which is
deliberate: renumbering would be a second change in the same variant.
"""

import sys
from collections import Counter
from pathlib import Path

PROJECT = Path(r"D:\Research\Sentiment-Analysis")
PROMPTS = PROJECT / "prompts" / "prompting"
SOURCE = PROMPTS / "zero_shot_v1.txt"

# element name -> the exact single line to delete, or the 2-line rule to delete
ABLATIONS = {
    # the Mixed label's own definition, line 8 of the source
    "drop_mixed_def": [8],
    "drop_rule1": [11],
    "drop_rule2": [12, 13],
    "drop_rule3": [14],
    "drop_rule4": [15, 16],
    "drop_rule5": [17],
    "drop_rule6": [18],
    "drop_rule7": [19],
}


def main():
    original = SOURCE.read_text(encoding="utf-8")
    lines = original.splitlines(keepends=True)
    print(f"source: {SOURCE.name}  ({len(lines)} lines)")
    print()

    made = []
    for name, numbers in ABLATIONS.items():
        removed = [lines[number - 1].rstrip("\n") for number in numbers]
        kept = [line for position, line in enumerate(lines, start=1)
                if position not in numbers]
        variant = "".join(kept)

        target = PROMPTS / f"zero_shot_ablate_{name}_v1.txt"
        target.write_text(variant, encoding="utf-8")

        # independent check by line multiset: the variant must be missing exactly the
        # target lines and must not have introduced or altered any line of its own
        before = Counter(line.rstrip("\n") for line in lines)
        after = Counter(line.rstrip("\n") for line in variant.splitlines())
        gone = before - after      # in the source, absent from the variant
        added = after - before     # the variant introduced these
        ok = gone == Counter(text for text in removed if text.strip()) and not added
        made.append((name, target.name, removed, ok))

    print(f"  {'variant':<20}{'removed':<70}{'exact'}")
    print("  " + "-" * 100)
    for name, filename, removed, ok in made:
        text = " | ".join(removed).strip()
        if len(text) > 66:
            text = text[:63] + "..."
        print(f"  {name:<20}{text:<70}{'yes' if ok else 'NO'}")
    print()
    print(f"{sum(1 for _, _, _, ok in made if ok)}/{len(made)} ablations remove exactly "
          "the intended text and nothing else")

    # lengths, as a coarse sanity check that each variant is smaller than the source
    print()
    print(f"  source length: {len(original)} chars")
    for name, filename, _, _ in made:
        variant = (PROMPTS / filename).read_text(encoding="utf-8")
        print(f"  {name:<20}{len(variant):>6} chars   (-{len(original) - len(variant)})")


if __name__ == "__main__":
    main()
