"""Is gold the outlier?

Two models from different families re-annotate the same rows using nothing but the
definitions BnSentMix states for its labels - no examples, no extra instructions, no
tuning. Then three agreement rates are compared:

    modelA vs modelB     two independent readings of the same definition
    modelA vs gold       each model against the published labels
    modelB vs gold

If the two models agree with each other more than either agrees with gold, then the
gold labels are the inconsistent party rather than the models.

Sampled from the TRAIN split so the test set stays untouched for evaluation.

Run:  python -m src.audit.reannotation
"""

import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# this module analyses a specific dataset
DATASET = "bnsentmix"

from src.common.config import LABEL_MAPPING, prompting_dir, SEED
from src.common.data import load_bnsentmix, split_indices
from src.common.io_utils import append_jsonl, load_jsonl
from src.prompting.runner import classify_batch
from src.prompting.providers import build_sender

PROMPT_NAME = "annotation_definition_only_v1"
SAMPLE_PER_GROUP = 300
BATCH_SIZE = 100

# Three annotators from three different families. gemini-3.6-flash was the intended
# second family but returned 503 UNAVAILABLE throughout, so it is not used.
ANNOTATORS = [
    ("groq", "openai/gpt-oss-20b"),   # OpenAI open-weights
    ("groq", "allam-2-7b"),           # ALLaM, a distinct lineage
    ("groq", "qwen/qwen3.8-27b"),     # Alibaba; also the leaderboard model, noted below
]


def build_sample(dataset):
    """300 gold-Mixed rows plus 300 rows drawn from the remaining classes."""
    train_rows = sorted(split_indices(dataset)["train"])
    rng = random.Random(SEED)

    mixed = [i for i in train_rows if dataset[i]["Label"] == 3]
    other = [i for i in train_rows if dataset[i]["Label"] != 3]

    sample = rng.sample(mixed, min(SAMPLE_PER_GROUP, len(mixed)))
    sample += rng.sample(other, min(SAMPLE_PER_GROUP, len(other)))
    rng.shuffle(sample)
    return sorted(sample)


def annotate(provider, model, sample_indices, dataset, log=print):
    """Annotate the sample with one model, checkpointing after every batch.

    Rows already on disk are skipped, so an interrupted run resumes instead of
    paying for the same requests twice.
    """
    safe_model = model.replace("/", "_").replace(":", "_")
    path = prompting_dir(DATASET, "predictions") / f"reannotation_{safe_model}__train.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)

    done = {record["index"]: record["prediction"] for record in load_jsonl(path)}
    todo = [index for index in sample_indices if done.get(index) is None]

    log(f"  {model:<28} {len(done)} on disk, {len(todo)} to annotate")
    if todo:
        sender = build_sender(provider, model)
        for start in range(0, len(todo), BATCH_SIZE):
            chunk = todo[start:start + BATCH_SIZE]
            texts = [dataset[index]["Sentence"] for index in chunk]
            labels = classify_batch(sender, texts, PROMPT_NAME, log=lambda message: None)
            records = [
                {"index": index, "prediction": labels[position]}
                for position, index in enumerate(chunk)
            ]
            append_jsonl(records, path)
            for record in records:
                done[record["index"]] = record["prediction"]
            log(f"  {model:<28} {min(start + BATCH_SIZE, len(todo))}/{len(todo)}")

    return [done.get(index) for index in sample_indices]


def agreement(first, second):
    pairs = [(a, b) for a, b in zip(first, second) if a is not None and b is not None]
    if not pairs:
        return float("nan"), 0
    return sum(1 for a, b in pairs if a == b) / len(pairs), len(pairs)


def main():
    dataset = load_bnsentmix()
    sample = build_sample(dataset)
    gold = [dataset[i]["Label"] for i in sample]

    print("=" * 90)
    print("DEFINITION-ONLY RE-ANNOTATION - is gold the outlier?")
    print("=" * 90)
    print(f"  prompt: {PROMPT_NAME} (the dataset's own definitions, nothing else)")
    print(f"  sample: {len(sample)} train rows")
    counts = Counter(LABEL_MAPPING[value] for value in gold)
    print(f"  gold composition: " + "  ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    print()

    predictions = {}
    for provider, model in ANNOTATORS:
        print(f"annotating with {model} ({provider})")
        predictions[model] = annotate(provider, model, sample, dataset)
        unparsed = sum(1 for value in predictions[model] if value is None)
        print(f"  done, unparsed: {unparsed}")
        print()

    names = [model for _, model in ANNOTATORS]

    print("=" * 90)
    print("AGREEMENT MATRIX")
    print("=" * 90)
    pairs = {}
    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            value, count = agreement(predictions[names[left]], predictions[names[right]])
            pairs[f"{names[left]} vs {names[right]}"] = round(value, 4)
            print(f"  {names[left]:<26} vs {names[right]:<26} {value:>7.3f}   (n={count})")
    gold_rates = {}
    for model in names:
        value, count = agreement(predictions[model], gold)
        gold_rates[f"{model} vs gold"] = round(value, 4)
        print(f"  {model:<26} vs {'gold':<26} {value:>7.3f}   (n={count})")

    print()
    print("=" * 90)
    print("IS GOLD THE OUTLIER?")
    print("=" * 90)
    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            a, b = names[left], names[right]
            between = agreement(predictions[a], predictions[b])[0]
            to_gold = max(agreement(predictions[a], gold)[0], agreement(predictions[b], gold)[0])
            verdict = "yes" if between > to_gold else "no"
            print(f"  {a} vs {b}:")
            print(f"    agree with each other {between:.3f}   best of the two vs gold {to_gold:.3f}"
                  f"   -> gold is the outlier: {verdict}")

    between_all = max(
        agreement(predictions[names[left]], predictions[names[right]])[0]
        for left in range(len(names))
        for right in range(left + 1, len(names))
    )
    gold_best = max(agreement(predictions[model], gold)[0] for model in names)
    overall_verdict = bool(between_all > gold_best)
    print()
    print(f"  strongest model-pair agreement {between_all:.3f} "
          f"vs best model-vs-gold {gold_best:.3f}   -> {overall_verdict}")

    print()
    print("=" * 90)
    print("ATTRACTION: what do the models call rows gold calls Mixed?")
    print("=" * 90)
    mixed_rows = [index for index, value in enumerate(gold) if value == 3]
    for model in names:
        counts = Counter(
            LABEL_MAPPING[predictions[model][i]]
            for i in mixed_rows
            if predictions[model][i] is not None
        )
        total = sum(counts.values())
        if total == 0:
            print(f"  {model:<30}no usable predictions")
            continue
        dist = "  ".join(
            f"{LABEL_MAPPING[v]} {counts.get(LABEL_MAPPING[v], 0) / total:5.1%}"
            for v in LABEL_MAPPING
        )
        print(f"  {model:<30}{dist}")
    gold_counts = Counter(LABEL_MAPPING[gold[i]] for i in mixed_rows)
    print(f"  {'(gold)':<30}" + "  ".join(
        f"{LABEL_MAPPING[v]} {gold_counts.get(LABEL_MAPPING[v], 0) / len(mixed_rows):5.1%}"
        for v in LABEL_MAPPING
    ))

    payload = {
        "experiment": "definition_only_reannotation",
        "prompt": PROMPT_NAME,
        "sample_rows": len(sample),
        "sample_source": "train split",
        "annotators": names,
        "agreement_between_models": pairs,
        "agreement_with_gold": gold_rates,
        "strongest_model_pair_agreement": round(between_all, 4),
        "best_model_vs_gold": round(gold_best, 4),
        "gold_is_outlier": overall_verdict,
        "predictions": {
            model: [
                {"index": sample[position], "prediction": predictions[model][position]}
                for position in range(len(sample))
            ]
            for model in names
        },
    }
    out_dir = prompting_dir(DATASET, "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "definition_only_reannotation.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print()
    print("saved:", out_path)


if __name__ == "__main__":
    main()
