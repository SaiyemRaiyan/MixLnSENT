"""EXPERIMENT 4 - MIXSARC MULTI-LABEL PROMPTING.

Implicit-meaning identification on Bangla-English code-mixed text: four independent
binary labels per sentence rather than one categorical label. A different task needs a
different instrument, so this module carries its own prompt builder, parser and
scorer; the runner dispatches on the dataset's task rather than assuming sentiment.

The prompt text is the one published with the MixSarc corpus, kept verbatim apart from
the packing: their prompt labels one sentence per request, while the batch format here
numbers the sentences so a run costs hundreds of requests rather than thousands. The
label definitions and the JSON contract are unchanged, so an answer means the same
thing; only how many sentences travel together differs.
"""

import json
import re
from pathlib import Path

from ..common.config import PROJECT_ROOT

MIXSARC_PROMPTS_DIR = PROJECT_ROOT / "prompts" / "mixsarc"
DEFAULT_PROMPT_NAME = "zero_shot_v1"
BATCH_FORMAT_PATH = MIXSARC_PROMPTS_DIR / "batch_format_v1.txt"

LABELS = ("Humorous", "Sarcastic", "Offensive", "Vulgar")
LABEL_INDEX = {name: position for position, name in enumerate(LABELS)}

JSON_OBJECT = re.compile(r"\{[^{}]*\}")
INDEX_FIELD = re.compile(r'"?index"?\s*:\s*(\d+)', re.IGNORECASE)


def prompt_path(name=None):
    return MIXSARC_PROMPTS_DIR / f"{name or DEFAULT_PROMPT_NAME}.txt"


def load_prompt(name=None):
    path = prompt_path(name)
    if not path.exists():
        available = sorted(p.name for p in path.parent.glob("*.txt"))
        raise FileNotFoundError(
            f"{path.name} not found. Available prompts: {', '.join(available)}"
        )
    return path.read_text(encoding="utf-8")


def prompt_label(name=None):
    return f"mixsarc_{name or DEFAULT_PROMPT_NAME}"


def build_batch_prompt(texts, name=None):
    template = load_prompt(name)
    batch_format = BATCH_FORMAT_PATH.read_text(encoding="utf-8")
    count_instruction = (
        f"There are exactly {len(texts)} sentences. "
        f"Return exactly {len(texts)} JSON lines.\n"
    )
    items = "\n".join(
        f"{index}: {text}" for index, text in enumerate(texts, start=1)
    )
    return template + count_instruction + "Sentences:\n" + items + "\n" + batch_format


def _vector_from_object(payload):
    """Turn one parsed JSON object into a 4-tuple, or None if a label is missing."""
    values = []
    for label in LABELS:
        if label in payload:
            raw = payload[label]
        else:
            match = next(
                (value for key, value in payload.items()
                 if key.lower() == label.lower()),
                None,
            )
            raw = match
        if raw is None:
            return None
        values.append(1 if str(raw).strip() in ("1", "true", "True", "yes") else 0)
    return tuple(values)


def parse_batch(response_text, expected_count):
    """Return (vectors, missing_indices). vectors[i] is a 4-tuple or None.

    Accepts the documented one-object-per-line shape, and also a single JSON array or
    an object keyed by sentence number, because models drift between those on their
    own. Anything unparseable is left as None and retried rather than guessed.
    """
    text = response_text or ""
    found = {}

    # a whole-response JSON payload first, so an array reply is not mis-read as lines
    stripped = text.strip()
    if stripped.startswith("[") or stripped.startswith("{"):
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, list):
            for position, item in enumerate(payload, start=1):
                if isinstance(item, dict):
                    vector = _vector_from_object(item)
                    if vector is not None:
                        found[position] = vector
        elif isinstance(payload, dict):
            for key, item in payload.items():
                if isinstance(item, dict):
                    vector = _vector_from_object(item)
                    if vector is not None and str(key).isdigit():
                        found[int(key)] = vector

    if not found:
        for line in text.splitlines():
            for blob in JSON_OBJECT.findall(line):
                try:
                    payload = json.loads(blob)
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, dict):
                    continue
                vector = _vector_from_object(payload)
                if vector is None:
                    continue
                index_match = INDEX_FIELD.search(blob)
                if index_match is None:
                    # no index: fall back to the order the objects appeared
                    next_index = max(found) + 1 if found else 1
                    found[next_index] = vector
                else:
                    found[int(index_match.group(1))] = vector

    vectors = [found.get(index) for index in range(1, expected_count + 1)]
    missing = [i for i, v in enumerate(vectors, start=1) if v is None]
    return vectors, missing


def exact_match_accuracy(gold, predicted):
    """Share of sentences where all four labels are right, as the paper reports."""
    correct = sum(1 for g, p in zip(gold, predicted) if tuple(g) == tuple(p))
    return correct / len(gold) if gold else float("nan")


def per_label_scores(gold, predicted):
    """Precision, recall and F1 for each of the four labels."""
    scores = {}
    for position, label in enumerate(LABELS):
        true_positive = false_positive = false_negative = 0
        for g, p in zip(gold, predicted):
            actual, guess = g[position], p[position]
            if guess == 1 and actual == 1:
                true_positive += 1
            elif guess == 1 and actual == 0:
                false_positive += 1
            elif guess == 0 and actual == 1:
                false_negative += 1
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
        scores[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "positives": true_positive + false_negative,
            "predicted": true_positive + false_positive,
        }
    return scores


def multilabel_metrics(gold, predicted):
    """Exact-match accuracy plus macro-averaged precision, recall and F1."""
    per_label = per_label_scores(gold, predicted)
    macro = {
        key: sum(scores[key] for scores in per_label.values()) / len(LABELS)
        for key in ("precision", "recall", "f1")
    }
    return {
        "exact_match_accuracy": exact_match_accuracy(gold, predicted),
        "macro": macro,
        "per_label": per_label,
        "rows": len(gold),
    }
