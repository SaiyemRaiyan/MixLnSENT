"""EXPERIMENT 2 - SENTIMENT PROMPTING.

Zero/few-shot classification of sentence sentiment with no training. The same
sentences used by Experiment 1 are scored on the same fixed evaluation splits.
"""

import re
from pathlib import Path

from ..common.config import (
    LABEL_MAPPING,
    PROMPTING_PROMPTS_DIR,
    prompting_dir,
)

DEFAULT_PROMPT_NAME = "zero_shot_v1"
BATCH_FORMAT_PATH = PROMPTING_PROMPTS_DIR / "batch_format_v1.txt"
LABEL_BY_NAME = {name.lower(): value for value, name in LABEL_MAPPING.items()}
ITEM_LINE = re.compile(r"^\s*(\d+)\s*[:.)\-\u2013]\s*(.+?)\s*$")


def sentiment_prompt_path(name=None):
    name = name or DEFAULT_PROMPT_NAME
    return PROMPTING_PROMPTS_DIR / f"{name}.txt"


def load_sentiment_prompt(name=None):
    path = sentiment_prompt_path(name)
    if not path.exists():
        available = sorted(p.name for p in path.parent.glob("*.txt"))
        raise FileNotFoundError(
            f"{path.name} not found. Available prompts: {', '.join(available)}"
        )
    return path.read_text(encoding="utf-8")


def prompt_label(name=None):
    return f"sentiment_{name or DEFAULT_PROMPT_NAME}"


def build_sentiment_prompt(text, name=None):
    return load_sentiment_prompt(name).format(sentence=text)


def build_sentiment_batch_prompt(texts, name=None):
    template = load_sentiment_prompt(name)
    batch_format = BATCH_FORMAT_PATH.read_text(encoding="utf-8")
    body = template.replace("Text:\n{sentence}", "Texts:\n")
    count_instruction = (
        f"There are exactly {len(texts)} texts. "
        f"Return exactly {len(texts)} decision lines.\n"
    )
    items = "\n".join(
        f"{index}: {text}" for index, text in enumerate(texts, start=1)
    )
    return body + count_instruction + items + "\n" + batch_format


def parse_sentiment_batch(response_text, expected_count):
    """Return (labels, missing_indices). labels[i] is an int label or None.

    Accepts `1: Positive`, `1. Positive`, `1) Positive`. Lines whose label is not
    one of the four known names are ignored, so prose is skipped safely.
    """
    found = {}
    for line in response_text.strip().splitlines():
        match = ITEM_LINE.match(line)
        if not match:
            continue
        index_text, label_text = match.group(1), match.group(2)
        label_key = re.sub(r"[^a-z]", "", label_text.lower())
        if label_key not in LABEL_BY_NAME:
            continue
        found[int(index_text)] = LABEL_BY_NAME[label_key]

    labels = [found.get(index) for index in range(1, expected_count + 1)]
    missing = [index for index, label in enumerate(labels, start=1) if label is None]
    return labels, missing


def predictions_path(provider, model, prompt_name=None, scope="full", dataset="bnsentmix"):
    safe_model = model.replace("/", "_").replace(":", "_")
    return (
        prompting_dir(dataset, "predictions")
        / f"{provider}_{safe_model}_{prompt_label(prompt_name)}__{scope}.jsonl"
    )


TASK_PREFIXES = ("sentiment_", "mixsarc_")


def parse_predictions_name(path):
    """Recover (provider, model, prompt_name, scope) from a prediction file name.

    The task prefix is stripped along with the record, because a prompt name is only
    meaningful within its task: zero_shot_v1 exists for both and they are different
    files.
    """
    stem = Path(path).stem
    body, _, scope = stem.rpartition("__")
    provider, _, remainder = body.partition("_")
    for prefix in TASK_PREFIXES:
        prompt_start = remainder.find(f"_{prefix}")
        if prompt_start != -1:
            safe_model = remainder[:prompt_start]
            return provider, safe_model, remainder[prompt_start + len(prefix) + 1:], scope
    return provider, remainder, DEFAULT_PROMPT_NAME, scope
