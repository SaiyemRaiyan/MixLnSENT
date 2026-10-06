"""Dataset registry.

The pipeline was written against BnSentMix and assumed its shape: a Hugging Face
dataset object, a "Sentence"/"Label" schema, four labels, and a fixed 70:15:15 split.
SentMix-3L differs on every one of those, so the differences are described here and
the runner consumes this interface rather than the raw dataset.

Descriptor fields:
  name           used in output paths, so runs on different datasets cannot collide
  texts          list[str], one per row
  gold           list[int], labels as integers
  label_mapping  {int: str} for display and parsing
  indices(scope) list[int] of row positions to evaluate for a named scope

BnSentMix label integers are the existing convention:
    0 Positive, 1 Negative, 2 Neutral, 3 Mixed
SentMix-3L reuses it and simply never emits 3, so metrics code is shared.
"""

import csv
import io
from pathlib import Path

from .config import PROJECT_ROOT
from .data import load_bnsentmix, scope_indices

SENTMIX3L_CSV = PROJECT_ROOT / "data" / "sentmix3l_sen_1k.csv"
SENTMIX3L_URL = (
    "https://raw.githubusercontent.com/LanguageTechnologyLab/SentMix-3L/main/sen_1k.csv"
)

SENTMIX3L_LABEL_MAPPING = {0: "Positive", 1: "Negative", 2: "Neutral"}
SENTMIX3L_LABEL_BY_NAME = {
    name: value for value, name in SENTMIX3L_LABEL_MAPPING.items()
}

BNSENTMIX_LABEL_MAPPING = {0: "Positive", 1: "Negative", 2: "Neutral", 3: "Mixed"}


class Dataset:
    """Everything a run needs to know about a dataset.

    `task` selects which prompt builder, parser and scorer the runner uses. Datasets
    that carry one categorical label per row are "sentiment"; datasets with several
    independent binary labels per row are "multilabel". The two cannot share a
    scorer, so the runner dispatches on this rather than assuming.
    """

    def __init__(self, name, texts, gold, label_mapping, scope_resolver, task="sentiment"):
        self.name = name
        self.texts = texts
        self.gold = gold
        self.label_mapping = label_mapping
        self.task = task
        self._scope_resolver = scope_resolver

    def __len__(self):
        return len(self.texts)

    def indices(self, scope):
        return self._scope_resolver(scope)

    def label_names(self):
        return [self.label_mapping[value] for value in sorted(self.label_mapping)]


def _bnsentmix(scope_resolver=None):
    dataset = load_bnsentmix()
    texts = list(dataset["Sentence"])
    gold = list(dataset["Label"])

    def resolve(scope):
        if scope_resolver is not None:
            return scope_resolver(scope)
        return scope_indices(scope, dataset)

    return Dataset("bnsentmix", texts, gold, BNSENTMIX_LABEL_MAPPING, resolve)


def _sentmix3l():
    if not SENTMIX3L_CSV.exists():
        raise FileNotFoundError(
            f"{SENTMIX3L_CSV} is missing. Fetch it with:\n"
            f"  python -m src.common.fetch_data\n"
            f"(source: {SENTMIX3L_URL})"
        )
    rows = list(csv.DictReader(io.StringIO(SENTMIX3L_CSV.read_text(encoding="utf-8"))))
    texts = [row["text"] for row in rows]
    unknown = sorted({row["label"].strip() for row in rows} - set(SENTMIX3L_LABEL_BY_NAME))
    if unknown:
        raise ValueError(f"unexpected labels in {SENTMIX3L_CSV.name}: {unknown}")
    gold = [SENTMIX3L_LABEL_BY_NAME[row["label"].strip()] for row in rows]

    # the dataset is test-only: one scope, covering every row
    def resolve(scope):
        if scope not in ("full", "test"):
            raise ValueError(
                f"sentmix3l has no '{scope}' split; it is test-only. Use 'full'."
            )
        return list(range(len(texts)))

    return Dataset("sentmix3l", texts, gold, SENTMIX3L_LABEL_MAPPING, resolve)


MIXSARC_LABELS = ("Humorous", "Sarcastic", "Offensive", "Vulgar")
# the released corpus names its columns in lowercase; the labels are reported capitalised
MIXSARC_COLUMNS = ("humorous", "sarcastic", "offensive", "vulgar")
MIXSARC_SPLIT_SEED = 42


def _mixsarc():
    """MixSarc: four independent binary labels per row, so the task is multilabel.

    The corpus is published as a single 9,087-row train split. The authors report a
    70:15:15 partition stratified on the humor label but do not release it or its
    seed, so the split here is rebuilt the same way under a fixed seed. It is
    therefore the same construction but not necessarily the same rows as theirs, and
    comparisons against their published test numbers carry that caveat.
    """
    import random

    from datasets import load_dataset

    hf = load_dataset("ajwad-abrar/MixSarc", split="train")
    texts = list(hf["sentence"])
    gold = [
        tuple(1 if row[column] == 1 else 0 for column in MIXSARC_COLUMNS)
        for row in hf
    ]

    # stratified on the humorous label, matching the published construction
    by_stratum = {}
    for index, row in enumerate(hf):
        by_stratum.setdefault(row["humorous"], []).append(index)

    rng = random.Random(MIXSARC_SPLIT_SEED)
    train, validation, test = [], [], []
    for stratum in sorted(by_stratum):
        indices = list(by_stratum[stratum])
        rng.shuffle(indices)
        n = len(indices)
        cut_a = int(0.70 * n)
        cut_b = int(0.85 * n)
        train += indices[:cut_a]
        validation += indices[cut_a:cut_b]
        test += indices[cut_b:]

    splits = {
        "train": sorted(train),
        "validation": sorted(validation),
        "test": sorted(test),
        "full": list(range(len(texts))),
    }

    def resolve(scope):
        if scope not in splits:
            raise ValueError(
                f"mixsarc has no '{scope}' split. Use one of: {', '.join(splits)}"
            )
        return splits[scope]

    mapping = {value: name for value, name in enumerate(MIXSARC_LABELS)}
    return Dataset("mixsarc", texts, gold, mapping, resolve, task="multilabel")


LOADERS = {
    "bnsentmix": _bnsentmix,
    "sentmix3l": _sentmix3l,
    "mixsarc": _mixsarc,
}


def load(name):
    if name not in LOADERS:
        raise ValueError(f"Unknown dataset {name!r}. Known: {', '.join(LOADERS)}")
    return LOADERS[name]()
