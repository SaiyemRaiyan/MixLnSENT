import re

from datasets import load_dataset
from sklearn.model_selection import train_test_split

from .config import DEV_SIZE, SCOPES, SEED


def load_bnsentmix():
    return load_dataset("aplycaebous/BnSentMix", split="train")


def split_indices(dataset):
    """Return {'train': [...], 'validation': [...], 'test': [...]} as row positions."""
    row_ids = list(range(len(dataset)))
    labels = dataset["Label"]
    train_ids, temporary_ids = train_test_split(
        row_ids,
        test_size=0.30,
        random_state=SEED,
        stratify=labels,
    )
    temporary_labels = [labels[index] for index in temporary_ids]
    validation_ids, test_ids = train_test_split(
        temporary_ids,
        test_size=0.50,
        random_state=SEED,
        stratify=temporary_labels,
    )
    return {
        "train": train_ids,
        "validation": validation_ids,
        "test": test_ids,
    }


def create_fixed_splits(dataset):
    indices = split_indices(dataset)
    return {
        name: dataset.select(index_list)
        for name, index_list in indices.items()
    }


def dev_indices(dataset):
    """Stratified sample drawn only from the train split.

    Prompts are developed against this set so that validation and test are never
    looked at while tuning; the reported test number therefore stays held out.
    """
    train_ids = split_indices(dataset)["train"]
    labels = [dataset[index]["Label"] for index in train_ids]
    sample_ids, _ = train_test_split(
        train_ids,
        train_size=DEV_SIZE,
        random_state=SEED,
        stratify=labels,
    )
    return sorted(sample_ids)


def scope_indices(scope, dataset):
    """Row positions to classify for a named scope."""
    if scope == "dev":
        return dev_indices(dataset)
    if scope == "full":
        return list(range(len(dataset)))
    if scope in ("train", "validation", "test"):
        return sorted(split_indices(dataset)[scope])
    raise ValueError(f"Unknown scope {scope!r}. Known: {', '.join(SCOPES)}")


def label_counts(data):
    return {
        label: data["Label"].count(label)
        for label in sorted(set(data["Label"]))
    }


def normalize_sentence(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def duplicate_summary(dataset):
    groups = {}
    for index, (sentence, label) in enumerate(
        zip(dataset["Sentence"], dataset["Label"])
    ):
        groups.setdefault(normalize_sentence(sentence), []).append((index, label))

    duplicate_groups = {key: members for key, members in groups.items() if len(members) > 1}
    conflicts = {
        key: members
        for key, members in duplicate_groups.items()
        if len({label for _, label in members}) > 1
    }
    return {
        "rows": len(dataset),
        "unique_normalized": len(groups),
        "duplicate_groups": len(duplicate_groups),
        "rows_in_duplicate_groups": sum(len(members) for members in duplicate_groups.values()),
        "conflict_groups": len(conflicts),
        "rows_in_conflict_groups": sum(len(members) for members in conflicts.values()),
    }


def leakage_free_indices(eval_indices, train_indices, dataset):
    """Drop evaluation rows whose normalized text also occurs in the training rows."""
    train_keys = {normalize_sentence(dataset[index]["Sentence"]) for index in train_indices}
    return [
        index
        for index in eval_indices
        if normalize_sentence(dataset[index]["Sentence"]) not in train_keys
    ]


def leakage_free_subset(eval_data, train_data):
    train_keys = {normalize_sentence(sentence) for sentence in train_data["Sentence"]}
    keep = [
        index
        for index, sentence in enumerate(eval_data["Sentence"])
        if normalize_sentence(sentence) not in train_keys
    ]
    stats = {
        "rows_in": len(eval_data),
        "rows_out": len(keep),
        "rows_removed": len(eval_data) - len(keep),
    }
    return eval_data.select(keep), stats
