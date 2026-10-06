"""Task registry: which prompt, parser and scorer a dataset's runs use.

The runner is generic over batching, checkpointing and provider failure handling, and
those parts are identical whether a row carries one label or four. What differs is how
a batch is phrased, how the reply is read back, and how predictions are scored, so
those three are looked up here from the dataset's declared task instead of being
assumed to be sentiment.
"""

from types import SimpleNamespace

from . import multilabel
from .prompts import build_sentiment_batch_prompt, parse_sentiment_batch, prompt_label


def task_for(dataset):
    """Return the build/parse/label bundle for a dataset."""
    if getattr(dataset, "task", "sentiment") == "multilabel":
        return SimpleNamespace(
            build=multilabel.build_batch_prompt,
            parse=multilabel.parse_batch,
            label=multilabel.prompt_label,
            name="multilabel",
        )
    return SimpleNamespace(
        build=build_sentiment_batch_prompt,
        parse=parse_sentiment_batch,
        label=prompt_label,
        name="sentiment",
    )
