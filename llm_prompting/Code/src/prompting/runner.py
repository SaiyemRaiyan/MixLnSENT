"""Shared prompting run loop.

The notebook and batch drivers both call these functions, so batching,
checkpointing, parsing and scoring exist in exactly one place.

`index` in every saved record is the row position in the full dataset, never the
position inside a scope. A dev run and a full run therefore refer to the same rows.
"""

import json
import time
from pathlib import Path

from ..common.config import prompting_dir
from ..common.io_utils import append_jsonl, load_jsonl
from ..common.metrics import sentiment_metrics
from .models import model_spec
from .multilabel import multilabel_metrics
from .tasks import task_for
from .providers import (
    build_sender,
    is_rate_limit_error,
    is_request_too_large,
    is_upstream_contention,
    retry_after_seconds,
)

QUOTA_MESSAGE = (
    "Provider quota reached. Progress is saved - replace the API key in .env or "
    "wait for the reset, then re-run to continue from the last saved row."
)

# How long to keep retrying when the upstream serving a model is busy. Aggregators
# route through shared pools that saturate intermittently, and a single batch can
# need several waits before it gets through. The cap keeps a permanently dead
# upstream from looping forever.
MAX_CONTENTION_WAITS = 30
CONTENTION_WAIT_CAP = 120


class QuotaReached(RuntimeError):
    """Raised when the provider refuses further requests for this key."""


class BatchTooLarge(RuntimeError):
    """Raised when the provider rejects a batch for exceeding a size limit.

    Separate from QuotaReached because the remedy is different: the batch must be
    split, not the key rotated. The provider's limit is per minute and includes the
    prompt, so a batch that fits at zero shot can exceed it once the prompt carries
    demonstration examples, and the longest rows can exceed it on their own.
    """


def write_records(path, records):
    ordered = [records[index] for index in sorted(records)]
    Path(path).write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in ordered) + "\n",
        encoding="utf-8",
    )


def classify_batch(send_prompt, texts, task, prompt_name, attempts=3, log=print):
    """Label one batch, retrying only the rows that came back unparsed.

    If the whole batch comes back unlabelled, split it and retry the halves. A
    provider-side refusal rejects an entire request, so without splitting one
    objectionable sentence would silently discard every label in its batch.

    A batch the provider rejects as too large is split for the same reason: the
    limit counts the prompt plus the batch against a per-minute token budget, so
    the batch size that works at zero shot can fail at five shots, and one long
    row can fail a batch on its own.
    """
    try:
        labels = _classify_once(send_prompt, texts, task, prompt_name, attempts, log)
    except BatchTooLarge:
        if len(texts) == 1:
            raise
        middle = len(texts) // 2
        log(f"  batch of {len(texts)} rejected as too large, splitting")
        return (
            classify_batch(send_prompt, texts[:middle], task, prompt_name, attempts, log)
            + classify_batch(send_prompt, texts[middle:], task, prompt_name, attempts, log)
        )

    if len(texts) > 1 and all(label is None for label in labels):
        middle = len(texts) // 2
        log(f"  batch of {len(texts)} returned nothing, splitting to isolate")
        return (
            classify_batch(send_prompt, texts[:middle], task, prompt_name, attempts, log)
            + classify_batch(send_prompt, texts[middle:], task, prompt_name, attempts, log)
        )

    return labels


def _classify_once(send_prompt, texts, task, prompt_name, attempts, log):
    labels = [None] * len(texts)
    pending = list(range(len(texts)))

    attempt = 0
    waits = 0

    while pending and attempt < attempts:
        chunk = [texts[index] for index in pending]
        try:
            prompt = task.build(chunk, prompt_name)
            parsed, _ = task.parse(send_prompt(prompt), len(chunk))
        except Exception as error:
            if is_request_too_large(error):
                # the request is oversized, so no key will help; bubble up so the
                # batch can be split rather than burning the key pool
                raise BatchTooLarge(str(error)[:200]) from error
            if is_rate_limit_error(error):
                # A per-minute allowance is not an exhausted account: the provider
                # names the wait in the message, and waiting that long is enough.
                # Rotating keys cannot outrun a clock, and stopping would abandon a
                # run over a limit that clears by itself.
                hint = retry_after_seconds(error)
                if hint is not None:
                    waits += 1
                    if waits > MAX_CONTENTION_WAITS:
                        log(f"  still rate limited after {waits} waits, stopping")
                        raise QuotaReached(QUOTA_MESSAGE) from error
                    delay = min(CONTENTION_WAIT_CAP, hint + 1)
                    log(f"  rate limited, waiting {delay:.0f}s (wait {waits})")
                    time.sleep(delay)
                    continue
                # no hint means the account itself is out of allowance
                log(f"  quota stop: {type(error).__name__}: {str(error)[:300]}")
                raise QuotaReached(QUOTA_MESSAGE) from error
            if is_upstream_contention(error):
                # the pool serving this model is busy; our key is fine and the same
                # one will work shortly, so back off without spending an attempt
                waits += 1
                if waits > MAX_CONTENTION_WAITS:
                    log(f"  upstream still busy after {waits} waits, stopping")
                    raise QuotaReached(QUOTA_MESSAGE) from error
                delay = min(CONTENTION_WAIT_CAP, 5 * waits)
                log(f"  upstream busy, waiting {delay}s (wait {waits})")
                time.sleep(delay)
                continue
            attempt += 1
            log(f"  request failed: {type(error).__name__}: {error}")
            time.sleep(5 * attempt)
            continue

        for position, label in enumerate(parsed):
            if label is not None:
                labels[pending[position]] = label
        pending = [index for index in pending if labels[index] is None]
        attempt += 1
        if pending:
            time.sleep(2)

    return labels


def run_scope(provider, model, prompt_name, scope, dataset, batch_size=None, log=print):
    """Classify every row in `scope` for one model and prompt. Resumable.

    `dataset` is a src.common.datasets.Dataset, which supplies the texts, the row
    indices for the scope, and the name used in output paths. The runner therefore
    knows nothing about which corpus it is running on.

    Returns (path, {index: record}). Raises QuotaReached to stop cleanly so the
    caller can swap keys and resume rather than losing the run.
    """
    spec = model_spec(provider, model)
    batch_size = batch_size or spec["batch_size"]
    send_prompt = build_sender(provider, model)
    task = task_for(dataset)

    texts = dataset.texts
    indices = dataset.indices(scope)
    safe_model = model.replace("/", "_").replace(":", "_")
    path = (
        prompting_dir(dataset.name, "predictions")
        / f"{provider}_{safe_model}_{task.label(prompt_name)}__{scope}.jsonl"
    )
    path.parent.mkdir(parents=True, exist_ok=True)

    done = {record["index"]: record for record in load_jsonl(path)}

    # a saved row without a label is not finished work: retry it
    todo = [
        index
        for index in indices
        if index not in done or done[index]["prediction"] is None
    ]
    log(f"{model} [{scope}] {prompt_name}: {len(done)} on disk, {len(todo)} to classify")
    if not todo:
        write_records(path, done)
        return path, done

    for start in range(0, len(todo), batch_size):
        chunk = todo[start:start + batch_size]
        labels = classify_batch(
            send_prompt, [texts[index] for index in chunk], task, prompt_name, log=log
        )

        records = [
            {"index": index, "prediction": labels[position]}
            for position, index in enumerate(chunk)
        ]
        append_jsonl(records, path)
        for record in records:
            done[record["index"]] = record

        filled = sum(1 for record in records if record["prediction"] is not None)
        log(
            f"  {min(start + batch_size, len(todo))}/{len(todo)} done, "
            f"labelled in batch: {filled}"
        )

    write_records(path, done)
    return path, done


def score_scope(scope, dataset, records):
    """Score saved predictions over the rows of `scope`.

    Rows with no prediction count as wrong, which keeps a partial run honest.
    Returns (metrics, missing_indices, indices_used).
    """
    indices = dataset.indices(scope)
    gold = dataset.gold
    missing = [
        index
        for index in indices
        if index not in records or records[index]["prediction"] is None
    ]

    if getattr(dataset, "task", "sentiment") == "multilabel":
        # a missing prediction cannot be scored, so it is excluded rather than
        # counted as wrong: unlike a single label there is no wrong value to use
        scored = [index for index in indices if index not in missing]
        predicted = [tuple(records[index]["prediction"]) for index in scored]
        return (
            multilabel_metrics([gold[index] for index in scored], predicted),
            missing,
            indices,
        )

    predicted = [
        records[index]["prediction"]
        if index not in missing
        else -1
        for index in indices
    ]
    return sentiment_metrics([gold[index] for index in indices], predicted), missing, indices
