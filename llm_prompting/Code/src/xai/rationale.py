"""Extractive rationale collection and auditable token matching."""

import json
import re
from types import SimpleNamespace

from .config import XAI_PROMPTS_DIR
from .tokenize import tokenize

_ITEM = re.compile(r"^\s*(\d+)\s*[:.)-]\s*(.*?)\s*$")


def parse_rationale_batch(response: str, expected_count: int):
    found: dict[int, str | None] = {}
    for line in response.splitlines():
        match = _ITEM.match(line)
        if not match:
            continue
        index = int(match.group(1))
        words = [part.strip() for part in re.split(r"\s*(?:\||;|,)\s*", match.group(2))]
        words = [word for word in words if word]
        if len(words) == 1 and words[0].lower() in ("none", "n/a"):
            found[index] = "NONE"
        else:
            found[index] = " | ".join(words[:3]) if words else None
    parsed = [found.get(index) for index in range(1, expected_count + 1)]
    missing = [index for index, value in enumerate(parsed, 1) if value is None]
    return parsed, missing


def _prompt_template() -> str:
    return (XAI_PROMPTS_DIR / "rationale_extractive_v1.txt").read_text(encoding="utf-8")


def make_rationale_task(_predictions: dict[int, int]):
    template = _prompt_template()

    def build(texts, _prompt_name):
        items = []
        for position, text in enumerate(texts, 1):
            source_index, predicted_label = text.rsplit("\n[XAI_LABEL] ", 1)
            label_name = ("Positive", "Negative", "Neutral", "Mixed")[int(predicted_label)]
            items.append(f"{position}: {source_index}\nPredicted label: {label_name}")
        batch_format = (
            XAI_PROMPTS_DIR / "rationale_batch_format_v1.txt"
        ).read_text(encoding="utf-8")
        return template.replace("{items}", "\n\n".join(items)) + "\n" + batch_format

    return SimpleNamespace(
        build=build,
        parse=parse_rationale_batch,
        label=lambda _name=None: "xai_rationale_extractive_v1",
        name="rationale",
    )


def rationale_variants(rows: list[dict], predictions: dict[int, int]) -> list[dict]:
    variants = []
    for row in rows:
        if row["index"] not in predictions:
            continue
        variants.append({
            "variant_id": f"{row['index']}:rationale:00000000",
            "src_index": row["index"],
            "kind": "rationale",
            "mask": [],
            "rep": 0,
            "text": f"{row['sentence']}\n[XAI_LABEL] {predictions[row['index']]}",
        })
    return variants


def match_rationale_words(sentence: str, rationale: str) -> dict:
    tokens = tokenize(sentence)
    words = [word.strip() for word in re.split(r"\s*(?:\||;|,)\s*", rationale) if word.strip()]
    normalized = {
        token.index: token.text.strip(".,!?;:()[]{}\"'").casefold()
        for token in tokens
    }
    matched = []
    unmatched = []
    used = set()
    for word in words[:3]:
        key = word.strip(".,!?;:()[]{}\"'").casefold()
        candidates = [index for index, text in normalized.items() if text == key and index not in used]
        if candidates:
            index = candidates[0]
            matched.append({"word": word, "token_index": index})
            used.add(index)
        else:
            unmatched.append(word)
    return {
        "words": words[:3],
        "matched": matched,
        "unmatched": unmatched,
        "match_rate": len(matched) / len(words[:3]) if words[:3] else 1.0,
    }


def validate_rationales(rows: list[dict], outputs: list[dict]) -> dict:
    row_by_index = {row["index"]: row for row in rows}
    parsed = 0
    requested = 0
    verbatim = 0
    total_words = 0
    records = []
    for output in outputs:
        row = row_by_index[output["src_index"]]
        requested += 1
        rationale = output.get("rationale")
        if rationale is None:
            continue
        parsed += 1
        if rationale.strip().upper() == "NONE":
            records.append({**output, "words": [], "matched": [], "unmatched": [], "match_rate": 1.0})
            continue
        matched = match_rationale_words(row["sentence"], rationale)
        total_words += len(matched["words"])
        verbatim += len(matched["matched"])
        records.append({**output, **matched})
    return {
        "parse_rate": parsed / requested if requested else 0.0,
        "verbatim_match_rate": verbatim / total_words if total_words else 0.0,
        "requested": requested,
        "parsed": parsed,
        "words": total_words,
        "matched_words": verbatim,
        "records": records,
    }


def run_rationales(rows, predictions, provider, model, output_path, scope, sender):
    from pathlib import Path

    from .oracle import run_variants

    variants = rationale_variants(rows, predictions)
    path = Path(output_path)
    task = make_rationale_task(predictions)
    results = run_variants(
        variants, provider, model, path, scope,
        prompt_name="rationale_extractive_v1",
        sender=sender,
        task=task,
    )
    for record in results:
        record["rationale"] = record["prediction"]
        record["text"] = record["text"].rsplit("\n[XAI_LABEL] ", 1)[0]
        record["prompt"] = "rationale_extractive_v1"
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in results) + "\n",
        encoding="utf-8",
    )
    return results
