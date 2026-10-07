"""Context-preserving language tag execution through the shared batch runner."""

import json
from pathlib import Path
from types import SimpleNamespace

from .config import XAI_PROMPTS_DIR
from .langtag import parse_langtag_batch
from .oracle import run_variants
from .tokenize import tokenize


def language_tag_variants(rows: list[dict]) -> list[dict]:
    variants = []
    for row in rows:
        for token in tokenize(row["sentence"]):
            variant_id = f"{row['index']}:langtag:{token.index:08d}"
            variants.append({
                "variant_id": variant_id,
                "src_index": row["index"],
                "kind": "langtag",
                "mask": [token.index],
                "rep": 0,
                "text": json.dumps(
                    {"context": row["sentence"], "token": token.text},
                    ensure_ascii=False,
                ),
            })
    return variants


def make_langtag_task():
    template = (XAI_PROMPTS_DIR / "langtag_v1.txt").read_text(encoding="utf-8")
    batch_format = (
        XAI_PROMPTS_DIR / "langtag_batch_format_v1.txt"
    ).read_text(encoding="utf-8")

    def build(records, _prompt_name):
        items = []
        for index, record in enumerate(records, 1):
            value = json.loads(record)
            items.append(
                f"{index}: Context: {value['context']}\nToken: {value['token']}"
            )
        return template.replace("{items}", "\n\n".join(items)) + "\n" + batch_format

    return SimpleNamespace(
        build=build,
        parse=parse_langtag_batch,
        label=lambda _name=None: "xai_langtag_v1",
        name="langtag",
    )


def run_langtags(rows, provider, model, output_path, scope, sender=None):
    from src.prompting.providers import build_sender

    return run_variants(
        language_tag_variants(rows),
        provider,
        model,
        Path(output_path),
        scope,
        prompt_name="langtag_v1",
        sender=sender or build_sender(provider, model),
        task=make_langtag_task(),
    )
