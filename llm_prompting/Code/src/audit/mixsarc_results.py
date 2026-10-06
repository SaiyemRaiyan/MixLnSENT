"""MixSarc results table.

Reports each completed run per label against the numbers published with the corpus.
The paper reports exact-match accuracy plus macro precision, recall and F1, and a
per-label breakdown, so both are reproduced here.

Comparability caveat: the authors do not release their 70:15:15 partition or its
seed. The split used here is rebuilt the same way under a fixed seed, so it is the
same construction rather than the same rows. The headline comparison is against their
published zero-shot LLM results.

Run:  python -m src.audit.mixsarc_results
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common import datasets
from src.common.config import prompting_dir
from src.common.io_utils import load_jsonl
from src.prompting.multilabel import LABELS, multilabel_metrics
from src.prompting.prompts import parse_predictions_name

DATASET = "mixsarc"
PREDICTIONS = prompting_dir(DATASET, "predictions")
RESULTS = prompting_dir(DATASET, "results")

# Table 5 of the corpus paper: zero-shot LLMs on their held-out test split.
PUBLISHED = {
    "Claude Opus 4.8": {"Humorous": 0.6294, "Sarcastic": 0.3611,
                        "Offensive": 0.3206, "Vulgar": 0.3784},
    "GPT-5.5": {"Humorous": 0.6282, "Sarcastic": 0.2408,
                "Offensive": 0.3071, "Vulgar": 0.2981},
    "LLaMA 4 Maverick": {"Humorous": 0.5911, "Sarcastic": 0.3905,
                         "Offensive": 0.2083, "Vulgar": 0.2018},
    "Kimi K2 0711": {"Humorous": 0.5369, "Sarcastic": 0.3281,
                     "Offensive": 0.2548, "Vulgar": 0.3188},
    "Gemini 2.5 Flash": {"Humorous": 0.4796, "Sarcastic": 0.1873,
                         "Offensive": 0.1923, "Vulgar": 0.3247},
}

# supervised, for the boundary the task sets on prompting
SUPERVISED = {
    "Banglish-BERT": {"Humorous": 0.7080, "Sarcastic": 0.3938,
                      "Offensive": 0.0563, "Vulgar": 0.1928},
    "Gemma-2B": {"Humorous": 0.7031, "Sarcastic": 0.1553,
                 "Offensive": 0.0000, "Vulgar": 0.0563},
}

SHORT = {
    "zai-org_GLM-5.3": "GLM-5.3",
    "gpt-5.6-sol": "GPT-5.6-Sol",
    "deepseek_deepseek-v4.1-flash": "DeepSeek-V4.1",
    "Qwen_Qwen3.8-27B": "Qwen3.8-27B",
    "google_gemini-3.8-flash": "Gemini-3.8-Flash",
    "openai_gpt-oss-20b": "gpt-oss-20B",
    "openai_gpt-oss-120b": "gpt-oss-120B",
    "allam-2-7b": "ALLaM-7B",
    "meta-llama_Llama-3.3-70B-Instruct": "Llama-3.3-70B",
}


def main():
    dataset = datasets.load(DATASET)
    gold = dataset.gold
    rows = dataset.indices("test")

    print("=" * 96)
    print(f"MIXSARC - {len(rows)} test rows, {len(LABELS)} binary labels, zero-shot")
    print("=" * 96)
    print()
    header = (f"  {'model':<18}{'exact':>8}" + "".join(f"{n[:9]:>10}" for n in LABELS)
              + f"{'macroF1':>9}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    payload = {}
    for path in sorted(PREDICTIONS.glob("*.jsonl")):
        records = {r["index"]: r["prediction"] for r in load_jsonl(path)}
        scored = [i for i in rows if records.get(i) is not None]
        if not scored:
            continue
        _, model, prompt_name, _ = parse_predictions_name(path)
        metrics = multilabel_metrics([gold[i] for i in scored],
                                     [records[i] for i in scored])
        complete = len(scored) == len(rows)
        name = SHORT.get(model, model)
        flag = "" if complete else f"  ({len(scored)}/{len(rows)})"
        cells = "".join(f"{metrics['per_label'][n]['f1']:>10.4f}" for n in LABELS)
        print(f"  {name:<18}{metrics['exact_match_accuracy']:>8.4f}{cells}"
              f"{metrics['macro']['f1']:>9.4f}{flag}")
        payload[name] = {
            "complete": complete,
            "rows": len(scored),
            "exact_match": round(metrics["exact_match_accuracy"], 4),
            "macro_f1": round(metrics["macro"]["f1"], 4),
            "per_label_f1": {n: round(metrics["per_label"][n]["f1"], 4)
                             for n in LABELS},
        }

    print()
    print("  PUBLISHED ZERO-SHOT LLMs (their test split, their prompt)")
    print("  " + "-" * (len(header) - 2))
    for name, scores in PUBLISHED.items():
        macro = sum(scores.values()) / len(LABELS)
        cells = "".join(f"{scores[n]:>10.4f}" for n in LABELS)
        print(f"  {name:<18}{'--':>8}{cells}{macro:>9.4f}")

    print()
    print("  PUBLISHED SUPERVISED (fine-tuned, for the boundary)")
    print("  " + "-" * (len(header) - 2))
    for name, scores in SUPERVISED.items():
        macro = sum(scores.values()) / len(LABELS)
        cells = "".join(f"{scores[n]:>10.4f}" for n in LABELS)
        print(f"  {name:<18}{'--':>8}{cells}{macro:>9.4f}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "mixsarc_table.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print()
    print("saved:", out)
    print()
    print("  caveat: their 70:15:15 partition is not released, so the split used here is")
    print("  the same construction under a fixed seed rather than the same rows.")


if __name__ == "__main__":
    main()
