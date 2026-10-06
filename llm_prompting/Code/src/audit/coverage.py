"""Coverage matrix: which model has which prompt condition on a scope.

The paper's tables depend on knowing exactly which combinations were run, and on
distinguishing a complete run from a partial one. This prints that grid from the
files on disk, so no claim rests on memory.

Run:  python -m src.audit.coverage [dataset] [scope]
      python -m src.audit.coverage                 # bnsentmix, test
      python -m src.audit.coverage sentmix3l       # sentmix3l, full
"""

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common import datasets
from src.common.config import prompting_dir
from src.common.io_utils import load_jsonl
from src.prompting.prompts import parse_predictions_name

SHORT = {
    "qwen_qwen3.8-27b": "Qwen3.8-27B",
    "Qwen_Qwen3.8-27B": "Qwen3.8-27B/CC",
    "qwen_qwen3.8-27b_free": "Qwen3.8-27B/OR",
    "openai_gpt-oss-20b": "gpt-oss-20B",
    "openai_gpt-oss-120b": "gpt-oss-120B",
    "allam-2-7b": "ALLaM-7B",
    "google_gemini-3.8-flash": "Gemini-3.8-Flash",
    "meta_muse-spark-1.3-contributor": "Muse-Spark-1.3",
    "deepseek_deepseek-v4.1-flash": "DeepSeek-V4.1",
    "zai-org_GLM-5.3": "GLM-5.3",
    "gpt-5.6-sol": "GPT-5.6-Sol",
    "meta-llama_Llama-3.3-70B-Instruct": "Llama-3.3-70B",
}

# the elaboration axis, then the example-count axis
COLUMNS = [
    ("elab 0", "zero_shot_v1"),
    ("elab 2", "two_shot_v1"),
    ("elab 5", "five_shot_v1"),
    ("bare 0", "zero_shot_minimal_v1"),
    ("bare 2", "two_shot_minimal_v1"),
    ("bare 5", "five_shot_minimal_v1"),
]

PAIRS = [
    ("0-shot", "zero_shot_v1", "zero_shot_minimal_v1"),
    ("2-shot", "two_shot_v1", "two_shot_minimal_v1"),
    ("5-shot", "five_shot_v1", "five_shot_minimal_v1"),
]

# what each dataset's prompt-condition grid is expected to contain, so a missing
# cell reads as a gap rather than simply not being printed
EXPECTED = {
    "bnsentmix": [name for name, _ in COLUMNS],
    "sentmix3l": ["bare 0", "bare 2", "bare 5"],
}


def build(predictions, scope, expected):
    grid = defaultdict(dict)
    for path in sorted(predictions.glob(f"*__{scope}.jsonl")):
        _, model, prompt_name, _ = parse_predictions_name(path)
        records = {record["index"]: record["prediction"] for record in load_jsonl(path)}
        labelled = sum(1 for i in expected if records.get(i) is not None)
        grid[model][prompt_name] = (labelled, len(expected))
    return grid


def main():
    args = sys.argv[1:]
    dataset_name = args[0] if args else "bnsentmix"
    dataset = datasets.load(dataset_name)
    scope = args[1] if len(args) > 1 else ("full" if dataset_name == "sentmix3l" else "test")

    expected = set(dataset.indices(scope))
    predictions = prompting_dir(dataset_name, "predictions")
    grid = build(predictions, scope, expected)
    order = sorted(grid, key=lambda model: SHORT.get(model, model))

    columns = [
        (name, prompt) for name, prompt in COLUMNS
        if name in EXPECTED.get(dataset_name, [name for name, _ in COLUMNS])
    ]

    print("=" * 96)
    print(f"COVERAGE - {dataset_name}, {scope}, {len(expected):,} rows")
    print("=" * 96)
    print()
    header = f"  {'model':<20}" + "".join(f"{name:>9}" for name, _ in columns)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for model in order:
        line = f"  {SHORT.get(model, model):<20}"
        for _, prompt in columns:
            if prompt not in grid[model]:
                line += f"{'-':>9}"
            else:
                got, total = grid[model][prompt]
                line += f"{'ok' if got == total else str(got):>9}"
        print(line)
    print()
    print("  ok = complete;  a number = that many rows labelled;  - = never run")

    print()
    print("=" * 96)
    print("PER-CONDITION COVERAGE")
    print("=" * 96)
    for name, prompt in columns:
        complete = [
            model for model in order
            if prompt in grid[model] and grid[model][prompt][0] == grid[model][prompt][1]
        ]
        partial = [
            model for model in order
            if prompt in grid[model] and grid[model][prompt][0] != grid[model][prompt][1]
        ]
        print(f"  {name:<9} complete {len(complete):>2}   partial {len(partial):>2}   "
              f"{', '.join(SHORT.get(m, m) for m in complete)}")

    print()
    print("=" * 96)
    print("MODELS WITH BOTH PROMPT FAMILIES AT THE SAME EXAMPLE COUNT")
    print("=" * 96)
    for label, elaborate, bare in PAIRS:
        both = [
            model for model in order
            if elaborate in grid[model] and bare in grid[model]
            and grid[model][elaborate][0] == grid[model][elaborate][1]
            and grid[model][bare][0] == grid[model][bare][1]
        ]
        names = ", ".join(SHORT.get(m, m) for m in both) if both else "none"
        print(f"  {label:<9}{len(both)} models   {names}")


if __name__ == "__main__":
    main()
