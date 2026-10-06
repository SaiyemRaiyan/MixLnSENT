"""EXPERIMENT 1 - FILTERING.

Prompt-based re-audit of the released BnSentMix data: decide whether each
sentence is valid Bengali-English code-mixed text (KEEP) or not (REMOVE).
"""

from ..common.config import FILTERING_OUTPUTS_DIR, FILTERING_PROMPTS_DIR

DEFAULT_PROMPT_VERSION = "v3"
PROMPT_FILENAME_TEMPLATE = "code_mix_filter_{version}.txt"
BATCH_FORMAT_PATH = FILTERING_PROMPTS_DIR / "batch_format_v1.txt"


def filter_prompt_path(version=None):
    version = version or DEFAULT_PROMPT_VERSION
    return FILTERING_PROMPTS_DIR / PROMPT_FILENAME_TEMPLATE.format(version=version)


def load_filter_prompt(version=None):
    path = filter_prompt_path(version)
    if not path.exists():
        available = sorted(p.name for p in path.parent.glob("code_mix_filter_*.txt"))
        raise FileNotFoundError(
            f"{path.name} not found. Available prompts: {', '.join(available)}"
        )
    return path.read_text(encoding="utf-8")


def prompt_version_label(version=None):
    return f"code_mix_filter_{version or DEFAULT_PROMPT_VERSION}"


def build_filter_prompt(sentence, version=None):
    return load_filter_prompt(version).format(sentence=sentence)


def build_batch_prompt(sentences, version=None):
    filter_prompt = load_filter_prompt(version)
    batch_format = BATCH_FORMAT_PATH.read_text(encoding="utf-8")
    items = [
        f"{index}: {sentence}"
        for index, sentence in enumerate(sentences, start=1)
    ]
    count_instruction = (
        f"There are exactly {len(sentences)} sentences. "
        f"Return exactly {len(sentences)} decision lines."
    )
    prompt = filter_prompt.replace("Sentence:\n{sentence}", "Sentences:\n")
    return prompt + count_instruction + "\n" + "\n".join(items) + batch_format


def parse_batch_response(response_text, expected_count):
    decisions = {}
    for line in response_text.strip().splitlines():
        if ":" not in line:
            continue
        index_text, decision_text = line.split(":", 1)
        index_text = index_text.strip()
        decision = decision_text.strip().upper()
        if not index_text.isdigit() or decision not in {"KEEP", "REMOVE"}:
            continue
        decisions[int(index_text)] = decision == "KEEP"

    expected_indices = set(range(1, expected_count + 1))
    if set(decisions) != expected_indices:
        raise ValueError(
            f"Could not parse all {expected_count} decisions: {response_text!r}"
        )
    return [decisions[index] for index in range(1, expected_count + 1)]


def decisions_path(provider, model, prompt_version=None):
    version = prompt_version or DEFAULT_PROMPT_VERSION
    safe_model = model.replace("/", "_").replace(":", "_")
    return FILTERING_OUTPUTS_DIR / "decisions" / f"{provider}_{safe_model}_{version}.jsonl"


def load_decisions(path):
    """Return {train_row_index: keep_bool} from a decisions JSONL file."""
    import json
    from pathlib import Path

    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        return {
            json.loads(line)["sentence_index"]: json.loads(line)["keep"]
            for line in handle
            if line.strip()
        }


def keep_indices(decisions, row_count):
    return [index for index in range(row_count) if decisions[index]]
