"""Frozen settings and output locations for the BnSentMix XAI experiment."""

from pathlib import Path

from src.common.config import PROJECT_ROOT, SEED

XAI_PROMPTS_DIR = PROJECT_ROOT / "prompts" / "xai"
XAI_OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "xai"
SAMPLES_DIR = XAI_OUTPUTS_DIR / "samples"
VARIANTS_DIR = XAI_OUTPUTS_DIR / "variants"
LABELS_DIR = XAI_OUTPUTS_DIR / "labels"
ATTRIBUTIONS_DIR = XAI_OUTPUTS_DIR / "attributions"
RESULTS_DIR = XAI_OUTPUTS_DIR / "results"
FIGURES_DIR = XAI_OUTPUTS_DIR / "figures"
ANNOTATION_DIR = XAI_OUTPUTS_DIR / "annotation"

PRIMARY_PROMPT = "zero_shot_minimal_v1"
SECONDARY_PROMPT = "zero_shot_v1"
FULL_MODELS = (
    ("groq", "qwen/qwen3.8-27b"),
    ("groq", "openai/gpt-oss-20b"),
    ("huggingface", "meta-llama/Llama-3.1-8B-Instruct"),
)
GEMINI_MODEL = ("commandcode", "google/gemini-3.8-flash")

N_LABELS = 4
SAMPLE_PER_CLASS = 200
ATTRIBUTION_PER_CLASS = 50
GEMINI_PER_CLASS = 75
MAX_TOKENS = 30
BASELINE_REPLICATES = 3
EVIDENCE_SIZES = (1, 2, 3)
RANDOM_DRAWS = 5
COALITION_BUDGETS = (32, 64, 96, 192)
COALITION_ALPHA = 0.01
BOOTSTRAP_RESAMPLES = 10_000

SAMPLE_SEED = SEED
VARIANT_SEED = SEED + 1
BOOTSTRAP_SEED = SEED + 2

PROMPT_HASH_FILES = (
    XAI_PROMPTS_DIR / "rationale_extractive_v1.txt",
    XAI_PROMPTS_DIR / "rationale_batch_format_v1.txt",
    XAI_PROMPTS_DIR / "langtag_v1.txt",
    XAI_PROMPTS_DIR / "langtag_batch_format_v1.txt",
    PROJECT_ROOT / "prompts" / "prompting" / "zero_shot_minimal_v1.txt",
    PROJECT_ROOT / "prompts" / "prompting" / "batch_format_v1.txt",
)


def ensure_output_dirs() -> None:
    for directory in (
        SAMPLES_DIR, VARIANTS_DIR, LABELS_DIR, ATTRIBUTIONS_DIR,
        RESULTS_DIR, FIGURES_DIR, ANNOTATION_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def output_path(*parts: str) -> Path:
    """Return a path below outputs/xai without allowing path traversal."""
    path = (XAI_OUTPUTS_DIR / Path(*parts)).resolve()
    if not path.is_relative_to(XAI_OUTPUTS_DIR.resolve()):
        raise ValueError("XAI output paths must remain under outputs/xai")
    return path
