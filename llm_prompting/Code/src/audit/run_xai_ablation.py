"""Run single-element XAI prompt ablations on SentMix-3L.

Evaluates how removing specific instructions, lexicon cues, or label definitions
affects model accuracy and invalid label emissions (e.g. Mixed on a 3-class corpus).
"""

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.datasets import load
from src.prompting.runner import run_scope

PROMPTS = [
    "zero_shot_v1",  # Baseline: 7 rules, lexicon cues, Mixed guidance
    "zero_shot_ablate_drop_mixed_def_v1",  # Removes Mixed label definition
    "zero_shot_ablate_drop_rule1_v1",  # Removes bilingual understanding instruction
    "zero_shot_ablate_drop_rule2_v1",  # Removes Banglish lexicon cues (bhalo, kharap, kintu)
    "zero_shot_ablate_drop_rule3_v1",  # Removes full sentence interpretation rule
    "zero_shot_ablate_drop_rule4_v1",  # Removes strict Mixed constraint rule
    "zero_shot_ablate_drop_rule5_v1",  # Removes Neutral fallback for unclear text
    "zero_shot_ablate_drop_rule6_v1",  # Removes translation prohibition
    "zero_shot_ablate_drop_rule7_v1",  # Removes label invention prohibition
]


def main():
    provider = "groq"
    model = "qwen/qwen3.8-27b"
    scope = "full"
    batch_size = 20  # Safe size for Groq 8k TPM on SentMix-3L

    dataset = load("sentmix3l")
    print(f"Loaded {dataset.name} with {len(dataset)} rows.")
    print(f"Running XAI Prompt Ablations for {model} on {provider}...")
    print("-" * 70)

    for idx, prompt_name in enumerate(PROMPTS, 1):
        print(f"\n[{idx}/{len(PROMPTS)}] Executing ablation: {prompt_name}")
        t0 = time.time()
        try:
            path, done = run_scope(
                provider=provider,
                model=model,
                prompt_name=prompt_name,
                scope=scope,
                dataset=dataset,
                batch_size=batch_size,
                log=lambda *a: print(*a, flush=True),
            )
            elapsed = time.time() - t0
            print(f"-> Completed {prompt_name}: {len(done)} records in {elapsed:.1f}s")
        except Exception as e:
            print(f"-> ERROR on {prompt_name}: {e}")
            raise e

    print("\nAll prompt ablations finished successfully!")


if __name__ == "__main__":
    main()
