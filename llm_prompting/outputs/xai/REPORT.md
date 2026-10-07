# Behavioral explainability of BnSentMix sentiment classification

**Report status: DEVELOPMENT / NOT FROZEN** — regenerated from complete run manifests only. A run is included only when its manifest says complete and its JSONL row count and variant-ID digest both match.

## Setup and instrument checks (E0, E1)

Full G1/G2 preregistration status: {"status": "incomplete; no dev_gate.json"}.

Partial G1 development instrument results (agreement; ε = 1 − repeat agreement):

| Model | A–A2 repeat | Noise ε | A–B shuffled | A–C single |
|---|---:|---:|---:|---:|
| groq/qwen/qwen3.8-27b | 1.000 | 0.000 | 0.885 | 0.764 |
| groq/openai/gpt-oss-20b | 0.986 | 0.014 | 0.817 | 0.732 |

Required but unmeasured model(s): huggingface/meta-llama/Llama-3.1-8B-Instruct.

LR-oracle E1: n=300, identifiable=71, unidentifiable=229, median Spearman=0.683 (gate ≥0.600), top-1 agreement=0.676 (gate ≥0.700).

E1 correlations and top-1 agreement are evaluated only where hard-label occlusion produces a non-constant token ranking; constant rows are unidentifiable rather than assigned a fabricated zero correlation or first-token tie-break. E1 thresholds require median Spearman ≥0.600 and top-1 agreement ≥0.700. Because the local LR-oracle top-1 threshold failed, coalition/KernelSHAP claims are not validated. No planted-cue gate, rationale freeze, preregistration, or test-scope inference is recorded.

| Model / sample | n | Accuracy | Per-class recall (label ids: 0 Positive, 1 Negative, 2 Neutral, 3 Mixed) |
|---|---:|---:|---|
| groq/qwen/qwen3.8-27b [dev] | 580 | 0.734 | 0:0.799, 1:0.670, 2:0.784, 3:0.569 |

## Registered hypotheses

### P1 Attribution validity

| Model/condition | n | Effect | 95% CI | Outcome |
|---|---:|---:|---|---|
| — | 0 | — | — | Inconclusive (not run) |

### P2 Self-rationale faithfulness

| Model/condition | n | Effect | 95% CI | Outcome |
|---|---:|---:|---|---|
| — | 0 | — | — | Inconclusive (not run) |

### P3 Mixed evidence grounding

| Model/condition | n | Effect | 95% CI | Outcome |
|---|---:|---:|---|---|
| — | 0 | — | — | Inconclusive (not run) |

### P4 Polarity sensitivity

| Model/condition | n | Effect | 95% CI | Outcome |
|---|---:|---:|---|---|
| — | 0 | — | — | Inconclusive (not run) |

### P5 Prompt components (McNemar, Holm-adjusted within model)

| Model/condition | n | Effect | 95% CI | Outcome |
|---|---:|---:|---|---|
| — | 0 | — | — | Inconclusive (not run) |

## Per-model faithfulness with confidence intervals

| Model | Evidence intervention | n | Rate | 95% CI |
|---|---|---:|---:|---|
| — | — | 0 | — | — |

## Mixed analysis (E8 C1–C3)

Clause deletion transition counts (baseline label to variant label):

| Model | Operator | Transitions |
|---|---|---|
| — | — | — |

## Prompt-component results (E9)

P5 outcome table above is derived from common source rows in complete primary and ablation runs. Each recorded comparison includes the per-sentence correctness delta and Holm-adjusted p-value.

## Exploratory results

Language/position, demonstration, and cross-method analyses are exploratory. Language-tag findings are not validated unless the 300-token hand-label gate G4 reaches 0.90 accuracy.

## Complete runs

| Stage | Model | Scope | Prompt | Complete rows |
|---|---|---|---|---:|
| E0_A | groq/openai/gpt-oss-20b | dev | sentiment_zero_shot_minimal_v1 | 585 |
| E0_A | groq/qwen/qwen3.8-27b | dev | sentiment_zero_shot_minimal_v1 | 585 |
| E0_A2 | groq/openai/gpt-oss-20b | dev | sentiment_zero_shot_minimal_v1 | 585 |
| E0_A2 | groq/qwen/qwen3.8-27b | dev | sentiment_zero_shot_minimal_v1 | 585 |
| E0_B | groq/openai/gpt-oss-20b | dev | sentiment_zero_shot_minimal_v1 | 585 |
| E0_B | groq/qwen/qwen3.8-27b | dev | sentiment_zero_shot_minimal_v1 | 585 |
| E0_C | groq/openai/gpt-oss-20b | dev | sentiment_zero_shot_minimal_v1 | 585 |
| E0_C | groq/qwen/qwen3.8-27b | dev | sentiment_zero_shot_minimal_v1 | 585 |
| E0_D | groq/openai/gpt-oss-20b | dev | sentiment_zero_shot_minimal_v1 | 1170 |
| E0_D | groq/qwen/qwen3.8-27b | dev | sentiment_zero_shot_minimal_v1 | 1170 |
| E1_planted | groq/qwen/qwen3.8-27b | dev | sentiment_zero_shot_minimal_v1 | 200 |
| E3 | groq/qwen/qwen3.8-27b | dev | sentiment_zero_shot_minimal_v1 | 1755 |

## Limitations

Hosted models expose hard labels in this pipeline. Results are behavioural intervention evidence, not access to internal representations or proof of causal mechanisms. The optional white-box arm is not claimed unless its GPU dependencies and local/API agreement gate are satisfied.

## Deviations

See [DEVIATIONS.md](./DEVIATIONS.md). No hypothesis outcome or metric is filled from incomplete checkpoints.
