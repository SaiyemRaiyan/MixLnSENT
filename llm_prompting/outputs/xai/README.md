# XAI experiment operator guide

This is a BnSentMix-only behavioural-XAI implementation. It does not modify the
existing filtering or prompting pipelines, and its hosted-model requests are opt-in.
All inference uses the already-configured provider keys; never place keys in run
files or notebooks.

## 1. Prepare and validate on train-only development data

From the repository root, run:

```powershell
python -m src.xai.cli prepare-dev
python -m src.xai.cli run --stage instrument --sample dev --scope dev --provider groq --model qwen/qwen3.8-27b
python -m src.xai.cli run --stage instrument --sample dev --scope dev --provider groq --model openai/gpt-oss-20b
python -m src.xai.cli run --stage instrument --sample dev --scope dev --provider huggingface --model meta-llama/Llama-3.1-8B-Instruct
python -m src.xai.cli run --stage baseline --sample dev --scope dev --provider groq --model qwen/qwen3.8-27b
python -m src.xai.cli run --stage rationale --sample dev --scope dev --provider groq --model qwen/qwen3.8-27b
python -m src.xai.cli run --stage baseline --sample dev --scope dev --provider groq --model openai/gpt-oss-20b
python -m src.xai.cli run --stage rationale --sample dev --scope dev --provider groq --model openai/gpt-oss-20b
python -m src.xai.cli run --stage baseline --sample dev --scope dev --provider huggingface --model meta-llama/Llama-3.1-8B-Instruct
python -m src.xai.cli run --stage rationale --sample dev --scope dev --provider huggingface --model meta-llama/Llama-3.1-8B-Instruct
python -m src.xai.cli validate-dev --mode lr
python -m src.xai.cli validate-dev --mode planted-cue --coalitions 96
```

If the planted-cue gate fails, repeat it at larger budgets through 192. To choose
the coalition budget, run `--stage coalition --sample dev --scope dev` with
`--coalitions 32`, `64`, `96`, and `192`, then:

```powershell
python -m src.xai.cli select-coalition-budget --provider groq --model qwen/qwen3.8-27b
python -m src.xai.cli finalize-dev-gate
```

`prepare-dev` adds only as many extra neutral rows from the training split as needed
to provide 200 neutral examples for the two planted-cue groups. It never samples
validation or test rows.

## 2. Create samples, prepare human materials, freeze

```powershell
python -m src.xai.cli prepare-test
python -m src.xai.cli prepare-annotations
python -m src.xai.cli make-demo-ablation-prompts
```

Complete and approve every row in
`outputs/xai/annotation/operator_signoff.csv`. Have annotators complete the
language-tag and evidence-token sheets. Language/position claims require G4 accuracy
of at least 0.90; otherwise those findings must stay labelled unvalidated.

After G1, G2, and all three model-specific rationale prompt checks (at least 95%
parsed and at least 90% verbatim word matches) have passed, operator sign-off is
complete, and dev-only prompt choices are frozen:

```powershell
python -m src.xai.cli preregister
python -m src.xai.cli check-prereg
```

The test-scope runner fails closed if the preregistration, either gate, or any
hashed sample/config/prompt/source file is absent or has changed.

## 3. Run the frozen test sequence

The full-model commands below are examples; repeat each stage for all three
full-analysis models, in this order. They resume from JSONL checkpoints after quota
interruptions. `QuotaReached` is surfaced; it is not converted to a successful
partial run.

```powershell
python -m src.xai.cli run --stage baseline --sample xai800 --scope test --provider groq --model qwen/qwen3.8-27b
python -m src.xai.cli run --stage occlusion --sample xai800 --scope test --provider groq --model qwen/qwen3.8-27b
python -m src.xai.cli run --stage counterfactual --sample xai800 --scope test --provider groq --model qwen/qwen3.8-27b
python -m src.xai.cli run --stage rationale --sample xai800 --scope test --provider groq --model qwen/qwen3.8-27b
python -m src.xai.cli run --stage coalition --sample xai200 --scope test --provider groq --model qwen/qwen3.8-27b
python -m src.xai.cli run --stage faithfulness --sample xai200 --scope test --provider groq --model qwen/qwen3.8-27b
```

E9 uses `--stage prompt --sample xai800 --scope test` with `zero_shot_v1`,
`zero_shot_ablate_drop_rule1_v1` through `zero_shot_ablate_drop_rule7_v1`,
`zero_shot_ablate_drop_mixed_def_v1`, `five_shot_minimal_v1`, and each generated
leave-one-demonstration-out prompt. The unchanged primary prompt remains
`zero_shot_minimal_v1`.

For E13, use `commandcode/google/gemini-3.8-flash` on `gemini300` for E3, E4, E6,
and E8. Use one baseline pass on `gemini300`, then a one-pass baseline on
`gemini_noise100` for the planned noise estimate.

Generate derived tables, report, and figures only from complete runs:

```powershell
python -m src.xai.cli report
```

## Run record integrity

Every classified variant is appended to JSONL with its source row, perturbation,
mask, replicate, prediction, provider/model/prompt, and timestamp. A sidecar manifest
stores the expected row count, variant/input hashes, prompt hashes, batch settings,
and git commit. Downstream analysis rejects missing, unparsed, changed, or partial
runs. Reported metric computation never scrapes a partial run.
