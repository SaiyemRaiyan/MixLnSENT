# XAI Experiment Plan: Behavioural Explainability of LLM Sentiment Classification on Code-Mixed Bengali-English Text

Status: v1, written for implementation by an AI agent. Target repo: Sentiment-Analysis (BnSentMix prompting study). Scope of v1: BnSentMix only. SentMix-3L and MixSarc are out of scope.

## 0. Rules of engagement (read first)

1. Read before writing: README.md; src/common/{config,data,datasets,io\_utils,metrics}.py; src/prompting/{prompts,providers,models,runner,tasks}.py; src/audit/definition\_compliance.py; prompts/prompting/\*.txt; notebooks 07 and 08. Reuse build\_sender, classify\_batch, the JSONL append/resume helpers and the task.build / task.parse path. Do not re-implement providers, batching, retries or quota handling.
2. Secrets: never open, print, log or commit .env. No keys in any output file.
3. Additive only: do not edit or delete existing files in src/, prompts/, outputs/ or the notebooks. New code goes in src/xai/, new prompts in prompts/xai/, new outputs in outputs/xai/. The only allowed edit to an existing file is adding a MODEL\_SPECS entry for a model that lacks one, with values measured on dev.
4. Hold the instrument constant: every classification query uses an existing prompt file byte-for-byte through the existing batch-format path. A perturbed text only replaces the sentence slot.
5. No test peeking: develop and tune on dev, freeze, then run test once. Section 7 defines mechanical gates; do not bypass them.
6. Numbers come only from complete runs: every table or statement is generated from run files whose row count equals the manifest count (assert this in code, never scrape partial runs).
7. Report as found. Never tune a choice toward a hoped-for result. Every deviation from this plan goes in outputs/xai/DEVIATIONS.md (what, why, date, and whether it was made before or after seeing test data).
8. Quota: stop on QuotaReached and resume later from the checkpoint. Use only the keys already configured in the project.
9. If something is unclear, write it in outputs/xai/OPEN\_QUESTIONS.md, take the conservative option and continue. Do not block.

## 1. Verified starting point

- Dataset: BnSentMix, 20,015 rows, labels 0 Positive, 1 Negative, 2 Neutral, 3 Mixed. Fixed stratified 70/15/15 split, SEED 42. Test has 3,003 rows (Mixed 281). A 600-row dev set is drawn from train (dev\_indices). Median row length is about 9 words.
- Instrument: sentences are classified in numbered-list batches (prompts/prompting/batch\_format\_v1.txt plus a prompt file). Batch sizes live in src/prompting/models.py (35 on Groq, 200 on Hugging Face). Temperature is 0 everywhere. Reasoning is hidden or low for reasoning models.
- Saved prediction rows contain only index, sentence, gold, prediction, provider, model, prompt. There are no logits, no logprobs and no rationales.
- Reference test results (3,003 rows): Qwen3.8-27B bare 0.7766 and elaborate 0.7556; gpt-oss-20b bare 0.7952 (Mixed precision 0.977, recall 0.299) and elaborate 0.7433; Llama-3.1-8B zero-shot 0.666 (Mixed F1 about 0.18); Gemini 3.8 Flash five-shot minimal 0.8388 (best).
- Existing assets to reuse: TF-IDF Logistic Regression, Random Forest and SVM (src/filtering/classifiers.py); contrast-marker definitions (src/audit/definition\_compliance.py); leakage and duplicate helpers (src/common/data.py); rule-ablation prompts zero\_shot\_ablate\_drop\_rule1..7 and zero\_shot\_ablate\_drop\_mixed\_def in prompts/prompting/; few-shot prompts including five\_shot\_minimal\_v1.
- Constraint that shapes everything: the main models are API-served and return hard labels only. Therefore the XAI here is behavioural (perturb the input or prompt, observe the label). Gradient methods are possible only on the open-weight Llama-3.1-8B-Instruct and are an optional extension (E12).

## 2. Objective, research questions and pre-registered hypotheses

Objective: determine whether LLM sentiment predictions on Bengali-English code-mixed text rest on the right evidence, and whether the models' own explanations are faithful to what actually drives their predictions.

Why this is worth asking: gpt-oss-20b has high accuracy but predicts Mixed only when very sure (precision 0.98, recall 0.30); Llama-3.1-8B collapses on Mixed; few-shot hurt the 8B model. Explanation experiments can say why.

Research questions

- RQ1 Where is the evidence? Do models put attribution on polarity words, and does the mass differ between English and romanized-Bengali tokens or by position?
- RQ2 What produces Mixed? Hypothesis: Mixed is predicted only when the model finds a positive and a negative span.
- RQ3 Are self-explanations faithful, in the sense that removing the words a model names changes its label more than removing random words?
- RQ4 Which prompt components and demonstrations drive predictions?
- RQ5 (optional, GPU) Do black-box attributions agree with gradient attributions on an open-weight model?

Primary hypotheses (each tested per model, effect floor delta = 0.10 where stated; all thresholds fixed now)

- P1 Attribution validity: top-3 words by coalition attribution give higher comprehensiveness than 3 random words (paired bootstrap CI of the difference excludes 0).
- P2 Rationale faithfulness: the model's self-reported evidence words give higher comprehensiveness than a size-matched random set (CI excludes 0). Falsifier: rationale about equal to random means the explanation is unfaithful.
- P3 Mixed is evidence-grounded: among rows the model labels Mixed at baseline and that contain a strict contrast marker, deleting either clause moves the label away from Mixed at a rate at least 0.10 above the noise-floor rate.
- P4 Polarity sensitivity: on single-polarity rows with exactly one lexicon hit, swapping the hit for its antonym flips the label to the opposite polarity at a rate at least 0.10 above the noise-floor rate.
- P5 Prompt components: dropping each of the 8 prompt components (rules 1 to 7, Mixed definition) changes per-sentence correctness (McNemar, Holm-corrected within the family). Exploratory (labelled as such in the report): language split of attribution mass, positional bias, cross-model and cross-method agreement, demonstration ablation, white-box agreement.

## 3. Frozen design decisions

| Item | Decision |
| --- | --- |
| Full-analysis models | groq qwen/qwen3.8-27b; groq openai/gpt-oss-20b; huggingface meta-llama/Llama-3.1-8B-Instruct |
| Restricted replication | commandcode google/gemini-3.8-flash on a 300-row subset (75 per class): E3, E4, E6, E8 only |
| Primary prompt | prompts/prompting/zero\_shot\_minimal\_v1.txt. Reason: best accuracy for two of the three models and it contains no hand-written cue lists, so attributions cannot be artefacts of lexicon hints in the prompt |
| Secondary prompt | zero\_shot\_v1.txt, used only in E9 |
| Units | whitespace tokens with attached punctuation kept; each token has an index and character span |
| Primary masking | delete the token. Sensitivity masking: replace with the literal token \[MASK\]. Never produce an empty text; skip empty variants |
| Test sample (XAI-800) | from the test split; exclude rows in duplicate groups with conflicting labels and rows over 30 tokens; seeded shuffle; 200 per gold class (Mixed 200 of 281); log the excluded fraction |
| Attribution subset (XAI-200) | 50 per class taken from XAI-800 by seeded shuffle |
| Gemini subset | 75 per class from XAI-800 by seeded shuffle |
| Dev sample | dev\_indices filtered by the same rules, plus neutral dev rows for the planted-cue test |
| Baseline label y0 | majority of 3 replicate passes; a row with no majority is flagged unstable and excluded from attribution (reported) |
| Evidence size k | 1, 2, 3 |
| Random baselines | 5 seeded draws per sentence per k |
| Coalition count M | chosen on dev from {32, 64, 96, 192} as the smallest M whose two-seed top-3 overlap (Jaccard) averages at least 0.8 on Qwen; cap 192 |
| Batching protocol P\_batch | production batch size from MODEL\_SPECS; seeded shuffle; no two variants of the same source sentence in one batch (prevents the model enforcing consistency across near-duplicates) |
| Seeds | one seed constant in src/xai/config.py; every sample file stores its SHA256 |

Definitions

- Outcome of a variant: yhat(variant), a label from the primary prompt.
- Agreement outcome: v = 1 if yhat(variant) equals y0, else 0.
- Noise floor eps\_m for model m: label disagreement rate between two identical replicate passes (from E0 on dev and from E3 on test).

## 4. Experiments

Every experiment writes JSONL under outputs/xai/ and records provider, model, prompt name, date and git commit.

### E0 Instrument equivalence and noise floor (dev, per model)

Conditions on the 600 dev rows: A production-order batches; A2 an identical rerun of A; B shuffled batches; C single-sentence calls; D batches built under P\_batch with variants of the same sentence kept apart (use original sentences plus one deletion variant each). Outputs: raw agreement and Cohen's kappa for A vs A2, A vs B, A vs C. eps\_m = 1 minus agreement(A, A2). Gate G1: record eps\_m per model. If agreement(A, A2) is below 0.90 the model is flagged noisy and analysed on stable rows only. If agreement(A, C) is more than 0.02 below agreement(A, A2), single-sentence results (E12) must not be compared to batch results and this is stated in the report.

### E1 Pipeline validation on ground truth (dev, no LLM except the planted-cue step)

(a) Unit tests with a mock black box (for example: label Positive iff the token bhalo is present). Occlusion and coalition attribution must recover the planted token. Coalition attribution must match exact Shapley enumeration for n up to 8 within 1e-6 when all coalitions are enumerated. (b) Fit TF-IDF Logistic Regression on the train split. Treat it as a hard-label oracle. Compute pipeline attributions on 300 dev rows with at least 4 tokens, and compare with the exact per-token probability drop from predict\_proba. (c) Planted-cue test on Qwen3.8-27B: insert bhalo into 100 neutral dev rows and kharap into 100 others at random positions; measure how often the planted token has the top attribution among rows where the label changed. Gate G2: LR median per-sentence Spearman at least 0.6 and top-1 agreement at least 0.7; planted-cue top-1 rate at least 0.8. If failed, raise M (up to 192) and retest; if still failing, report the pipeline limit and restrict claims to occlusion results.

### E2 Coalition budget and masking sensitivity (dev)

Pick M by the rule in section 3. Compare delete vs \[MASK\] attributions on dev (top-3 overlap) for information only; delete stays primary.

### E3 Baseline labels (XAI-800)

3 replicate passes with P\_batch. Output y0, unstable flag, accuracy and per-class recall (must match the earlier test numbers within the noise floor; investigate if not).

### E4 Word occlusion (XAI-800, stable rows)

One variant per token (delete token i). Output f\_i = 1 if the label differs from y0.

### E5 Coalition attribution (XAI-200, stable rows)

Sample M coalitions: size s drawn with probability proportional to (n-1)/(s(n-s)), subset uniform given size, always paired with its complement, de-duplicated; enumerate all if 2^n minus 2 is at most M. Exclude the empty and full coalitions from queries. Fit weighted ridge regression with intercept (alpha 0.01, fixed) of v on the binary mask, with Shapley-kernel weights. phi\_i is the coefficient. Run two independent seeds to measure stability.

### E6 Self-rationale (XAI-800)

New prompt prompts/xai/rationale\_extractive\_v1.txt plus a batch format: given the text and the model's predicted label, return up to 3 words copied exactly from the text that most determined the label, one line per item. Freeze the prompt on dev using mechanical criteria only (at least 95 percent parsed, at least 90 percent of words found verbatim in the text); never use faithfulness results to edit it. Match words to token indices; record the match rate. Unmatched words are dropped and reported.

### E7 Faithfulness evaluation (XAI-200, plus rationale on XAI-800 where available)

Evidence sources: coalition top-k, occlusion top-k, rationale (own size), random (5 draws), last-k tokens (position baseline). For each, build variants: remove the evidence set (comprehensiveness) and keep only the evidence set (sufficiency). Ties in ranking break by occlusion score then by position.

### E8 Targeted counterfactuals (XAI-800, only rows where the operator applies)

All operators are deterministic and fixed in advance.

- C1 contrast-marker deletion: delete tokens in the strict marker list taken from src/audit/definition\_compliance.py (reuse the list verbatim).
- C2 clause-order swap: split at the marker and swap the two clauses (expected: label invariant).
- C3 clause deletion: for rows with a strict marker and at baseline Mixed, delete the left clause, then separately the right clause; record both labels.
- C4 antonym swap: swap one token from a fixed polar lexicon of about 20 pairs (bhalo/kharap, valo/baje, good/bad and similar), for rows with exactly one hit and gold Positive or Negative.
- C5 negation deletion: delete the tokens nai or nei in rows that contain them. Each operator also records how many rows qualified.

### E9 Prompt and demonstration attribution (XAI-800, three full-analysis models)

Run the elaborate prompt zero\_shot\_v1 and the 8 ablated prompts (drop rule1..7, drop mixed definition) under P\_batch. For few-shot, run five\_shot\_minimal\_v1 and its leave-one-demonstration-out versions (5 files created under prompts/xai/ by deleting one demonstration each, otherwise byte-identical). Output per-prompt accuracy, per-class recall and per-sentence correctness changes.

### E10 Language and position analysis (exploratory)

Tag each token as EN, BN-roman or OTHER (numbers, punctuation) with a frozen tagging prompt run in context via the existing runner on a free model. Gate G4: hand-label 300 tokens (human task, section 8); tagging accuracy at least 0.90, otherwise this analysis is reported as unvalidated. Compute attribution mass share by tag and by relative-position tercile, with a permutation test against shuffled positions.

### E11 Plausibility (human task)

See section 8. Compare model evidence with human evidence by token-level F1.

### E12 Optional white-box arm (requires a GPU; skip cleanly if absent)

Llama-3.1-8B-Instruct, 4-bit, local. Single-sentence prompt built from the same primary prompt file. Score each label by next-token logits restricted to the four label tokens. Integrated Gradients (Captum, 50 steps, pad-embedding baseline) over input embeddings; aggregate subword scores to tokens by sum. Check that local predictions agree with the API predictions on dev (at least 0.90), else label results as local-model results. Compare with E4/E5 by Spearman and top-3 overlap, and compute soft comprehensiveness on logit probabilities.

### E13 Gemini restricted replication

E3 (one pass plus a 100-row second pass for the noise floor), E4, E6, E8 on the 300-row subset. No coalition attribution (cost).

## 5. Metrics and statistics

- Comprehensiveness at k: share of sentences where removing the evidence set changes the label. Sufficiency at k: share where keeping only the evidence set keeps y0. Also report AOPC over k = 1..3. Exclude sentences with fewer than k plus 1 tokens for that k.
- Faithfulness gap: evidence source minus random baseline (paired per sentence, random averaged over its 5 draws).
- Stability: top-3 Jaccard and Kendall tau between two coalition seeds.
- Agreement between methods and models: Spearman on per-token scores, top-3 Jaccard.
- Counterfactual effect: rate of the expected label change, minus the noise-floor rate on the same rows, with a bootstrap CI.
- Intervals: bootstrap over source sentences (10,000 resamples, seeded), stratified by gold class where class-level results are reported. Report per class always; Mixed has the smallest n.
- Tests: paired bootstrap for P1 to P4; McNemar for P5. Apply Holm correction within each family (P1, P2, P3, P4, P5 separately). Report effect sizes with CIs, not only p-values.
- Qualitative examples are selected by a rule fixed in advance (random with the seed, plus per-metric median and worst case). Never hand-pick.

## 6. Repository layout and schemas

src/xai/: config.py (seeds, sizes, k, paths), sampling.py (builds samples and writes SHA256), tokenize.py (tokens, spans, delete and \[MASK\] operations), variants.py (generators for occlusion, coalitions, evidence ablations, counterfactual operators), oracle.py (VariantRunner built on build\_sender and classify\_batch; enforces P\_batch; JSONL resume keyed by variant\_id; a mock sender for offline tests), attribution.py (occlusion, KernelSHAP-style fit, top-k), rationale.py, faithfulness.py, counterfactual.py, langtag.py, stats.py, gates.py, whitebox/ig.py (optional). tests/xai/: pytest suite using the mock oracle (planted-cue recovery, exact-Shapley match for small n, batching rule, resume, masking edge cases such as single-token rows and repeated tokens). prompts/xai/: rationale\_extractive\_v1.txt, langtag\_v1.txt, their batch formats, leave-one-demo-out prompts. Notebooks: 09\_xai\_validity.ipynb, 10\_xai\_runs.ipynb, 11\_xai\_results.ipynb, optional 12\_xai\_whitebox.ipynb. Notebooks only call src/xai functions. outputs/xai/: samples/, variants/, labels/, attributions/, results/, figures/, annotation/, PREREG.md, DEVIATIONS.md, OPEN\_QUESTIONS.md, dev\_gate.json, REPORT.md.

Variant record (one JSON per line): {variant\_id, src\_index, kind, mask (list of token indices removed or kept, per kind), rep, text, prediction, provider, model, prompt}. src\_index is the row position in the full dataset, as in the existing runner. variant\_id is src\_index:kind:hash8 of the mask.

## 7. Phases, gates and execution order

Phase 0 Scaffold and offline tests. Phase 1 Dev: E0, E1, E2, E6 prompt freeze, E10 tagging prompt freeze. Phase 2 Freeze: write outputs/xai/PREREG.md containing the hypothesis list, SHA256 of the sample files, config and every prompt file used, and the git commit; write outputs/xai/dev\_gate.json with G1 and G2 results (and G4 if the language analysis will be reported). Phase 3 Test runs. Phase 4 Analysis and report. Phase 5 optional E12.

Anti-peeking mechanism: gates.py exposes check\_prereg(). Any command with scope test calls it first and refuses to run if dev\_gate.json is missing, if G1 or G2 is not recorded, or if a hashed file differs from PREREG.md. Dev-phase code may not read test labels except through the frozen sample file.

Test-phase order (each step is independently valuable, so if quota runs out stop after the last completed step): E3, E4, E8, E6, E5, E7, E9, E10, E13, E12.

Budget per full-analysis model, in classified rows: E3 2.4k; E4 about 9k; E5 about 19k at M of 96; E7 about 8k; E6 0.8k; E8 about 5k; E9 about 11k. Total about 55k, roughly 2.8 times one full-dataset pass. Gemini subset about 7k. Dev costs about 15k extra per model. Run free providers first; run Gemini last.

## 8. Human tasks (cannot be automated; the agent prepares templates and scoring scripts)

1. Review and sign off the polar antonym lexicon, the negation tokens and the contrast-marker list before the freeze.
2. Hand-label 300 tokens for language tags (EN, BN-roman, OTHER), drawn by seed from the dev tokens, for gate G4.
3. Annotate evidence tokens (the minimal words that justify the gold label) for 120 sentences (30 per class) from XAI-800; a second annotator labels a 40-sentence overlap so Cohen's kappa is reported. Annotators do not see model outputs. Provide CSV templates in outputs/xai/annotation/ and scripts that validate and score them.

## 9. Threats to validity and mitigations

| Threat | Mitigation |
| --- | --- |
| Batch context changes labels | E0, P\_batch, one-variant-per-source rule |
| Deleted words make unnatural text | \[MASK\] sensitivity; claims called robust only if they hold under both maskings |
| Hard labels give coarse signal | coalition averaging; logprobs only where a provider exposes them (not required) |
| Model nondeterminism at temperature 0 | noise floor, 3 replicates, unstable-row exclusion |
| Self-rationales are post-hoc | evaluated by intervention, never taken at face value |
| Small Mixed n | report per class with CIs; never pool away Mixed |
| Intra-word switching (for example movietar) and spelling variants | whitespace tokens by design; limitation stated; no token merging |
| Hosted model drift | record model id and date; complete each model's runs in one sitting where quota allows |
| Tagging errors | G4 validation gate |
| Selective reporting | PREREG.md, fixed example-selection rule, DEVIATIONS.md |

## 10. Deliverables and definition of done

REPORT.md with fixed sections: setup and instrument checks (E0, E1 results), hypotheses table with outcome (supported, not supported, inconclusive) for P1 to P5, per-model faithfulness table with CIs, Mixed analysis (E8 C1 to C3), prompt-component results (E9), exploratory results labelled as such, limitations, deviations. Figures: faithfulness gap by model, comprehensiveness and sufficiency curves, Mixed transition matrices, attribution mass by language tag and position, prompt-ablation deltas.

Done means: all tests pass; PREREG.md and dev\_gate.json exist and match the run; every reported number traces to a complete run file; all hypotheses have a stated outcome including negative or inconclusive ones; DEVIATIONS.md is complete; the report reproduces from the notebooks with no manual edits.
