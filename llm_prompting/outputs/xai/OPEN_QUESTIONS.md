# Open questions and conservative decisions

1. **Few-shot demonstration count:** the filename and plan say “five-shot”, but the
   actual frozen source prompt has 20 examples. The conservative implementation
   preserves all 20 and generates 20 leave-one-out variants. A different choice
   requires a dated entry in `DEVIATIONS.md` before test runs.
2. **Development neutral count:** 600 stratified dev rows do not provide 200
   Neutral examples for the disjoint planted-cue groups. The implementation samples
   additional Neutral rows from the remaining training partition only; the manifest
   records how many were needed.
3. **Human sign-off and G4:** the antonym/negation/marker sheet, evidence annotations,
   and 300-token language-tag annotations require people. No entries or quality
   scores are fabricated. Test preregistration requires completed operator sign-off.
4. **Language/position claims:** until the human tag sheet is scored and reaches the
   0.90 threshold, all such results must be described as unvalidated exploratory
   results or omitted.
5. **White-box E12:** the optional local IG arm requires a CUDA GPU plus compatible
   quantization/Captum dependencies. Hosted hard-label runs remain the primary,
   complete method; E12 is not required for the definition of done.
6. **Execution status:** train-only development preparation is complete. G1 was
   measured for Qwen3.8-27B and GPT-OSS-20B; the Hugging Face Llama calibration
   stopped before producing rows because the Inference Providers API returned HTTP
   402 (no remaining credits). The local LR-oracle E1 gate also failed both
   thresholds (median Spearman 0.00; top-1 agreement 0.257). No planted-cue gate,
   rationale freeze, operator sign-off, preregistration, or test-scope inference
   has been completed. Do not run test scope unless all preregistered gates pass.
