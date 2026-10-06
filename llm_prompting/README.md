# LLM prompting (zero-shot, no training)

Prompt-engineering results for the same datasets this benchmark fine-tunes on. The
experiments here train nothing: a frozen model is asked to label each sentence, and the
same prompt is reused across every model and corpus so the comparison is like for like.

## Experiments

| Dataset | Rows | Task | Labels |
|---|---|---|---|
| BnSentMix | 3,003 (test) | single-label sentiment | Positive, Negative, Neutral, Mixed |
| SentMix-3L | 1,007 (test-only) | single-label sentiment | Positive, Negative, Neutral |
| MixSarc | 1,364 (test) | multi-label implicit meaning | Humorous, Sarcastic, Offensive, Vulgar |

Three conditions are run on every model: zero-shot, two-shot and five-shot, counted as
examples *per label*, so two-shot is 8 examples across four labels and five-shot is 20.
A second prompt family carries full label definitions and lexicon cues, against a
minimal family that states the task in a single line.

## Layout

```
llm_prompting/
├── Code/                 the pipeline and the notebooks that drive it
├── Results/              one CSV per dataset and model
└── Results_Graph/        per-dataset charts
```

## Running it

The pipeline is provider-agnostic: one variable selects provider and model, and adding
a model means adding one entry in `src/prompting/models.py`.

```bash
python -m src.common.fetch_data                      # SentMix-3L CSV (see below)
python run_study.py mixsarc test "provider:model" "zero_shot_v1"
```

Every run is checkpointed per batch, so an interrupted run resumes from the last saved
row rather than restarting.

## Notes and caveats

**SentMix-3L is not included.** The corpus is distributed as a CSV from its own source;
`python -m src.common.fetch_data` retrieves it into `data/`.

**MixSarc's split is reconstructed.** The corpus paper reports a 70:15:15 partition
stratified on humor but releases neither the partition nor its seed. The split here is
rebuilt the same way under a fixed seed, so it is the same construction rather than the
same rows; test-set prevalence matches closely (humor 55.7%, sarcasm 24.4%, offense
4.0%, vulgarity 5.4%).

**The MixSarc prompt is the published one**, with the same definitions and JSON
contract. Only the packing differs: the published prompt labels one sentence per
request, so batching numbers the sentences to keep the request count in the hundreds
rather than the thousands.

**Some models answer empty.** On MixSarc the gpt-oss pair and ALLaM-7B return the
all-zero vector on 88-99% of rows, which drives every per-label score to near zero
while exact-match stays near the rate of unlabelled rows (21.5%). This was tested three
ways -- unbatched, with a different format example, and with reasoning enabled -- and
holds in all three, so it is reported rather than treated as a harness fault.

**Two accuracy figures that disagree are both reported.** On MixSarc, exact-match
accuracy and macro F1 rank models differently, because one rewards confidence and the
other rewards coverage of rare labels.

## Provider keys

Credentials are read from a `.env` file that is not committed. Several keys per
provider are supported as `PROVIDER_API_KEY_1`, `_2`, ... and rotation advances on a
quota failure so long runs proceed unattended.
