# MixLnSent

**MixLnSent** is a research codebase for fine-tuning and evaluating transformer
models on Bengali and code-mixed language tasks. It includes experiments for
sentiment analysis, sarcasm and humor detection, and hate-speech detection.

## Experiments

| Experiment | Task | Dataset |
| --- | --- | --- |
| `BnSENTMIX` | Sentiment classification, including few-shot and full fine-tuning, hyperparameter tuning, and soft-voting ensembling | [BnSentMix](https://huggingface.co/datasets/aplycaebous/BnSentMix) |
| `sentmix3l_results` | Bangla-Hindi-English code-mixed sentiment classification | SentMix-3L (`sen_1k.csv`, supplied locally) |
| `mixsarc_results` | Multi-label humor, sarcasm, offense, and vulgarity classification | [MixSarc](https://huggingface.co/datasets/ajwad-abrar/MixSarc) |
| `banth_results` | Two-stage transliterated Bangla hate-speech and target-group classification | [BanTH](https://huggingface.co/datasets/aplycaebous/BanTH) |

The experiments compare multilingual and Bengali-focused transformer models,
including mBERT, XLM-RoBERTa, BanglaBERT, BanglishBERT, DistilBERT, and BERT.
The available models and evaluation settings vary by experiment.

## Repository layout

```text
.
├── BnSENTMIX/
│   ├── Code/bnsentmix_finetuning.py
│   ├── Results/
│   └── Results_Graph/
├── banth_results/
│   ├── Code/banth_finetuning.py
│   ├── Results/
│   └── Results_Graph/
├── mixsarc_results/
│   ├── Code/mixsarc_finetuning.py
│   ├── Results/
│   └── Results_Graph/
└── sentmix3l_results/
    ├── Code/sentmix3l_finetuning.py
    ├── Results/
    └── Results_Graph/
```

Each experiment directory contains its training/evaluation script and, where
committed, result tables, experiment logs, saved state, and visualizations.

## Running an experiment

The scripts are exported notebook-style files and include IPython/Colab
installation commands. Run them in Google Colab or a Jupyter environment that
supports IPython magics; they are not plain standalone Python scripts.

1. Open the script for the experiment you want to run in a notebook environment.
2. Use a GPU runtime for practical model fine-tuning.
3. For BnSentMix, MixSarc, and BanTH, the script loads the dataset from Hugging
   Face Datasets, so the runtime needs internet access.
4. For SentMix-3L, place `sen_1k.csv` in the notebook's working directory or
   upload it when prompted. The dataset is not included in this repository.
5. Run the notebook cells from top to bottom. The scripts install their Python
   dependencies and download pretrained model weights as needed.

Training time and memory use depend on the selected model, experiment settings,
and available hardware. Check each script for its seeds, splits, task-specific
metrics, and exact training configuration.

## Results

The experiment folders include CSV result summaries and experiment logs, along
with plots such as comparative macro-F1 charts, per-label metrics, confusion
matrices, and sample-efficiency analyses. Some experiments also save serialized
state files. Rerunning an experiment may overwrite or create output files in
the notebook's current working directory.

For imbalanced datasets, consult the task-specific metrics in addition to
accuracy. The scripts report metrics such as macro-F1 and per-class/per-label
scores; multi-label tasks also report metrics such as subset accuracy.

## Data, models, and citation

Dataset and pretrained-model terms are set by their respective owners. Review
the dataset cards, model cards, and applicable licenses before using or
redistributing their data or weights. The SentMix-3L CSV must be obtained
separately from its authorized source.

When using this repository in research, cite the relevant dataset and model
publications and identify the experiment script and configuration used. See
the script headers for the dataset references recorded by each experiment.

## License

No repository license is specified. Unless a license is added, the repository
contents should not be assumed to be available for redistribution or reuse.
