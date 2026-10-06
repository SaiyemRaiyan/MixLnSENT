from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = PROJECT_ROOT / "prompts"
FILTERING_PROMPTS_DIR = PROMPTS_DIR / "filtering"
PROMPTING_PROMPTS_DIR = PROMPTS_DIR / "prompting"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FILTERING_OUTPUTS_DIR = OUTPUTS_DIR / "filtering"
PROMPTING_OUTPUTS_DIR = OUTPUTS_DIR / "prompting"


def prompting_dir(dataset, kind):
    """Output directory for one dataset: kind is predictions, metrics or results.

    The dataset name is part of the path rather than the filename, so runs on
    different datasets cannot collide however similar their run identity looks.
    """
    return PROMPTING_OUTPUTS_DIR / dataset / kind


SEED = 42
DEV_SIZE = 600
SCOPES = ("dev", "train", "validation", "test", "full")
LABEL_MAPPING = {
    0: "Positive",
    1: "Negative",
    2: "Neutral",
    3: "Mixed",
}


# Table 4 of Alam et al. (2025), weighted Precision/Recall/F1, for direct comparison.
PUBLISHED_BASELINES = {
    "logistic-regression": {
        "paper_name": "Logistic Regression",
        "group": "Machine Learning Models",
        "validation": {"accuracy": 0.668, "precision": 0.656, "recall": 0.668, "f1": 0.662},
        "test": {"accuracy": 0.667, "precision": 0.614, "recall": 0.667, "f1": 0.639},
    },
    "random-forest": {
        "paper_name": "Random Forest",
        "group": "Machine Learning Models",
        "validation": {"accuracy": 0.672, "precision": 0.661, "recall": 0.672, "f1": 0.666},
        "test": {"accuracy": 0.648, "precision": 0.635, "recall": 0.648, "f1": 0.641},
    },
    "svm": {
        "paper_name": "SVM",
        "group": "Machine Learning Models",
        "validation": {"accuracy": 0.694, "precision": 0.676, "recall": 0.694, "f1": 0.685},
        "test": {"accuracy": 0.660, "precision": 0.637, "recall": 0.660, "f1": 0.648},
    },
    "rnn": {
        "paper_name": "RNN",
        "group": "Recurrent Neural Network Variants",
        "validation": {"accuracy": 0.406, "precision": 0.308, "recall": 0.406, "f1": 0.350},
        "test": {"accuracy": 0.401, "precision": 0.352, "recall": 0.401, "f1": 0.375},
    },
    "lstm": {
        "paper_name": "LSTM",
        "group": "Recurrent Neural Network Variants",
        "validation": {"accuracy": 0.678, "precision": 0.670, "recall": 0.678, "f1": 0.674},
        "test": {"accuracy": 0.670, "precision": 0.657, "recall": 0.670, "f1": 0.663},
    },
    "xlm-roberta": {
        "paper_name": "XLM-RoBERTa",
        "group": "Multilingual Language Models",
        "validation": {"accuracy": 0.726, "precision": 0.709, "recall": 0.726, "f1": 0.717},
        "test": {"accuracy": 0.698, "precision": 0.642, "recall": 0.698, "f1": 0.669},
    },
    "mbert": {
        "paper_name": "mBERT",
        "group": "Multilingual Language Models",
        "validation": {"accuracy": 0.726, "precision": 0.713, "recall": 0.726, "f1": 0.719},
        "test": {"accuracy": 0.694, "precision": 0.675, "recall": 0.694, "f1": 0.684},
    },
    "banglabert": {
        "paper_name": "BanglaBERT",
        "group": "Bangla Language Models",
        "validation": {"accuracy": 0.721, "precision": 0.668, "recall": 0.721, "f1": 0.693},
        "test": {"accuracy": 0.698, "precision": 0.642, "recall": 0.698, "f1": 0.669},
    },
    "banglishbert": {
        "paper_name": "BanglishBERT",
        "group": "Bangla Language Models",
        "validation": {"accuracy": 0.694, "precision": 0.715, "recall": 0.694, "f1": 0.704},
        "test": {"accuracy": 0.686, "precision": 0.653, "recall": 0.686, "f1": 0.669},
    },
    "distilbert": {
        "paper_name": "DistilBERT",
        "group": "English Language Models",
        "validation": {"accuracy": 0.701, "precision": 0.694, "recall": 0.701, "f1": 0.697},
        "test": {"accuracy": 0.672, "precision": 0.665, "recall": 0.672, "f1": 0.668},
    },
    "bert": {
        "paper_name": "BERT",
        "group": "English Language Models",
        "validation": {"accuracy": 0.727, "precision": 0.710, "recall": 0.724, "f1": 0.717},
        "test": {"accuracy": 0.695, "precision": 0.683, "recall": 0.694, "f1": 0.688},
    },
}


# SentMix-3L publishes weighted F1 only, so it has its own table rather than reusing
# the four-metric shape above. Raihan et al. (2023), trained on synthetic data and
# tested on the natural set.
SENTMIX3L_BASELINES = {
    "gpt-3.5-turbo": {"paper_name": "GPT-3.5 Turbo", "weighted_f1": 0.62},
    "xlm-r": {"paper_name": "XLM-R", "weighted_f1": 0.59},
    "banglishbert": {"paper_name": "BanglishBERT", "weighted_f1": 0.56},
    "mbert": {"paper_name": "mBERT", "weighted_f1": 0.56},
    "bert": {"paper_name": "BERT", "weighted_f1": 0.55},
    "roberta": {"paper_name": "roBERTa", "weighted_f1": 0.54},
    "muril": {"paper_name": "MuRIL", "weighted_f1": 0.54},
    "indicbert": {"paper_name": "IndicBERT", "weighted_f1": 0.53},
    "distilbert": {"paper_name": "DistilBERT", "weighted_f1": 0.53},
    "hindibert": {"paper_name": "HindiBERT", "weighted_f1": 0.48},
    "hingbert": {"paper_name": "HingBERT", "weighted_f1": 0.47},
    "banglabert": {"paper_name": "BanglaBERT", "weighted_f1": 0.47},
}
