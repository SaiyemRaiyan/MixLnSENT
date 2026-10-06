"""EXPERIMENT 1 - FILTERING: downstream sentiment classifiers.

Model A trains on the full training split, model B on the prompt-filtered
subset. Both are scored on identical evaluation sets.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from ..common.config import SEED


def _tfidf():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2)


def logistic_regression():
    return Pipeline([
        ("tfidf", _tfidf()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=SEED)),
    ])


def random_forest():
    return Pipeline([
        ("tfidf", _tfidf()),
        ("classifier", RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)),
    ])


def svm():
    return Pipeline([
        ("tfidf", _tfidf()),
        ("classifier", LinearSVC(random_state=SEED)),
    ])


CLASSICAL_MODELS = {
    "logistic-regression": logistic_regression,
    "random-forest": random_forest,
    "svm": svm,
}

# Registered for the GPU stage; each needs PyTorch fine-tuning.
TRANSFORMER_MODELS = {
    "distilbert": "distilbert-base-uncased",
    "bert": "bert-base-uncased",
    "mbert": "bert-base-multilingual-cased",
    "xlm-roberta": "xlm-roberta-base",
    "banglabert": "csebuetnlp/banglabert",
}
