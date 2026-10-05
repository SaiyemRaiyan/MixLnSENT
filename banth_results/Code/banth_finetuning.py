# -*- coding: utf-8 -*-
"""BanTH_Finetuning

# BanTH: Two-Stage Transliterated Bangla Hate Speech Detection

Dataset: `aplycaebous/BanTH` (Haider, Shifat, Ishmam, Barua, Sourove, Fahim, Alam --
Findings of NAACL 2025), loaded directly from Hugging Face.
"""

!pip -q install -U transformers datasets accelerate scikit-learn sentencepiece pandas numpy matplotlib

import os, re, json, pickle, shutil, random
import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
from datasets import load_dataset, Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                           TrainingArguments, Trainer, EarlyStoppingCallback, set_seed)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
                              confusion_matrix, multilabel_confusion_matrix, hamming_loss)

SEED = 100
set_seed(SEED)
random.seed(SEED); np.random.seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

experiment_log = []
def log_experiment(model_name, stage, setting, seed, config, metrics):
    experiment_log.append({"model": model_name, "stage": stage, "setting": setting, "seed": seed,
                            "config": config, "metrics": metrics})

"""## 1. Load Dataset (from Hugging Face, using the paper's own pre-made splits)"""

raw = load_dataset("aplycaebous/BanTH")
print(raw)
split_keys = list(raw.keys())
train_key = "train" if "train" in split_keys else split_keys[0]
val_key = "val" if "val" in split_keys else ("validation" if "validation" in split_keys else split_keys[1])
test_key = "test" if "test" in split_keys else split_keys[-1]
print("Using splits:", train_key, val_key, test_key)

train_df_raw = raw[train_key].to_pandas()
val_df_raw = raw[val_key].to_pandas()
test_df_raw = raw[test_key].to_pandas()
print(train_df_raw.shape, val_df_raw.shape, test_df_raw.shape)
train_df_raw.head()

"""## 2. Detect Columns: Text, Binary Hate Label, 7 Target-Group Labels
"""

def detect_columns(df):
    text_col = [c for c in df.columns if pd.api.types.is_string_dtype(df[c]) and df[c].astype(str).str.len().mean() > 10][0]
    binary_cols = [c for c in df.columns if c != text_col and set(pd.Series(df[c]).dropna().unique()).issubset({0, 1})]
    hate_col = next((c for c in binary_cols if c.strip().lower() in ("label", "hate", "is_hate", "binary_label")), binary_cols[0])
    target_cols = [c for c in binary_cols if c != hate_col]
    return text_col, hate_col, target_cols

text_col, hate_col, target_cols = detect_columns(train_df_raw)
print("text_col:", text_col, "| hate_col:", hate_col, "| target_cols:", target_cols)

def clean(df):
    d = df[[text_col, hate_col] + target_cols].dropna(subset=[text_col, hate_col]).reset_index(drop=True)
    d = d.rename(columns={text_col: "text"})
    for c in target_cols:
        d[c] = d[c].fillna(0).astype(int)
    d[hate_col] = d[hate_col].astype(int)
    return d

train_df = clean(train_df_raw)
val_df = clean(val_df_raw)
test_df = clean(test_df_raw)
print(len(train_df), len(val_df), len(test_df))

"""## 3. Dataset Diagnostics
"""

full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
hate_counts = full_df[hate_col].value_counts().sort_index()
print("Hate-label distribution (0=Non-Hate, 1=Hate):\n", hate_counts)
majority_baseline_acc = hate_counts.loc[0] / hate_counts.sum()
print(f"\nDummy all-non-hate classifier accuracy: {majority_baseline_acc*100:.2f}% "
      f"(paper reports 72.73%) -- any model/setting scoring below this has learned nothing useful.")

hate_only_full = full_df[full_df[hate_col] == 1]
target_counts = hate_only_full[target_cols].sum().sort_values(ascending=False)
target_pct = (target_counts / len(hate_only_full) * 100).round(2)
print(f"\nTarget-group distribution within {len(hate_only_full)} hate-labeled rows:")
print(pd.DataFrame({"count": target_counts, "pct_of_hate_rows": target_pct}))

"""## 4. Co-occurrence Heatmap
"""

cooc = pd.DataFrame(0, index=target_cols, columns=target_cols, dtype=int)
for c1 in target_cols:
    for c2 in target_cols:
        cooc.loc[c1, c2] = int(((hate_only_full[c1] == 1) & (hate_only_full[c2] == 1)).sum())

plt.rcParams.update({"font.family": "serif", "font.size": 11, "figure.dpi": 300})
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cooc.values, cmap="Purples")
ax.set_xticks(range(len(target_cols))); ax.set_xticklabels(target_cols, rotation=45, ha="right")
ax.set_yticks(range(len(target_cols))); ax.set_yticklabels(target_cols)
for i in range(len(target_cols)):
    for j in range(len(target_cols)):
        v = cooc.values[i, j]
        ax.text(j, i, str(v), ha="center", va="center", fontsize=8,
                color="white" if v > cooc.values.max() / 2 else "black")
ax.set_title("Co-occurrence of Multi-Label Hate Targets (re-derived from loaded data)")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig("banth_cooccurrence.pdf", bbox_inches="tight")
plt.savefig("banth_cooccurrence.png", dpi=300, bbox_inches="tight")
plt.show()

"""## 5. Metrics
"""

def compute_metrics_binary(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    bal_acc = balanced_accuracy_score(labels, preds)  # paper's "Macro-Accuracy"
    p_m, r_m, f1_m, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    return {"accuracy": acc, "macro_accuracy": bal_acc, "precision_macro": p_m, "recall_macro": r_m, "f1_macro": f1_m}

def make_compute_metrics_multilabel(label_cols, threshold=0.5):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = 1 / (1 + np.exp(-logits))
        preds = (probs >= threshold).astype(int)
        labels = labels.astype(int)

        p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
        p_mic, r_mic, f1_mic, _ = precision_recall_fscore_support(labels, preds, average="micro", zero_division=0)
        subset_acc = float(np.mean(np.all(labels == preds, axis=1)))
        hloss = hamming_loss(labels, preds)

        out = {"precision_macro": p_mac, "recall_macro": r_mac, "f1_macro": f1_mac,
               "precision_micro": p_mic, "recall_micro": r_mic, "f1_micro": f1_mic,
               "subset_accuracy": subset_acc, "accuracy": subset_acc, "hamming_loss": hloss}
        p_c, r_c, f1_c, _ = precision_recall_fscore_support(labels, preds, average=None, zero_division=0,
                                                              labels=list(range(len(label_cols))))
        for i, lbl in enumerate(label_cols):
            out[f"f1_{lbl}"] = f1_c[i]
            out[f"precision_{lbl}"] = p_c[i]
            out[f"recall_{lbl}"] = r_c[i]
        return out
    return compute_metrics

compute_metrics_multilabel = make_compute_metrics_multilabel(target_cols)

def metrics_row_binary(model_name, setting, seed, m, extra=None):
    row = {"model": model_name, "setting": setting, "seed": seed, "accuracy": m["eval_accuracy"],
           "macro_accuracy": m["eval_macro_accuracy"], "precision_macro": m["eval_precision_macro"],
           "recall_macro": m["eval_recall_macro"], "f1_macro": m["eval_f1_macro"]}
    if extra: row.update(extra)
    return row

def metrics_row_multilabel(model_name, setting, seed, m, label_cols, extra=None):
    row = {"model": model_name, "setting": setting, "seed": seed, "subset_accuracy": m["eval_subset_accuracy"],
           "hamming_loss": m["eval_hamming_loss"], "precision_macro": m["eval_precision_macro"],
           "recall_macro": m["eval_recall_macro"], "f1_macro": m["eval_f1_macro"],
           "f1_micro": m["eval_f1_micro"]}
    for lbl in label_cols:
        row[f"f1_{lbl}"] = m[f"eval_f1_{lbl}"]
        row[f"precision_{lbl}"] = m[f"eval_precision_{lbl}"]
        row[f"recall_{lbl}"] = m[f"eval_recall_{lbl}"]
    if extra: row.update(extra)
    return row

"""## 6. Models """

MODEL_LIST = {
    "mBERT": "bert-base-multilingual-cased",
    "XLM-RoBERTa": "xlm-roberta-base",
    "BanglaBERT": "csebuetnlp/banglabert",
    "BanglishBERT": "csebuetnlp/banglishbert",
    "DistilBERT": "distilbert-base-multilingual-cased",
    "BERT": "bert-base-uncased",
}
MAX_LEN = 128

"""## 7. Fine-Tuning Functions (Stage A: binary, Stage B: multi-label)
"""

def freeze_base_layers(model, unfreeze_last_n=2):
    for _, p in model.named_parameters():
        p.requires_grad = False
    for n, p in model.named_parameters():
        if "classifier" in n or "pooler" in n:
            p.requires_grad = True
    layer_ids = {int(m.group(1)) for n, _ in model.named_parameters()
                 for m in [re.search(r"\.layer\.(\d+)\.", n)] if m}
    if layer_ids:
        keep = set(range(max(layer_ids) - unfreeze_last_n + 1, max(layer_ids) + 1))
        for n, p in model.named_parameters():
            m = re.search(r"\.layer\.(\d+)\.", n)
            if m and int(m.group(1)) in keep:
                p.requires_grad = True
    return model

def _common_args(out_dir, epochs, lr, bs, weight_decay, warmup_steps, patience, metric_name):
    return TrainingArguments(
        output_dir=out_dir, num_train_epochs=epochs,
        per_device_train_batch_size=bs, per_device_eval_batch_size=bs,
        learning_rate=lr, weight_decay=weight_decay, warmup_steps=warmup_steps,
        eval_strategy="epoch", save_strategy="epoch" if patience else "no",
        save_total_limit=1, load_best_model_at_end=bool(patience),
        metric_for_best_model=metric_name, greater_is_better=True,
        logging_steps=25, report_to="none", seed=SEED,
    )

def tokenize_binary(ds, tokenizer, hate_col, max_len):
    return ds.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=max_len), batched=True)

def finetune_binary(model_name, train_ds, val_ds, test_ds, hate_col, epochs=5, lr=2e-5, bs=16,
                     tag="full", seed=SEED, keep_model=False, weight_decay=0.01,
                     label_smoothing_factor=0.0, freeze_base=False, unfreeze_last_n=2, patience=None):
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(DEVICE)
    if freeze_base:
        model = freeze_base_layers(model, unfreeze_last_n)

    def _map(ex):
        enc = tokenizer(ex["text"], truncation=True, padding="max_length", max_length=MAX_LEN)
        enc["labels"] = int(ex[hate_col])
        return enc
    train_tok, val_tok, test_tok = train_ds.map(_map), val_ds.map(_map), test_ds.map(_map)

    steps_per_epoch = max(1, len(train_tok) // bs)
    warmup_steps = max(1, int(0.1 * steps_per_epoch * epochs))
    out_dir = f"./out_{model_name.replace('/','_')}_{tag}_{seed}"
    args = _common_args(out_dir, epochs, lr, bs, weight_decay, warmup_steps, patience, "f1_macro")
    args.label_smoothing_factor = label_smoothing_factor
    callbacks = [EarlyStoppingCallback(early_stopping_patience=patience)] if patience else []
    trainer = Trainer(model=model, args=args, train_dataset=train_tok, eval_dataset=val_tok,
                       compute_metrics=compute_metrics_binary, callbacks=callbacks)
    trainer.train()
    test_metrics = trainer.evaluate(test_tok)
    test_pred = trainer.predict(test_tok)
    y_true, y_pred = test_pred.label_ids, np.argmax(test_pred.predictions, axis=-1)

    if os.path.exists(out_dir): shutil.rmtree(out_dir, ignore_errors=True)
    if keep_model:
        return test_metrics, trainer.state.log_history, (model, tokenizer), (y_true, y_pred)
    del model; torch.cuda.empty_cache()
    return test_metrics, trainer.state.log_history, None, (y_true, y_pred)

def finetune_multilabel(model_name, train_ds, val_ds, test_ds, label_cols, epochs=5, lr=2e-5, bs=16,
                         tag="full", seed=SEED, keep_model=False, weight_decay=0.01,
                         freeze_base=False, unfreeze_last_n=2, patience=None):
    set_seed(seed)
    num_labels = len(label_cols)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels, problem_type="multi_label_classification").to(DEVICE)
    if freeze_base:
        model = freeze_base_layers(model, unfreeze_last_n)

    def _map(ex):
        enc = tokenizer(ex["text"], truncation=True, padding="max_length", max_length=MAX_LEN)
        enc["labels"] = [float(ex[c]) for c in label_cols]
        return enc
    train_tok, val_tok, test_tok = train_ds.map(_map), val_ds.map(_map), test_ds.map(_map)

    steps_per_epoch = max(1, len(train_tok) // bs)
    warmup_steps = max(1, int(0.1 * steps_per_epoch * epochs))
    out_dir = f"./out_{model_name.replace('/','_')}_{tag}_{seed}"
    args = _common_args(out_dir, epochs, lr, bs, weight_decay, warmup_steps, patience, "f1_macro")
    callbacks = [EarlyStoppingCallback(early_stopping_patience=patience)] if patience else []
    trainer = Trainer(model=model, args=args, train_dataset=train_tok, eval_dataset=val_tok,
                       compute_metrics=compute_metrics_multilabel, callbacks=callbacks)
    trainer.train()
    test_metrics = trainer.evaluate(test_tok)
    test_pred = trainer.predict(test_tok)
    probs = 1 / (1 + np.exp(-test_pred.predictions))
    y_pred = (probs >= 0.5).astype(int)
    y_true = test_pred.label_ids.astype(int)

    if os.path.exists(out_dir): shutil.rmtree(out_dir, ignore_errors=True)
    if keep_model:
        return test_metrics, trainer.state.log_history, (model, tokenizer), (y_true, y_pred)
    del model; torch.cuda.empty_cache()
    return test_metrics, trainer.state.log_history, None, (y_true, y_pred)

"""## 8. N-Shot Sampling (separate definitions for Stage A)
"""

def sample_n_shot_binary(df, hate_col, n_per_class, seed):
    parts = [g.sample(min(n_per_class, len(g)), random_state=seed) for _, g in df.groupby(hate_col)]
    sampled = pd.concat(parts).reset_index(drop=True)
    counts = sampled[hate_col].value_counts().sort_index()
    assert (counts == n_per_class).all(), f"Class imbalance in {n_per_class}-shot binary sample: {counts.to_dict()}"
    return sampled

def sample_n_shot_multilabel(hate_df, label_cols, n_per_label, seed):
    parts = [hate_df[hate_df[lbl] == 1].sample(min(n_per_label, (hate_df[lbl] == 1).sum()), random_state=seed)
             for lbl in label_cols]
    sampled = pd.concat(parts).drop_duplicates(subset=["text"]).reset_index(drop=True)
    return sampled

print("Stage A five-shot check:\n", sample_n_shot_binary(train_df, hate_col, 5, SEED)[hate_col].value_counts())
hate_train_df = train_df[train_df[hate_col] == 1].reset_index(drop=True)
print("\nStage B five-shot sample size:", len(sample_n_shot_multilabel(hate_train_df, target_cols, 5, SEED)))

"""## 9. Run Stage A: Binary Hate Detection (five-shot / few-shot / full, single seed = 100)
"""

EPOCHS_CAP = 3
LR = 2e-5

SHOT_SETTINGS_A = [
    ("five-shot", 5,  dict(epochs=EPOCHS_CAP, lr=LR, bs=4,  weight_decay=0.01, label_smoothing_factor=0.1,
                            freeze_base=True, unfreeze_last_n=2, patience=5)),
    ("few-shot",  10, dict(epochs=EPOCHS_CAP, lr=LR, bs=8,  weight_decay=0.01, label_smoothing_factor=0.1,
                            freeze_base=True, unfreeze_last_n=2, patience=5)),
]
FULL_HP_A = dict(epochs=EPOCHS_CAP, lr=LR, bs=32, weight_decay=0.01, label_smoothing_factor=0.05,
                  freeze_base=False, patience=3)
print(f"Stage A: all settings use epochs<={EPOCHS_CAP} and lr={LR}.")

val_ds_A = Dataset.from_pandas(val_df[["text", hate_col]].reset_index(drop=True))
test_ds_A = Dataset.from_pandas(test_df[["text", hate_col]].reset_index(drop=True))

all_results_A = []
raw_predictions_A = {}
loss_curves_A = {}

for name, model_name in MODEL_LIST.items():
    for setting, n_per_class, hp in SHOT_SETTINGS_A:
        shot_df = sample_n_shot_binary(train_df, hate_col, n_per_class, seed=SEED)
        run_ds = Dataset.from_pandas(shot_df[["text", hate_col]].reset_index(drop=True))
        print(f"=== Stage A | {name} | {setting} | seed={SEED} ===")
        m, lh, _, (yt, yp) = finetune_binary(model_name, run_ds, val_ds_A, test_ds_A, hate_col,
                                              tag=f"A_{setting}", seed=SEED, **hp)
        cfg = {"n_per_class": n_per_class, **hp}
        log_experiment(name, "A", setting, SEED, cfg, m)
        all_results_A.append(metrics_row_binary(name, setting, SEED, m, extra=cfg))
        raw_predictions_A[(name, setting)] = (yt, yp)

    run_ds = Dataset.from_pandas(train_df[["text", hate_col]].reset_index(drop=True))
    print(f"=== Stage A | {name} | full | seed={SEED} ===")
    m, lh, _, (yt, yp) = finetune_binary(model_name, run_ds, val_ds_A, test_ds_A, hate_col,
                                          tag="A_full", seed=SEED, **FULL_HP_A)
    cfg = {"n_per_class": None, **FULL_HP_A}
    log_experiment(name, "A", "full", SEED, cfg, m)
    all_results_A.append(metrics_row_binary(name, "full", SEED, m, extra=cfg))
    raw_predictions_A[(name, "full")] = (yt, yp)
    loss_curves_A[name] = lh

all_results_A_df = pd.DataFrame(all_results_A)
all_results_A_df.to_csv("banth_stageA_results.csv", index=False)
all_results_A_df

"""## 12. Graphs -- Stage A (Binary Hate Detection)"""

plt.rcParams.update({"font.family": "serif", "font.size": 12, "axes.linewidth": 1.0,
                      "axes.edgecolor": "black", "figure.dpi": 300})

plot_models_A = compA_f1.index.tolist()
x = np.arange(len(plot_models_A))
width = 0.25
colors = ["#4C72B0", "#DD8452", "#55A868"]

fig, ax = plt.subplots(figsize=(9, 5.5))
for i, setting in enumerate(settings_order):
    ax.bar(x + (i - 1) * width, compA_f1.loc[plot_models_A, setting].values, width,
           label=setting.replace("-", " ").title(), color=colors[i], edgecolor="black", linewidth=0.6)
ax.axhline(majority_baseline_acc, color="black", linestyle="--", linewidth=1.2, label="Dummy Baseline Acc.")
ax.set_xticks(x); ax.set_xticklabels(plot_models_A, rotation=20, ha="right")
ax.set_ylabel("Macro F1-Score"); ax.set_ylim(0, 1)
ax.set_title("Stage A: Macro-F1 Across Models and Learning Settings (BanTH, Binary)")
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
ax.grid(axis="y", linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig("banth_stageA_comparative_f1.pdf", bbox_inches="tight")
plt.savefig("banth_stageA_comparative_f1.png", dpi=300, bbox_inches="tight")
plt.show()

shot_counts_A = {"five-shot": 5 * 2, "few-shot": 20 * 2, "full": len(train_df)}
fig, ax = plt.subplots(figsize=(7.5, 5.5))
for m in plot_models_A:
    ys = [compA_f1.loc[m, s] for s in settings_order]
    ax.plot([shot_counts_A[s] for s in settings_order], ys, marker="o", linewidth=1.8, label=m)
ax.set_xscale("log")
ax.set_xlabel("Number of Training Examples (log scale)"); ax.set_ylabel("Macro F1-Score")
ax.set_title("Stage A Sample Efficiency (BanTH, Binary)")
ax.legend(frameon=False, fontsize=9, loc="lower right")
ax.grid(True, which="both", linestyle=":", alpha=0.5)
plt.tight_layout()
plt.savefig("banth_stageA_sample_efficiency.pdf", bbox_inches="tight")
plt.savefig("banth_stageA_sample_efficiency.png", dpi=300, bbox_inches="tight")
plt.show()

best_A = compA_f1["full"].idxmax()
yt, yp = raw_predictions_A[(best_A, "full")]
cm = confusion_matrix(yt, yp, labels=[0, 1])
fig, ax = plt.subplots(figsize=(4.5, 4))
ax.imshow(cm, cmap="Blues")
for r in range(2):
    for c in range(2):
        ax.text(c, r, str(cm[r, c]), ha="center", va="center",
                color="white" if cm[r, c] > cm.max()/2 else "black", fontsize=12)
ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred Non-Hate", "Pred Hate"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["True Non-Hate", "True Hate"])
ax.set_title(f"Stage A Confusion Matrix: {best_A} (full)")
plt.tight_layout()
plt.savefig("banth_stageA_confusion.pdf", bbox_inches="tight")
plt.savefig("banth_stageA_confusion.png", dpi=300, bbox_inches="tight")
plt.show()

"""## 14. Save Results, Configs, and Logs"""

os.makedirs("results", exist_ok=True)

all_results_A_df.to_csv("results/banth_stageA_results.csv", index=False)
compA_f1.to_csv("results/banth_stageA_f1_macro.csv")
compA_macc.to_csv("results/banth_stageA_macro_accuracy.csv")


with open("results/experiment_log.json", "w") as f:
    json.dump(experiment_log, f, indent=2, default=str)

with open("results/full_state.pkl", "wb") as f:
    pickle.dump({"all_results_A_df": all_results_A_df,
                 "compA_f1": compA_f1, "compA_macc": compA_macc,

                 "loss_curves_A": loss_curves_A,
                 "experiment_log": experiment_log,
                 "target_cols": target_cols, "hate_col": hate_col,
                 "majority_baseline_acc": majority_baseline_acc}, f)

for fig_file in ["banth_cooccurrence.pdf", "banth_cooccurrence.png",
                  "banth_stageA_comparative_f1.pdf", "banth_stageA_comparative_f1.png",
                  "banth_stageA_sample_efficiency.pdf", "banth_stageA_sample_efficiency.png",
                  "banth_stageA_confusion.pdf", "banth_stageA_confusion.png"
                  ]:
    if os.path.exists(fig_file):
        shutil.copy(fig_file, os.path.join("results", fig_file))

shutil.make_archive("banth_results", "zip", "results")
print("All results, configs, and logs saved to results/ and banth_results.zip")

try:
    from google.colab import files
    files.download("banth_results.zip")
except Exception:
    pass