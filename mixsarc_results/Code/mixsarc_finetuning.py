# -*- coding: utf-8 -*-
"""MixSarc_Finetuning

# MixSarc: Bangla-English Code-Mixed Multi-Label Fine-Tuning (Humor / Sarcasm / Offense / Vulgarity)

Dataset: `ajwad-abrar/MixSarc` (Alam, Chowdhury, Ahmed, Abrar, Haque -- IUT / UIC, 2026),
loaded directly from Hugging Face.
"""

!pip -q install -U transformers datasets accelerate scikit-learn sentencepiece pandas numpy matplotlib

import os, re, json, pickle, shutil, random
import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
from datasets import load_dataset, Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                           TrainingArguments, Trainer, EarlyStoppingCallback, set_seed)
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, multilabel_confusion_matrix

SEED = 100
set_seed(SEED)
random.seed(SEED); np.random.seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

experiment_log = []
def log_experiment(model_name, setting, seed, config, metrics):
    experiment_log.append({"model": model_name, "setting": setting, "seed": seed,
                            "config": config, "metrics": metrics})

"""## 1. Load Dataset (from Hugging Face)"""

raw = load_dataset("ajwad-abrar/MixSarc")
print(raw)
df = raw["train"].to_pandas()
print(df.shape)
print(df.columns.tolist())
df.head()

"""## 2. Detect Text Column and the Four Binary Label Columns
"""

text_col = [c for c in df.columns if pd.api.types.is_string_dtype(df[c]) and df[c].astype(str).str.len().mean() > 10][0]
label_cols = [c for c in df.columns if c != text_col and set(pd.Series(df[c]).dropna().unique()).issubset({0, 1})]
print("text_col:", text_col, "| label_cols:", label_cols)

df = df[[text_col] + label_cols].dropna().rename(columns={text_col: "text"}).reset_index(drop=True)
num_labels = len(label_cols)

pos_counts = df[label_cols].sum().sort_values(ascending=False)
pos_pct = (pos_counts / len(df) * 100).round(2)
print(f"\nTotal sentences: {len(df)}")
print(pd.DataFrame({"positive_count": pos_counts, "positive_pct": pos_pct}))

"""## 3. Train / Validation / Test Split (70:15:15)
"""

strat_col = "humorous" if "humorous" in label_cols else pos_counts.index[0]
print("Stratifying on:", strat_col)

train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df[strat_col], random_state=SEED)
val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df[strat_col], random_state=SEED)
print(len(train_df), len(val_df), len(test_df))

print("\nPositive counts per split:")
print(pd.DataFrame({"train": train_df[label_cols].sum(), "val": val_df[label_cols].sum(),
                     "test": test_df[label_cols].sum()}))

train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
val_ds = Dataset.from_pandas(val_df.reset_index(drop=True))
test_ds = Dataset.from_pandas(test_df.reset_index(drop=True))

"""## 4. Multi-Label Metrics
"""

def make_compute_metrics(label_cols, threshold=0.5):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = 1 / (1 + np.exp(-logits))
        preds = (probs >= threshold).astype(int)
        labels = labels.astype(int)

        p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
        p_mic, r_mic, f1_mic, _ = precision_recall_fscore_support(labels, preds, average="micro", zero_division=0)
        subset_acc = float(np.mean(np.all(labels == preds, axis=1)))

        out = {"precision_macro": p_mac, "recall_macro": r_mac, "f1_macro": f1_mac,
               "precision_micro": p_mic, "recall_micro": r_mic, "f1_micro": f1_mic,
               "subset_accuracy": subset_acc, "accuracy": subset_acc}

        p_c, r_c, f1_c, _ = precision_recall_fscore_support(labels, preds, average=None, zero_division=0,
                                                              labels=list(range(len(label_cols))))
        for i, lbl in enumerate(label_cols):
            out[f"f1_{lbl}"] = f1_c[i]
            out[f"precision_{lbl}"] = p_c[i]
            out[f"recall_{lbl}"] = r_c[i]
        return out
    return compute_metrics

compute_metrics = make_compute_metrics(label_cols)

def metrics_row(model_name, setting, seed, m, extra=None):
    row = {"model": model_name, "setting": setting, "seed": seed,
           "accuracy": m["eval_accuracy"], "subset_accuracy": m["eval_subset_accuracy"],
           "precision_macro": m["eval_precision_macro"], "recall_macro": m["eval_recall_macro"], "f1_macro": m["eval_f1_macro"],
           "precision_micro": m["eval_precision_micro"], "recall_micro": m["eval_recall_micro"], "f1_micro": m["eval_f1_micro"]}
    for lbl in label_cols:
        row[f"f1_{lbl}"] = m[f"eval_f1_{lbl}"]
        row[f"precision_{lbl}"] = m[f"eval_precision_{lbl}"]
        row[f"recall_{lbl}"] = m[f"eval_recall_{lbl}"]
    if extra:
        row.update(extra)
    return row

"""## 5. Models (same 6 as the BnSentMix / SentMix-3L notebooks, for cross-dataset comparability)"""

MODEL_LIST = {
    "mBERT": "bert-base-multilingual-cased",
    "XLM-RoBERTa": "xlm-roberta-base",
    "BanglaBERT": "csebuetnlp/banglabert",
    "BanglishBERT": "csebuetnlp/banglishbert",
    "DistilBERT": "distilbert-base-multilingual-cased",
    "BERT": "bert-base-uncased",
}
MAX_LEN = 128

"""## 6. Fine-Tuning Function (multi-label)
"""

def tokenize_ds(ds, tokenizer, label_cols, max_len):
    def _map(ex):
        enc = tokenizer(ex["text"], truncation=True, padding="max_length", max_length=max_len)
        enc["labels"] = [float(ex[c]) for c in label_cols]
        return enc
    return ds.map(_map)

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

def finetune(model_name, train_ds, val_ds, test_ds, label_cols, epochs=5, lr=2e-5, bs=16,
             tag="full", seed=SEED, keep_model=False, weight_decay=0.01,
             freeze_base=False, unfreeze_last_n=2, patience=None):
    set_seed(seed)
    num_labels = len(label_cols)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels, problem_type="multi_label_classification").to(DEVICE)
    if freeze_base:
        model = freeze_base_layers(model, unfreeze_last_n)

    train_tok = tokenize_ds(train_ds, tokenizer, label_cols, MAX_LEN)
    val_tok = tokenize_ds(val_ds, tokenizer, label_cols, MAX_LEN)
    test_tok = tokenize_ds(test_ds, tokenizer, label_cols, MAX_LEN)

    steps_per_epoch = max(1, len(train_tok) // bs)
    warmup_steps = max(1, int(0.1 * steps_per_epoch * epochs))

    out_dir = f"./out_{model_name.replace('/','_')}_{tag}_{seed}"
    args = TrainingArguments(
        output_dir=out_dir, num_train_epochs=epochs,
        per_device_train_batch_size=bs, per_device_eval_batch_size=bs,
        learning_rate=lr, weight_decay=weight_decay, warmup_steps=warmup_steps,
        eval_strategy="epoch", save_strategy="epoch" if patience else "no",
        save_total_limit=1, load_best_model_at_end=bool(patience),
        metric_for_best_model="f1_macro", greater_is_better=True,
        logging_steps=25, report_to="none", seed=seed,
    )
    callbacks = [EarlyStoppingCallback(early_stopping_patience=patience)] if patience else []
    trainer = Trainer(model=model, args=args, train_dataset=train_tok, eval_dataset=val_tok,
                       compute_metrics=compute_metrics, callbacks=callbacks)
    trainer.train()
    test_metrics = trainer.evaluate(test_tok)
    test_pred = trainer.predict(test_tok)
    probs = 1 / (1 + np.exp(-test_pred.predictions))
    y_pred = (probs >= 0.5).astype(int)
    y_true = test_pred.label_ids.astype(int)

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir, ignore_errors=True)

    if keep_model:
        return test_metrics, trainer.state.log_history, (model, tokenizer), (y_true, y_pred)
    del model
    torch.cuda.empty_cache()
    return test_metrics, trainer.state.log_history, None, (y_true, y_pred)

"""## 7. Multi-Label N-Shot Sampling
"""

def sample_n_shot(train_df, label_cols, n_per_label, seed):
    parts = [train_df[train_df[lbl] == 1].sample(min(n_per_label, (train_df[lbl] == 1).sum()), random_state=seed)
             for lbl in label_cols]
    sampled = pd.concat(parts).drop_duplicates(subset=["text"]).reset_index(drop=True)
    return sampled

s5 = sample_n_shot(train_df, label_cols, 5, SEED)
print("five-shot sample size:", len(s5), "\npositive counts:\n", s5[label_cols].sum())
s20 = sample_n_shot(train_df, label_cols, 20, SEED)
print("\nfew-shot sample size:", len(s20), "\npositive counts:\n", s20[label_cols].sum())

"""## 8. Run Five-Shot / Few-Shot / Full Fine-Tuning (single seed = 100)
"""

all_results = []
raw_predictions = {}
loss_curves = {}

EPOCHS_CAP = 10
LR = 2e-5

RUN_SETTINGS = [
    ("five-shot", 5,    dict(epochs=EPOCHS_CAP, lr=LR, bs=4,  weight_decay=0.01,
                              freeze_base=True, unfreeze_last_n=2, patience=5)),
    ("few-shot",  20,   dict(epochs=EPOCHS_CAP, lr=LR, bs=8,  weight_decay=0.01,
                              freeze_base=True, unfreeze_last_n=2, patience=5)),
    ("full",      None, dict(epochs=EPOCHS_CAP, lr=LR, bs=16, weight_decay=0.01,
                              freeze_base=False, patience=3)),
]
print(f"All settings use epochs<={EPOCHS_CAP} and lr={LR} (controlled comparison -- see Section 8 note).")

for name, model_name in MODEL_LIST.items():
    for setting, n_per_label, hp in RUN_SETTINGS:
        if n_per_label is None:
            run_ds = train_ds
            cfg = {"n_per_label": None, **hp}
        else:
            shot_df = sample_n_shot(train_df, label_cols, n_per_label, seed=SEED)
            run_ds = Dataset.from_pandas(shot_df.reset_index(drop=True))
            cfg = {"n_per_label": n_per_label, **hp}

        print(f"=== {name} | {setting} | seed={SEED} ===")
        m, lh, _, (yt, yp) = finetune(model_name, run_ds, val_ds, test_ds, label_cols,
                                       tag=setting, seed=SEED, **hp)
        log_experiment(name, setting, SEED, cfg, m)
        all_results.append(metrics_row(name, setting, SEED, m, extra=cfg))
        raw_predictions[(name, setting)] = (yt, yp)
        if setting == "full":
            loss_curves[name] = lh

all_results_df = pd.DataFrame(all_results)
all_results_df.to_csv("mixsarc_all_results.csv", index=False)
all_results_df

"""## 9. Comparative Tables
"""

settings_order = ["five-shot", "few-shot", "full"]
model_names = list(MODEL_LIST.keys())

comp_f1_macro = all_results_df.pivot(index="model", columns="setting", values="f1_macro").reindex(columns=settings_order)
comp_subset_acc = all_results_df.pivot(index="model", columns="setting", values="subset_accuracy").reindex(columns=settings_order)
comp_f1_macro["gain_full_vs_5shot"] = comp_f1_macro["full"] - comp_f1_macro["five-shot"]
comp_f1_macro = comp_f1_macro.round(4).sort_values("full", ascending=False)
comp_subset_acc = comp_subset_acc.round(4).loc[comp_f1_macro.index]

print("Macro-F1 (across 4 labels):\n", comp_f1_macro, "\n")
print("Subset (exact-match) accuracy:\n", comp_subset_acc)

best_model = comp_f1_macro["full"].idxmax()
print(f"\nBest model (full fine-tuning): {best_model} | macro-F1={comp_f1_macro.loc[best_model,'full']:.4f}")

per_label_full = all_results_df[all_results_df["setting"] == "full"].set_index("model")[[f"f1_{l}" for l in label_cols]]
per_label_full.columns = label_cols
per_label_full = per_label_full.round(4).loc[comp_f1_macro.index]
print("\nPer-label F1 (full fine-tuning) -- directly comparable to the paper's Table 5:\n", per_label_full)

"""## 10. Publication-Quality Comparative Graphs"""

plt.rcParams.update({"font.family": "serif", "font.size": 12, "axes.linewidth": 1.0,
                      "axes.edgecolor": "black", "figure.dpi": 300})

plot_models = comp_f1_macro.index.tolist()
x = np.arange(len(plot_models))
width = 0.25
colors = ["#4C72B0", "#DD8452", "#55A868"]

fig, ax = plt.subplots(figsize=(9, 5.5))
for i, setting in enumerate(settings_order):
    means = comp_f1_macro.loc[plot_models, setting].values
    ax.bar(x + (i - 1) * width, means, width, label=setting.replace("-", " ").title(),
           color=colors[i], edgecolor="black", linewidth=0.6)
ax.set_xticks(x)
ax.set_xticklabels(plot_models, rotation=20, ha="right")
ax.set_ylabel("Macro F1-Score (across 4 labels)")
ax.set_ylim(0, 1)
ax.set_title("Macro-F1 Comparison Across Models and Learning Settings on MixSarc")
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
ax.grid(axis="y", linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig("mixsarc_comparative_f1_macro.pdf", bbox_inches="tight")
plt.savefig("mixsarc_comparative_f1_macro.png", dpi=300, bbox_inches="tight")
plt.show()

shot_counts = {"five-shot": 5 * num_labels, "few-shot": 20 * num_labels, "full": len(train_df)}

fig, ax = plt.subplots(figsize=(7.5, 5.5))
for m in plot_models:
    ys = [comp_f1_macro.loc[m, s] for s in settings_order]
    ax.plot([shot_counts[s] for s in settings_order], ys, marker="o", linewidth=1.8, label=m)
ax.set_xscale("log")
ax.set_xlabel("Approx. Number of Training Examples (log scale)")
ax.set_ylabel("Macro F1-Score")
ax.set_title("Sample Efficiency: Macro-F1 vs. Training Set Size (MixSarc)")
ax.legend(frameon=False, fontsize=9, loc="lower right")
ax.grid(True, which="both", linestyle=":", alpha=0.5)
plt.tight_layout()
plt.savefig("mixsarc_sample_efficiency.pdf", bbox_inches="tight")
plt.savefig("mixsarc_sample_efficiency.png", dpi=300, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(figsize=(7.5, 5.5))
vals = comp_f1_macro[settings_order].values
im = ax.imshow(vals, cmap="Blues", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(settings_order)))
ax.set_xticklabels([s.replace("-", " ").title() for s in settings_order])
ax.set_yticks(range(len(plot_models)))
ax.set_yticklabels(plot_models)
for i in range(len(plot_models)):
    for j in range(len(settings_order)):
        v = vals[i, j]
        ax.text(j, i, f"{v:.3f}", ha="center", va="center", color="white" if v > 0.5 else "black", fontsize=10)
ax.set_title("Macro-F1 Heatmap: Models vs. Learning Settings (MixSarc)")
fig.colorbar(im, ax=ax, label="Macro F1-Score", fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig("mixsarc_f1_heatmap.pdf", bbox_inches="tight")
plt.savefig("mixsarc_f1_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()

"""## 11. Per-Label F1 Breakdown (best full-fine-tuned model)
"""

best_full_model = comp_f1_macro["full"].idxmax()
yt, yp = raw_predictions[(best_full_model, "full")]
p_c, r_c, f1_c, support_c = precision_recall_fscore_support(yt, yp, average=None, zero_division=0,
                                                              labels=list(range(num_labels)))

fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.bar(label_cols, f1_c, color="#4C72B0", edgecolor="black", linewidth=0.6)
for b, s in zip(bars, support_c):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"n={s}", ha="center", fontsize=9)
ax.set_ylim(0, 1)
ax.set_ylabel("F1-Score")
ax.set_title(f"Per-Label F1: {best_full_model} (full fine-tuning) -- cf. paper's Table 5")
ax.grid(axis="y", linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig("mixsarc_per_label_f1.pdf", bbox_inches="tight")
plt.savefig("mixsarc_per_label_f1.png", dpi=300, bbox_inches="tight")
plt.show()

print(pd.DataFrame({"label": label_cols, "precision": p_c, "recall": r_c, "f1": f1_c, "support": support_c}))

"""## 12. Per-Label Confusion Matrices (best full-fine-tuned model)
"""

mcm = multilabel_confusion_matrix(yt, yp, labels=list(range(num_labels)))

fig, axes = plt.subplots(1, num_labels, figsize=(4 * num_labels, 4))
for i, lbl in enumerate(label_cols):
    ax = axes[i] if num_labels > 1 else axes
    cm = mcm[i]
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
    for r in range(2):
        for c in range(2):
            ax.text(c, r, f"{cm[r,c]}", ha="center", va="center",
                    color="white" if cm[r, c] > cm.max() / 2 else "black", fontsize=11)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["True 0", "True 1"])
    ax.set_title(lbl)
plt.suptitle(f"Per-Label Confusion Matrices: {best_full_model} (full fine-tuning)")
plt.tight_layout()
plt.savefig("mixsarc_confusion_matrices.pdf", bbox_inches="tight")
plt.savefig("mixsarc_confusion_matrices.png", dpi=300, bbox_inches="tight")
plt.show()

"""## 13. Class-Imbalance Diagnostic
"""

fig, ax = plt.subplots(figsize=(6.5, 4.5))
order = pos_counts.index.tolist()
bars = ax.bar(order, pos_counts.loc[order].values, color="#DD8452", edgecolor="black", linewidth=0.6)
for b, pct in zip(bars, pos_pct.loc[order].values):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 20, f"{pct:.1f}%", ha="center", fontsize=9)
ax.set_ylabel("Positive Count (out of %d sentences)" % len(df))
ax.set_title("MixSarc Label Imbalance (full dataset)")
ax.grid(axis="y", linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig("mixsarc_label_imbalance.pdf", bbox_inches="tight")
plt.savefig("mixsarc_label_imbalance.png", dpi=300, bbox_inches="tight")
plt.show()

"""## 14. Save Results, Configs, and Logs"""

os.makedirs("results", exist_ok=True)

all_results_df.to_csv("results/mixsarc_all_results.csv", index=False)
comp_f1_macro.to_csv("results/mixsarc_f1_macro_comparison.csv")
comp_subset_acc.to_csv("results/mixsarc_subset_accuracy_comparison.csv")
per_label_full.to_csv("results/mixsarc_per_label_f1_full.csv")

with open("results/experiment_log.json", "w") as f:
    json.dump(experiment_log, f, indent=2, default=str)

with open("results/full_state.pkl", "wb") as f:
    pickle.dump({"all_results_df": all_results_df, "comp_f1_macro": comp_f1_macro,
                 "comp_subset_acc": comp_subset_acc, "per_label_full": per_label_full,
                 "loss_curves": loss_curves, "experiment_log": experiment_log,
                 "label_cols": label_cols}, f)

for fig_file in ["mixsarc_comparative_f1_macro.pdf", "mixsarc_comparative_f1_macro.png",
                  "mixsarc_sample_efficiency.pdf", "mixsarc_sample_efficiency.png",
                  "mixsarc_f1_heatmap.pdf", "mixsarc_f1_heatmap.png",
                  "mixsarc_per_label_f1.pdf", "mixsarc_per_label_f1.png",
                  "mixsarc_confusion_matrices.pdf", "mixsarc_confusion_matrices.png",
                  "mixsarc_label_imbalance.pdf", "mixsarc_label_imbalance.png"]:
    if os.path.exists(fig_file):
        shutil.copy(fig_file, os.path.join("results", fig_file))

shutil.make_archive("mixsarc_results", "zip", "results")
print("All results, configs, and logs saved to results/ and mixsarc_results.zip")

try:
    from google.colab import files
    files.download("mixsarc_results.zip")
except Exception:
    pass