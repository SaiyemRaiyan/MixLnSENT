# -*- coding: utf-8 -*-
"""SentMix3L_Finetuning
# SentMix-3L: Bangla-Hindi-English Code-Mixed Sentiment Fine-Tuning

Dataset: SentMix-3L (Raihan et al., 2023, SEALP Workshop @ IJCNLP-AACL 2023), `sen_1k.csv`
(1,007 instances; 3 labels: Positive 41.71%, Negative 35.05%, Neutral 23.24%).
"""

!pip -q install -U transformers datasets accelerate scikit-learn sentencepiece pandas numpy matplotlib

import os, re, json, pickle, shutil, random
import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                           TrainingArguments, Trainer, EarlyStoppingCallback, set_seed)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

SEED = 100
set_seed(SEED)
random.seed(SEED); np.random.seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

experiment_log = []
def log_experiment(model_name, setting, seed, config, metrics):
    experiment_log.append({"model": model_name, "setting": setting, "seed": seed,
                            "config": config, "metrics": metrics})

"""## 1. Load Dataset
"""

CSV_PATH = "sen_1k.csv"

if not os.path.exists(CSV_PATH):
    try:
        from google.colab import files
        print(f"'{CSV_PATH}' not found in the working directory -- please upload it.")
        uploaded = files.upload()
        CSV_PATH = list(uploaded.keys())[0]
    except ImportError:
        raise FileNotFoundError(f"'{CSV_PATH}' not found. Upload it to the working directory and re-run.")

df = pd.read_csv(CSV_PATH)
print(df.shape)
print(df.columns.tolist())
df.head()

text_col = [c for c in df.columns if pd.api.types.is_string_dtype(df[c]) and df[c].astype(str).str.len().mean() > 15][0]
label_col = [c for c in df.columns if c != text_col][0]
print("text_col:", text_col, "| label_col:", label_col)

df = df[[text_col, label_col]].dropna().rename(columns={text_col: "text"})
_cat = pd.Categorical(df[label_col])
label_names = list(_cat.categories)
df["label"] = _cat.codes
num_labels = len(label_names)
df = df[["text", "label"]].reset_index(drop=True)
print(label_names, num_labels)
print(df["label"].value_counts().sort_index())
print(df["label"].value_counts(normalize=True).sort_index().round(4) * 100)

"""## 2. Train / Validation / Test Split (70:15:15, stratified)
"""

train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=SEED)
val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["label"], random_state=SEED)
print(len(train_df), len(val_df), len(test_df))
print("Train class distribution:\n", train_df["label"].value_counts().sort_index())

train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
val_ds = Dataset.from_pandas(val_df.reset_index(drop=True))
test_ds = Dataset.from_pandas(test_df.reset_index(drop=True))

"""## 3. Metrics
"""

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    p_m, r_m, f1_m, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    return {"accuracy": acc, "precision": p_w, "recall": r_w, "f1": f1_w,
            "precision_macro": p_m, "recall_macro": r_m, "f1_macro": f1_m}

def metrics_row(model_name, setting, seed, m, extra=None):
    row = {"model": model_name, "setting": setting, "seed": seed,
           "accuracy": m["eval_accuracy"], "precision": m["eval_precision"], "recall": m["eval_recall"], "f1": m["eval_f1"],
           "precision_macro": m["eval_precision_macro"], "recall_macro": m["eval_recall_macro"], "f1_macro": m["eval_f1_macro"]}
    if extra:
        row.update(extra)
    return row

"""## 4. Models (same 6 as the BnSentMix notebook, for direct comparability)"""

MODEL_LIST = {
    "mBERT": "bert-base-multilingual-cased",
    "XLM-RoBERTa": "xlm-roberta-base",
    "BanglaBERT": "csebuetnlp/banglabert",
    "BanglishBERT": "csebuetnlp/banglishbert",
    "DistilBERT": "distilbert-base-multilingual-cased",
    "BERT": "bert-base-uncased",
}
MAX_LEN = 256

"""## 5. Fine-Tuning Function
"""

def tokenize_ds(ds, tokenizer):
    return ds.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=MAX_LEN),
                  batched=True)

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

def finetune(model_name, train_ds, val_ds, test_ds, num_labels, epochs=5, lr=2e-5, bs=16,
             tag="full", seed=SEED, keep_model=False, weight_decay=0.01, label_smoothing_factor=0.0,
             freeze_base=False, unfreeze_last_n=2, patience=None):
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(DEVICE)
    if freeze_base:
        model = freeze_base_layers(model, unfreeze_last_n)

    train_tok = tokenize_ds(train_ds, tokenizer)
    val_tok = tokenize_ds(val_ds, tokenizer)
    test_tok = tokenize_ds(test_ds, tokenizer)

    steps_per_epoch = max(1, len(train_tok) // bs)
    warmup_steps = max(1, int(0.1 * steps_per_epoch * epochs))

    out_dir = f"./out_{model_name.replace('/','_')}_{tag}_{seed}"
    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        learning_rate=lr,
        weight_decay=weight_decay,
        label_smoothing_factor=label_smoothing_factor,
        warmup_steps=warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch" if patience else "no",
        save_total_limit=1,
        load_best_model_at_end=bool(patience),
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=25,
        report_to="none",
        seed=seed,
    )
    callbacks = [EarlyStoppingCallback(early_stopping_patience=patience)] if patience else []
    trainer = Trainer(model=model, args=args, train_dataset=train_tok, eval_dataset=val_tok,
                       compute_metrics=compute_metrics, callbacks=callbacks)
    trainer.train()
    test_metrics = trainer.evaluate(test_tok)
    test_pred = trainer.predict(test_tok)
    y_true = test_pred.label_ids
    y_pred = np.argmax(test_pred.predictions, axis=-1)

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir, ignore_errors=True)

    if keep_model:
        return test_metrics, trainer.state.log_history, (model, tokenizer), (y_true, y_pred)
    del model
    torch.cuda.empty_cache()
    return test_metrics, trainer.state.log_history, None, (y_true, y_pred)

"""## 6. Stratified N-Shot Sampling
"""

def sample_n_shot(train_df, n_per_class, seed):
    parts = [g.sample(min(n_per_class, len(g)), random_state=seed) for _, g in train_df.groupby("label")]
    sampled = pd.concat(parts).reset_index(drop=True)
    counts = sampled["label"].value_counts().sort_index()
    assert (counts == n_per_class).all(), f"Class imbalance in {n_per_class}-shot sample: {counts.to_dict()}"
    return sampled

print("five-shot check:\n", sample_n_shot(train_df, 5, SEED)["label"].value_counts().sort_index())
print("few-shot check:\n", sample_n_shot(train_df, 20, SEED)["label"].value_counts().sort_index())

"""## 7. Run Five-Shot / Few-Shot / Full Fine-Tuning (single seed = 100)
"""

all_results = []
raw_predictions = {}
loss_curves = {}

EPOCHS_CAP = 10
LR = 2e-5

RUN_SETTINGS = [
    ("five-shot", 5,    dict(epochs=EPOCHS_CAP, lr=LR, bs=4,  weight_decay=0.01, label_smoothing_factor=0.1,
                              freeze_base=True, unfreeze_last_n=2, patience=5)),
    ("few-shot",  20,   dict(epochs=EPOCHS_CAP, lr=LR, bs=8,  weight_decay=0.01, label_smoothing_factor=0.1,
                              freeze_base=True, unfreeze_last_n=2, patience=5)),
    ("full",      None, dict(epochs=EPOCHS_CAP, lr=LR, bs=16, weight_decay=0.01, label_smoothing_factor=0.05,
                              freeze_base=False, patience=3)),
]
print(f"All settings use epochs<={EPOCHS_CAP} and lr={LR} (controlled comparison -- see Section 7 note).")

for name, model_name in MODEL_LIST.items():
    for setting, n_per_class, hp in RUN_SETTINGS:
        if n_per_class is None:
            run_ds = train_ds
            cfg = {"n_per_class": None, **hp}
        else:
            shot_df = sample_n_shot(train_df, n_per_class, seed=SEED)
            run_ds = Dataset.from_pandas(shot_df.reset_index(drop=True))
            cfg = {"n_per_class": n_per_class, **hp}

        print(f"=== {name} | {setting} | seed={SEED} ===")
        m, lh, _, (yt, yp) = finetune(model_name, run_ds, val_ds, test_ds, num_labels,
                                       tag=setting, seed=SEED, **hp)
        log_experiment(name, setting, SEED, cfg, m)
        all_results.append(metrics_row(name, setting, SEED, m, extra=cfg))
        raw_predictions[(name, setting)] = (yt, yp)
        if setting == "full":
            loss_curves[name] = lh

all_results_df = pd.DataFrame(all_results)
all_results_df.to_csv("sentmix3l_all_results.csv", index=False)
all_results_df

"""## 8. Comparative Tables (macro-F1 primary)
"""

settings_order = ["five-shot", "few-shot", "full"]
model_names = list(MODEL_LIST.keys())

comp_f1_macro = all_results_df.pivot(index="model", columns="setting", values="f1_macro").reindex(columns=settings_order)
comp_acc = all_results_df.pivot(index="model", columns="setting", values="accuracy").reindex(columns=settings_order)
comp_f1_macro["gain_full_vs_5shot"] = comp_f1_macro["full"] - comp_f1_macro["five-shot"]
comp_f1_macro = comp_f1_macro.round(4).sort_values("full", ascending=False)
comp_acc = comp_acc.round(4).loc[comp_f1_macro.index]

print("Macro-F1:\n", comp_f1_macro, "\n")
print("Accuracy:\n", comp_acc)

best_model = comp_f1_macro["full"].idxmax()
print(f"\nBest model (full fine-tuning): {best_model} | macro-F1={comp_f1_macro.loc[best_model,'full']:.4f}")
best_5shot = comp_f1_macro["five-shot"].idxmax()
print(f"Best five-shot model: {best_5shot} | macro-F1={comp_f1_macro.loc[best_5shot,'five-shot']:.4f}")
print("\nPaper's own best models on SentMix-3L (synthetic-train/natural-test) were MuRIL and XLM-R (0.77")
print("F1), with BanglaBERT/HindiBERT weakest -- since this dataset is trilingual, not just Bangla-English,")
print("models without Hindi coverage (BanglaBERT, BanglishBERT) are expected to lag XLM-R/mBERT here too.")

"""## 9. Comparative Graphs
"""

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
ax.set_ylabel("Macro F1-Score")
ax.set_ylim(0, 1)
ax.set_title("Macro-F1 Comparison Across Models and Learning Settings on SentMix-3L")
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
ax.grid(axis="y", linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig("sentmix3l_comparative_f1_macro.pdf", bbox_inches="tight")
plt.savefig("sentmix3l_comparative_f1_macro.png", dpi=300, bbox_inches="tight")
plt.show()

shot_counts = {"five-shot": 5 * num_labels, "few-shot": 20 * num_labels, "full": len(train_df)}

fig, ax = plt.subplots(figsize=(7.5, 5.5))
for m in plot_models:
    ys = [comp_f1_macro.loc[m, s] for s in settings_order]
    ax.plot([shot_counts[s] for s in settings_order], ys, marker="o", linewidth=1.8, label=m)
ax.set_xscale("log")
ax.set_xlabel("Number of Training Examples (log scale)")
ax.set_ylabel("Macro F1-Score")
ax.set_title("Sample Efficiency: Macro-F1 vs. Training Set Size (SentMix-3L)")
ax.legend(frameon=False, fontsize=9, loc="lower right")
ax.grid(True, which="both", linestyle=":", alpha=0.5)
plt.tight_layout()
plt.savefig("sentmix3l_sample_efficiency.pdf", bbox_inches="tight")
plt.savefig("sentmix3l_sample_efficiency.png", dpi=300, bbox_inches="tight")
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
        ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                color="white" if v > 0.5 else "black", fontsize=10)
ax.set_title("Macro-F1 Heatmap: Models vs. Learning Settings (SentMix-3L)")
fig.colorbar(im, ax=ax, label="Macro F1-Score", fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig("sentmix3l_f1_heatmap.pdf", bbox_inches="tight")
plt.savefig("sentmix3l_f1_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()

"""## 10. Confusion Matrix Grid (every model x every setting)
"""

fig, axes = plt.subplots(len(model_names), len(settings_order), figsize=(10, 2.6 * len(model_names)))
for i, name in enumerate(model_names):
    for j, setting in enumerate(settings_order):
        ax = axes[i, j]
        yt, yp = raw_predictions[(name, setting)]
        cm = confusion_matrix(yt, yp, labels=list(range(num_labels)))
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
        ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        for r in range(num_labels):
            for c in range(num_labels):
                ax.text(c, r, f"{cm[r,c]}", ha="center", va="center", fontsize=7,
                        color="white" if cm_norm[r, c] > 0.5 else "black")
        if i == 0:
            ax.set_title(setting.replace("-", " ").title(), fontsize=10)
        if j == 0:
            ax.set_ylabel(name, fontsize=9)
        ax.set_xticks(range(num_labels)); ax.set_xticklabels(label_names, fontsize=6, rotation=45)
        ax.set_yticks(range(num_labels)); ax.set_yticklabels(label_names, fontsize=6)
plt.suptitle("Confusion Matrices: Every Model x Every Learning Setting (SentMix-3L, counts shown, color = row-normalized)", y=1.001)
plt.tight_layout()
plt.savefig("sentmix3l_confusion_grid.pdf", bbox_inches="tight")
plt.savefig("sentmix3l_confusion_grid.png", dpi=300, bbox_inches="tight")
plt.show()

"""## 11. Per-Class F1 Breakdown (best full-fine-tuned model)
"""

best_full_model = comp_f1_macro["full"].idxmax()
yt, yp = raw_predictions[(best_full_model, "full")]
p_c, r_c, f1_c, support_c = precision_recall_fscore_support(yt, yp, labels=list(range(num_labels)), zero_division=0)

fig, ax = plt.subplots(figsize=(6.5, 4.5))
bars = ax.bar(label_names, f1_c, color="#4C72B0", edgecolor="black", linewidth=0.6)
for b, s in zip(bars, support_c):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"n={s}", ha="center", fontsize=9)
ax.set_ylim(0, 1)
ax.set_ylabel("F1-Score")
ax.set_title(f"Per-Class F1: {best_full_model} (full fine-tuning)")
ax.grid(axis="y", linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig("sentmix3l_per_class_f1.pdf", bbox_inches="tight")
plt.savefig("sentmix3l_per_class_f1.png", dpi=300, bbox_inches="tight")
plt.show()

print(pd.DataFrame({"label": label_names, "precision": p_c, "recall": r_c, "f1": f1_c, "support": support_c}))

"""## 12. Save Results, Configs, and Logs"""

os.makedirs("results", exist_ok=True)

all_results_df.to_csv("results/sentmix3l_all_results.csv", index=False)
comp_f1_macro.to_csv("results/sentmix3l_f1_macro_comparison.csv")
comp_acc.to_csv("results/sentmix3l_accuracy_comparison.csv")

with open("results/experiment_log.json", "w") as f:
    json.dump(experiment_log, f, indent=2, default=str)

with open("results/full_state.pkl", "wb") as f:
    pickle.dump({"all_results_df": all_results_df, "comp_f1_macro": comp_f1_macro, "comp_acc": comp_acc,
                 "loss_curves": loss_curves, "experiment_log": experiment_log}, f)

for fig_file in ["sentmix3l_comparative_f1_macro.pdf", "sentmix3l_comparative_f1_macro.png",
                  "sentmix3l_sample_efficiency.pdf", "sentmix3l_sample_efficiency.png",
                  "sentmix3l_f1_heatmap.pdf", "sentmix3l_f1_heatmap.png",
                  "sentmix3l_confusion_grid.pdf", "sentmix3l_confusion_grid.png",
                  "sentmix3l_per_class_f1.pdf", "sentmix3l_per_class_f1.png"]:
    if os.path.exists(fig_file):
        shutil.copy(fig_file, os.path.join("results", fig_file))

shutil.make_archive("sentmix3l_results", "zip", "results")
print("All results, configs, and logs saved to results/ and sentmix3l_results.zip")

try:
    from google.colab import files
    files.download("sentmix3l_results.zip")
except Exception:
    pass